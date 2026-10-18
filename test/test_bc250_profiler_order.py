# Owner(s): ["oncall: profiler"]
"""Fresh-process ROCm profiler ordering regressions; no external profiler required."""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import run_tests, TestCase


CHECK = TestCase()


def worker(order, output, expect_missing=False):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", order=order, captures=[], references=[])

    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def capture(label, gpu, fn):
        activities = [ProfilerActivity.CPU]
        if gpu:
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, profile_memory=True) as prof:
            value = fn()
        path = output / (label + ".json")
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        copies = [e for e in events if e.get("cat") == "gpu_memcpy"]
        runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "hip_runtime")]
        CHECK.assertTrue(any(e.get("cat") == "cpu_op" for e in events))
        entry = dict(
            label=label,
            gpu_requested=gpu,
            kernels=len(kernels),
            copies=len(copies),
            runtime=len(runtime),
        )
        if gpu and (not expect_missing):
            CHECK.assertTrue(kernels and len(copies) >= 2, entry)
            CHECK.assertTrue(
                all(math.isfinite(e["dur"]) and e["dur"] > 0 for e in kernels + copies)
            )
            kernel_ids = {e.get("args", {}).get("correlation") for e in kernels}
            copy_ids = {e.get("args", {}).get("correlation") for e in copies}
            runtime_ids = {e.get("args", {}).get("correlation") for e in runtime}
            CHECK.assertTrue((kernel_ids & runtime_ids) - {None}, entry)
            CHECK.assertTrue((copy_ids & runtime_ids) - {None}, entry)
            CHECK.assertTrue(any(e.kernels for e in prof.events()))
            entry["correlated_kernel_and_copy"] = True
        elif gpu:
            CHECK.assertTrue(not kernels and (not copies), entry)
            entry["expected_baseline_defect_reproduced"] = True
        else:
            CHECK.assertTrue(not kernels and (not copies), entry)
        report["captures"].append(entry)
        save()
        return value

    def cpu():
        value = torch.arange(33).sum().item()
        CHECK.assertTrue(value == 528)
        return value

    if order == "masked":
        CHECK.assertTrue(
            not torch.cuda.is_available() and (not torch.cuda.is_initialized())
        )
        capture("masked-cpu", False, cpu)
        CHECK.assertTrue(not torch.cuda.is_initialized())
        report.update(status="passed", no_visible_device=True)
        save()
        return
    if order == "cpu-cold":
        CHECK.assertTrue(not torch.cuda.is_initialized())
        capture("cold-cpu", False, cpu)
        CHECK.assertTrue(
            not torch.cuda.is_initialized(),
            "CPU capture must not run Torch CUDA lazy init",
        )
    CHECK.assertTrue(
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    )
    torch.manual_seed(190912)
    a = (torch.randn(32, 48, device="cuda") * 0.1).requires_grad_()
    b = (torch.randn(48, 24, device="cuda") * 0.1).requires_grad_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    host = torch.arange(65536, dtype=torch.float32).pin_memory()
    target = torch.empty_like(host, device="cuda")
    back = torch.empty_like(host, pin_memory=True)

    def work():
        with torch.cuda.stream(stream):
            y = (a @ b).sin()
            grad = torch.autograd.grad(y.square().mean(), (a, b))
            target.copy_(host, non_blocking=True)
            back.copy_(target, non_blocking=True)
        stream.synchronize()
        torch.testing.assert_close(back, host, rtol=0, atol=0)
        return (y, grad)

    plain, plain_grad = work()
    ca, cb = [x.detach().cpu().double().requires_grad_() for x in (a, b)]
    cy = (ca @ cb).sin()
    cg = torch.autograd.grad(cy.square().mean(), (ca, cb))
    torch.testing.assert_close(plain.cpu().double(), cy, atol=2e-06, rtol=0.0002)
    for actual, expected in zip(plain_grad, cg):
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=2e-06, rtol=0.0002
        )
    if order == "cpu-first":
        capture("first-cpu", False, cpu)
    for i in range(2):
        y, gradients = capture(f"gpu-{i}", True, work)
        torch.testing.assert_close(y, plain, rtol=0, atol=0)
        for actual, expected in zip(gradients, plain_grad):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        capture(f"cpu-after-{i}", False, work)
    report["references"].append(
        "exact profiled/unprofiled output+both gradients and pinned copy roundtrip; CPU float64 references; nondefault stream"
    )
    report.update(status="passed", expected_baseline_defect=expect_missing)
    save()


class TestBC250ProfilerOrder(TestCase):
    def test_fresh_process_orders(self):
        if torch.version.hip is None or not torch.cuda.is_available():
            self.skipTest("Requires native ROCm device")
        for order in ("cpu-first", "gpu-first", "cpu-cold", "masked"):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as output:
                env = dict(os.environ)
                if order == "masked":
                    env["HIP_VISIBLE_DEVICES"] = "-1"
                    env["ROCR_VISIBLE_DEVICES"] = "-1"
                subprocess.run(
                    [sys.executable, __file__, "--worker", order, "--output", output],
                    env=env,
                    check=True,
                    timeout=90,
                )


if __name__ == "__main__":
    if "--worker" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--worker",
            choices=("cpu-first", "gpu-first", "cpu-cold", "masked"),
            required=True,
        )
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--expect-missing", action="store_true")
        args = parser.parse_args()
        worker(args.worker, args.output, args.expect_missing)
    else:
        run_tests()
