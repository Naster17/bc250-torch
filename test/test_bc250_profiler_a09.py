# Owner(s): ["oncall: profiler"]
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


ROWS = []
OUTPUT = Path(os.environ["BC250_A09_OUTPUT"])
OUTPUT.mkdir(exist_ok=True)


def record(name, **data):
    ROWS.append(dict(name=name, status="passed", **data))
    (OUTPUT / "rows.json").write_text(json.dumps(ROWS, indent=2) + "\n")


def trace_summary(prof, path):
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "hip_runtime")]
    return events, kernels, runtime


class TestBC250Profiler(TestCase):
    def setUp(self):
        super().setUp()
        self.assertTrue(torch.autograd.kineto_available())
        self.assertIn(ProfilerActivity.CUDA, torch.profiler.supported_activities())
        self.assertIn("gfx1013", torch.cuda.get_device_properties(0).gcnArchName)
        torch.manual_seed(1909)

    @dtypes(torch.float32, torch.bfloat16)
    def test_trace_and_gradients(self, device, dtype):
        a = (torch.randn(64, 96, device=device) * 0.1).to(dtype).requires_grad_()
        b = (torch.randn(96, 48, device=device) * 0.1).to(dtype).requires_grad_()
        ca, cb = [x.detach().cpu().double().requires_grad_() for x in (a, b)]

        def work(x, y):
            z = (x @ y).sin()
            gradients = torch.autograd.grad(z.square().mean(), (x, y))
            return z, gradients

        expected, expected_gradients = work(ca, cb)
        plain, plain_gradients = work(a, b)
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with record_function("bc250_a09_reference_work"):
                actual, gradients = work(a, b)
                # Keep the asynchronous work inside the host range for correlation.
                torch.cuda.synchronize()
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.0002
        self.assertEqual(actual, plain, atol=0, rtol=0)
        for got, original, cpu in zip(gradients, plain_gradients, expected_gradients):
            self.assertEqual(got, original, atol=0, rtol=0)
            self.assertEqual(got.cpu().double(), cpu, atol=2e-6, rtol=tolerance)
        self.assertEqual(actual.cpu().double(), expected, atol=0.002, rtol=tolerance)
        events, kernels, runtime = trace_summary(
            prof, OUTPUT / (self._testMethodName + ".json")
        )
        self.assertTrue(kernels)
        self.assertTrue(runtime)
        self.assertTrue(all(math.isfinite(e["dur"]) and e["dur"] > 0 for e in kernels))
        markers = [
            e
            for e in events
            if e.get("name") == "bc250_a09_reference_work"
            and e.get("cat") == "user_annotation"
        ]
        self.assertEqual(len(markers), 1)
        marker = markers[0]
        self.assertTrue(
            all(
                marker["ts"] - 1000 <= e["ts"] <= marker["ts"] + marker["dur"] + 1000
                for e in kernels
            )
        )
        linked = [e for e in prof.events() if e.kernels]
        self.assertTrue(linked)
        kernel_ids = {e.get("args", {}).get("correlation") for e in kernels}
        runtime_ids = {e.get("args", {}).get("correlation") for e in runtime}
        self.assertTrue((kernel_ids & runtime_ids) - {None})
        record(
            self._testMethodName,
            kernels=len(kernels),
            runtime_events=len(runtime),
            linked_cpu_events=len(linked),
            exact_profiled_output_and_gradients=True,
            cpu_reference=True,
        )

    def test_memory(self, device):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with record_function("bc250_a09_allocation"):
                x = torch.ones(1024, 1024, device=device)
                self.assertEqual(x.sum().item(), 1024 * 1024)
                del x
                torch.cuda.synchronize()
        events, kernels, _ = trace_summary(prof, OUTPUT / "memory.json")
        memory = [e for e in events if e.get("name") == "[memory]"]
        self.assertTrue(memory)
        deltas = [e.get("args", {}).get("Bytes", 0) for e in memory]
        self.assertTrue(any(v >= 4 * 1024 * 1024 for v in deltas))
        self.assertTrue(any(v <= -4 * 1024 * 1024 for v in deltas))
        self.assertTrue(kernels)
        self.assertEqual(torch.cuda.memory_allocated(), before)
        record(
            self._testMethodName,
            allocation_and_free_recorded=True,
            allocation_bytes=4 * 1024 * 1024,
            memory_events=len(memory),
        )

    def test_schedule_and_stream(self, device):
        captures = []
        x = torch.randn(32, 32, device=device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def ready(prof):
            _, kernels, _ = trace_summary(
                prof, OUTPUT / f"scheduled-{len(captures)}.json"
            )
            self.assertTrue(kernels)
            captures.append(len(kernels))

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=2, repeat=2),
            on_trace_ready=ready,
        ) as prof:
            for _ in range(8):
                with torch.cuda.stream(stream):
                    y = x.sin() + 1
                stream.synchronize()
                self.assertEqual(y.cpu(), x.cpu().sin() + 1, atol=2e-6, rtol=2e-6)
                prof.step()
        self.assertEqual(len(captures), 2)
        record(self._testMethodName, captures=captures, nondefault_stream=True)

    def test_copies(self, device):
        host = torch.arange(65536, dtype=torch.float32).pin_memory()
        target = torch.empty_like(host, device=device)
        result = torch.empty_like(host, pin_memory=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            target.copy_(host, non_blocking=True)
            result.copy_(target, non_blocking=True)
            torch.cuda.synchronize()
        self.assertEqual(result, host, atol=0, rtol=0)
        events, _, runtime = trace_summary(prof, OUTPUT / "copies.json")
        copies = [e for e in events if e.get("cat") == "gpu_memcpy"]
        self.assertGreaterEqual(len(copies), 2)
        self.assertTrue(all(math.isfinite(e["dur"]) and e["dur"] > 0 for e in copies))
        copy_ids = {e.get("args", {}).get("correlation") for e in copies}
        runtime_ids = {e.get("args", {}).get("correlation") for e in runtime}
        self.assertTrue((copy_ids & runtime_ids) - {None})
        record(self._testMethodName, gpu_copies=len(copies), exact_roundtrip=True)

    def test_cpu_only(self, device):
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("bc250_a09_cpu_only"):
                value = torch.arange(33).sum().item()
        self.assertEqual(value, 528)
        _, kernels, _ = trace_summary(prof, OUTPUT / "cpu-only.json")
        self.assertFalse(kernels)
        self.assertTrue(prof.events())
        record(self._testMethodName, no_gpu_events=True)

    def test_overhead(self, device):
        x = torch.randn(128, 128, device=device)

        def work():
            for _ in range(5):
                y = x @ x
                y.sin_()
            torch.cuda.synchronize()

        work()
        plain, traced = [], []
        for _ in range(3):
            start = time.perf_counter()
            work()
            plain.append(time.perf_counter() - start)
            start = time.perf_counter()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                work()
            traced.append(time.perf_counter() - start)
            self.assertTrue(any(e.kernels for e in prof.events()))
        self.assertTrue(all(math.isfinite(x) and x > 0 for x in plain + traced))
        record(
            self._testMethodName,
            plain_seconds=plain,
            profiled_seconds=traced,
            median_wall_ratio=statistics.median(traced) / statistics.median(plain),
            scope="Three short eager windows including profiler setup/teardown; diagnostic overhead, not throughput or hardware counter qualification",
        )


instantiate_device_type_tests(TestBC250Profiler, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()
