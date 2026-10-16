# Owner(s): ["module: sparse"]
# Native gfx1013 first-order structured sparse gradients and bounded timings.
import json
import os
from pathlib import Path

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


ROWS = []


def record(name, **data):
    ROWS.append(dict(name=name, status="passed", **data))
    Path(os.environ["BC250_SPARSE_ROWS"]).write_text(json.dumps(ROWS, indent=2) + "\n")


def structured(m, k, dtype):
    group = 2 if dtype == torch.float32 else 4
    mask = torch.zeros(m, k // group, group, dtype=torch.bool)
    start = torch.arange(m * (k // group)).view(m, k // group) % group
    for i in range(group // 2):
        mask.scatter_(2, ((start + i) % group).unsqueeze(-1), True)
    mask = mask.view(m, k)
    values = (torch.randint(1, 4, (m, k)) * 0.125).to(dtype)
    return values * mask, mask


class TestSparseAutograd(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(251016)
        torch.set_num_threads(2)
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1013"
        )

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("transpose", [False, True])
    @parametrize("alpha_mode", ["none", "scalar", "vector"])
    def test_raw(self, device, dtype, transpose, alpha_mode):
        a, mask = structured(7, 12, dtype)
        b = (torch.randn(12, 9) * 0.2).to(dtype)
        gpu_a = a.to(device).requires_grad_()
        gpu_b = b.to(device).t().contiguous().t().detach().requires_grad_()
        bias = torch.linspace(-0.2, 0.3, 7, device=device, requires_grad=True)
        alpha = (
            None
            if alpha_mode == "none"
            else torch.tensor(-0.75, device=device, requires_grad=True)
            if alpha_mode == "scalar"
            else torch.linspace(0.25, 1, 7, device=device, requires_grad=True)
        )
        kwargs = dict(bias=bias, transpose_result=transpose)
        if alpha is not None:
            kwargs["alpha"] = alpha
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            packed = torch._cslt_compress(gpu_a)
            y = torch._cslt_sparse_mm(packed, gpu_b, **kwargs)
            g = torch.linspace(
                -0.25, 0.5, y.numel(), device=device, dtype=dtype
            ).view_as(y)
            inputs = [gpu_a, gpu_b, bias] + ([] if alpha is None else [alpha])
            actual = torch.autograd.grad(y, inputs, g)
        torch.cuda.current_stream().wait_stream(stream)
        ca, cb = a.double().requires_grad_(), b.double().requires_grad_()
        cbi = bias.detach().cpu().double().requires_grad_()
        calpha = (
            None if alpha is None else alpha.detach().cpu().double().requires_grad_()
        )
        ref = (ca * mask) @ cb
        if calpha is not None:
            ref = ref * (calpha if calpha.numel() == 1 else calpha[:, None])
        ref = (ref + cbi[:, None]).to(dtype).double()
        if transpose:
            ref = ref.t()
        refs = torch.autograd.grad(
            ref, [ca, cb, cbi] + ([] if calpha is None else [calpha]), g.cpu().double()
        )
        tol = {
            torch.float32: (2e-6, 4e-5),
            torch.float16: (0.002, 0.006),
            torch.bfloat16: (0.015, 0.04),
        }[dtype]
        self.assertEqual(y.cpu().double(), ref, atol=tol[0], rtol=tol[1])
        for result, expected in zip(actual, refs):
            self.assertEqual(result.cpu().double(), expected, atol=tol[0], rtol=tol[1])
        self.assertEqual(gpu_a.grad, None)
        record(self._testMethodName, dtype=str(dtype), all_gradients=True)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("trainable_weight", [False, True])
    @parametrize("linear", [False, True])
    @parametrize("rows", [32, 64])
    def test_public(self, device, dtype, trainable_weight, linear, rows):
        a, mask = structured(rows, 32, dtype)
        w = a.to(device).requires_grad_(trainable_weight)
        x = (torch.randn(16, 32, device=device, dtype=dtype) * 0.2).requires_grad_()
        sparse = torch.sparse.to_sparse_semi_structured(w)
        bias = torch.linspace(
            -0.2, 0.2, rows, device=device, dtype=dtype, requires_grad=True
        )
        y = (
            torch.nn.functional.linear(x, sparse, bias)
            if linear
            else torch.mm(sparse, x.t()).t()
        )
        y.square().float().mean().backward()
        cw = a.double().requires_grad_(trainable_weight)
        cx = x.detach().cpu().double().requires_grad_()
        cbias = bias.detach().cpu().double().requires_grad_()
        ref = cx @ (cw * mask).t()
        if linear:
            ref = ref + cbias
        ref = ref.to(dtype)
        ref.square().float().mean().backward()
        tol = 0.003 if dtype == torch.bfloat16 else 0.0005
        self.assertEqual(y.cpu(), ref, atol=tol, rtol=0.04)
        self.assertEqual(x.grad.cpu().double(), cx.grad, atol=tol, rtol=0.04)
        if linear:
            self.assertEqual(bias.grad.cpu().double(), cbias.grad, atol=tol, rtol=0.04)
        if trainable_weight:
            self.assertIsNotNone(w.grad)
            self.assertEqual(w.grad.cpu().double(), cw.grad, atol=tol, rtol=0.04)
            self.assertEqual(
                w.grad.cpu()[~mask],
                torch.zeros_like(w.grad.cpu()[~mask]),
                atol=0,
                rtol=0,
            )
        record(self._testMethodName, source_weight_gradient=trainable_weight)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_gradient_subsets(self, device, dtype):
        a, mask = structured(7, 12, dtype)
        b = (torch.randn(12, 9) * 0.2).to(dtype)
        for requested in ["a", "b", "bias", "alpha", "ab"]:
            ga = a.to(device).requires_grad_(
                "a" in requested and requested != "bias" and requested != "alpha"
            )
            gb = b.to(device).requires_grad_(requested in ("b", "ab"))
            bias = (
                torch.zeros(7, device=device, requires_grad=True)
                if requested == "bias"
                else None
            )
            alpha = (
                torch.tensor(0.5, device=device, requires_grad=True)
                if requested == "alpha"
                else None
            )
            actual = torch._cslt_sparse_mm(
                torch._cslt_compress(ga), gb, bias=bias, alpha=alpha
            )
            actual.sum().backward()
            ca = a.double().requires_grad_(ga.requires_grad)
            cb = b.double().requires_grad_(gb.requires_grad)
            cbi = (
                None if bias is None else bias.detach().cpu().double().requires_grad_()
            )
            calpha = (
                None
                if alpha is None
                else alpha.detach().cpu().double().requires_grad_()
            )
            expected = (ca * mask) @ cb
            if calpha is not None:
                expected = expected * calpha
            if cbi is not None:
                expected = expected + cbi[:, None]
            expected.to(dtype).sum().backward()
            tol = (
                0.02
                if dtype == torch.bfloat16
                else 0.003
                if dtype == torch.float16
                else 3e-6
            )
            for gpu, cpu in zip([ga, gb, bias, alpha], [ca, cb, cbi, calpha]):
                if gpu is not None and gpu.requires_grad:
                    self.assertEqual(
                        gpu.grad.cpu().double(), cpu.grad, atol=tol, rtol=0.04
                    )
        record(self._testMethodName, independent_gradient_subsets=5)

    def test_retained_zero(self, device):
        # Empty groups still retain deterministic slots; they must receive gradients.
        a = torch.zeros(7, 12, device=device, requires_grad=True)
        b = torch.ones(12, 9, device=device, requires_grad=True)
        packed = torch._cslt_compress(a)
        torch._cslt_sparse_mm(packed, b).sum().backward()
        self.assertEqual(a.grad[:, ::2], torch.full((7, 6), 9.0, device=device))
        self.assertEqual(a.grad[:, 1::2], torch.zeros(7, 6, device=device))
        self.assertEqual(b.grad, torch.zeros_like(b))
        record(self._testMethodName)


instantiate_device_type_tests(TestSparseAutograd, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
