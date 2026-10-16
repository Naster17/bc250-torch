# Owner(s): ["module: nn"]
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


if os.environ.get("BC250_A06_INSTALLED"):
    from torch.nn import bc250_gated_delta as ops
else:
    spec = importlib.util.spec_from_file_location(
        "gated_delta_candidate", Path(__file__).with_name("bc250_gated_delta.py")
    )
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
ROWS = []


def record(name, **data):
    ROWS.append(
        dict(
            name=name,
            status="passed",
            source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
            module=ops.__file__,
            **data,
        )
    )
    Path(os.environ["BC250_A06_ROWS"]).write_text(json.dumps(ROWS, indent=2) + "\n")


def reference_prepare(q, k, v, g, beta, normalize):
    b, t, h, dk = q.shape
    q, k, v, g, beta = [x.transpose(1, 2).contiguous() for x in (q, k, v, g, beta)]
    if normalize:
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    q = q * dk**-0.5
    pad = (-t) % 64
    q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    g, beta = [F.pad(x, (0, pad)) for x in (g, beta)]
    shape = (b, h, -1, 64)
    return (
        q.reshape(*shape, dk),
        k.reshape(*shape, dk),
        (k * beta[..., None]).reshape(*shape, dk),
        (v * beta[..., None]).reshape(*shape, v.shape[-1]),
        g.reshape(shape).cumsum(-1),
    )


def recurrent_reference(q, k, v, g, beta, normalize, initial=None):
    if normalize:
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    q = q * q.shape[-1] ** -0.5
    b, t, h, dk = q.shape
    state = (
        torch.zeros((b, h, dk, v.shape[-1]), dtype=q.dtype, device=q.device)
        if initial is None
        else initial
    )
    outputs = []
    for i in range(t):
        state = state * g[:, i].exp()[..., None, None]
        error = v[:, i] - (k[:, i, :, :, None] * state).sum(-2)
        state = (
            state + k[:, i, :, :, None] * (beta[:, i, :, None] * error)[..., None, :]
        )
        outputs.append((q[:, i, :, :, None] * state).sum(-2))
    return torch.stack(outputs, 1), state


def inputs(b, t, h, dk, dv, dtype, device):
    q, k = [
        torch.randn(b, t, h, dk, dtype=dtype, device=device) * 0.15 for _ in range(2)
    ]
    v = torch.randn(b, t, h, dv, dtype=dtype, device=device) * 0.2
    g = -(torch.rand(b, t, h, device=device) * 0.15 + 0.01)
    beta = torch.rand(b, t, h, device=device) * 0.8 + 0.1
    return [x.detach().requires_grad_() for x in (q, k, v, g, beta)]


class TestGatedDelta(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(171006)
        torch.set_num_threads(2)
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1013"
        )

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("normalize", [False, True])
    @parametrize(
        "geometry", [(2, 7, 3, 7, 11), (1, 65, 2, 32, 16), (1, 129, 2, 128, 128)]
    )
    def test_prepare(self, device, dtype, normalize, geometry):
        xs = inputs(*geometry, dtype, device)
        # Noncontiguous BLHD/BLH inputs exercise GPU copy autograd edges.
        xs = [
            x.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_()
            for x in xs
        ]
        refs = [x.detach().cpu().double().requires_grad_() for x in xs]
        expected = reference_prepare(*refs, normalize)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ys = ops.prepare_gated_delta(*xs, normalize=normalize)
            grads = [torch.randn_like(y) * 0.2 for y in ys]
            actual = torch.autograd.grad(ys, xs, grads)
        torch.cuda.current_stream().wait_stream(stream)
        wanted = torch.autograd.grad(expected, refs, [g.cpu().double() for g in grads])
        for y, ref in zip(ys, expected):
            self.assertEqual(y.cpu().double(), ref, atol=2e-5, rtol=2e-5)
        tol = {
            torch.float32: (3e-5, 6e-5),
            torch.float16: (0.003, 0.005),
            torch.bfloat16: (0.025, 0.03),
        }[dtype]
        for grad, ref in zip(actual, wanted):
            self.assertEqual(grad.cpu().double(), ref, atol=tol[0], rtol=tol[1])
        record(self._testMethodName, all_five_gradients=True)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("normalize", [False, True])
    @parametrize("length", [1, 65, 129])
    def test_recurrent(self, device, dtype, normalize, length):
        xs = inputs(1, length, 2, 7, 9, dtype, device)
        state = (torch.randn(1, 2, 7, 9, device=device) * 0.03).requires_grad_()
        refs = [x.detach().cpu().double().requires_grad_() for x in (*xs, state)]
        y, s = ops.chunk_gated_delta_rule(
            *xs,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=normalize,
        )
        yr, sr = recurrent_reference(*refs[:5], normalize, refs[5])
        gy, gs = torch.randn_like(y) * 0.1, torch.randn_like(s) * 0.1
        actual = torch.autograd.grad((y, s), (*xs, state), (gy, gs))
        expected = torch.autograd.grad(
            (yr, sr), refs, (gy.cpu().double(), gs.cpu().double())
        )
        tol = {
            torch.float32: (2e-5, 2e-4),
            torch.float16: (0.001, 0.008),
            torch.bfloat16: (0.006, 0.04),
        }[dtype]
        for a, r in [(y, yr), (s, sr), *zip(actual, expected)]:
            self.assertEqual(a.cpu().double(), r, atol=tol[0], rtol=tol[1])
        record(self._testMethodName, cpu_recurrent_reference=True, state_gradient=True)

    @dtypes(torch.float32)
    def test_zero_and_subsets(self, device, dtype):
        for index in range(5):
            xs = inputs(1, 7, 2, 7, 9, dtype, device)
            with torch.no_grad():
                xs[0].zero_()
                xs[1].zero_()
                xs[3].zero_()
                xs[4].zero_()
            refs = [x.detach().cpu().double().requires_grad_() for x in xs]
            ys = ops.prepare_gated_delta(*xs, normalize=True)
            expected = reference_prepare(*refs, True)
            actual = torch.autograd.grad(ys[index].sum(), xs, allow_unused=True)
            wanted = torch.autograd.grad(expected[index].sum(), refs, allow_unused=True)
            for x, a, r in zip(xs, actual, wanted):
                self.assertEqual(
                    torch.zeros_like(x).cpu().double()
                    if a is None
                    else a.cpu().double(),
                    torch.zeros_like(x).cpu().double() if r is None else r,
                    atol=2e-5,
                    rtol=2e-5,
                )
        record(
            self._testMethodName, absent_output_grads=True, zero_norms_and_gates=True
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_low_precision_gates(self, device, dtype):
        xs = inputs(2, 64, 3, 32, 7, dtype, device)
        xs = [x.detach().to(dtype).requires_grad_() for x in xs]
        refs = [x.detach().cpu().double().requires_grad_() for x in xs]
        ys = ops.prepare_gated_delta(*xs, normalize=True)
        expected = reference_prepare(*refs, True)
        grads = [torch.randn_like(y) * 0.1 for y in ys]
        actual = torch.autograd.grad(ys, xs, grads)
        wanted = torch.autograd.grad(expected, refs, [g.cpu().double() for g in grads])
        for y, ref in zip(ys, expected):
            self.assertEqual(y.cpu().double(), ref, atol=2e-5, rtol=2e-5)
        tol = (0.003, 0.005) if dtype == torch.float16 else (0.025, 0.03)
        for grad, ref in zip(actual, wanted):
            self.assertEqual(grad.cpu().double(), ref, atol=tol[0], rtol=tol[1])
        record(self._testMethodName, exact_chunk=True, low_precision_gates=True)

    @dtypes(torch.float32)
    def test_refusals(self, device, dtype):
        xs = inputs(1, 7, 2, 7, 9, dtype, device)
        cases = [
            [xs[0].cpu(), *xs[1:]],
            [xs[0].half(), *xs[1:]],
            [xs[0], xs[1][:, :, :, :3], *xs[2:]],
            [*xs[:3], xs[3].long(), xs[4]],
            [*xs[:4], xs[4][:, :2]],
        ]
        for case in cases:
            with self.assertRaises((ValueError, TypeError)):
                ops.prepare_gated_delta(*case)
        with self.assertRaises(ValueError):
            ops.prepare_gated_delta(*inputs(1, 1, 1, 129, 8, dtype, device))
        with self.assertRaises(ValueError):
            ops.chunk_gated_delta_rule(*xs, chunk_size=32)
        with self.assertRaises(ValueError):
            ops.chunk_gated_delta_rule(
                *xs,
                initial_state=torch.zeros(
                    1, 2, 7, 9, device=device, dtype=torch.float16
                ),
            )
        record(self._testMethodName, refusals=8)


instantiate_device_type_tests(TestGatedDelta, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()
