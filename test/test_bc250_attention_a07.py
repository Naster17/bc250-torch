# Owner(s): ["module: nn"]
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.utils.checkpoint import checkpoint


if os.environ.get("BC250_A07_CANDIDATE"):
    spec = importlib.util.spec_from_file_location(
        "attention_d256_candidate", os.environ["BC250_A07_CANDIDATE"]
    )
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
else:
    from torch.nn.attention import _gfx1013 as ops
ROWS = []


def record(name, **data):
    ROWS.append(
        dict(
            name=name,
            status="passed",
            module=ops.__file__,
            source_sha256=hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
            **data,
        )
    )
    Path(os.environ["BC250_A07_ROWS"]).write_text(json.dumps(ROWS, indent=2) + "\n")


def reference(q, k, v, mask=None, causal=False, scale=None):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    score = (q @ k.transpose(-1, -2)) * (
        q.shape[-1] ** -0.5 if scale is None else scale
    )
    if causal:
        allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        score = score.masked_fill(~allowed, -float("inf"))
    if mask is not None:
        score = (
            score.masked_fill(~mask, -float("inf"))
            if mask.dtype == torch.bool
            else score + mask
        )
    empty = (score == -float("inf")).all(-1, keepdim=True)
    prob = score.masked_fill(empty, 0).softmax(-1).masked_fill(empty, 0)
    return prob @ v


class TestAttentionD256(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(70717)
        torch.set_num_threads(4)
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1013"
        )

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize(
        "case",
        [
            (2, 4, 2, 17, 19, 256, "none", None),
            (1, 2, 1, 33, 41, 256, "causal", None),
            (1, 2, 1, 17, 33, 256, "bool", 0.13),
            (1, 4, 2, 19, 23, 256, "additive", -0.2),
            (1, 2, 1, 5, 7, 129, "none", 0.0),
            (1, 2, 1, 19, 3, 256, "causal", None),
            (1, 2, 1, 3, 19, 255, "causal", None),
        ],
    )
    def test_cpu_reference(self, device, dtype, case):
        b, h, hk, m, n, d, mode, scale = case
        xs = [
            (torch.randn(b, heads, length, d * 2, device=device, dtype=dtype) * 0.2)[
                ..., ::2
            ]
            .detach()
            .requires_grad_()
            for heads, length in [(h, m), (hk, n), (hk, n)]
        ]
        cpu = [x.detach().cpu().double().requires_grad_() for x in xs]
        mask = None
        if mode == "bool":
            mask = torch.rand(b, 1, m, n, device=device) > 0.25
            mask[:, :, 0] = False
            mask[:, :, 1, :16] = False
        elif mode == "additive":
            mask = torch.randn(m, n, device=device) * 0.1
            mask[0] = -float("inf")
        kwargs = dict(
            attn_mask=mask, is_causal=mode == "causal", scale=scale, enable_gqa=h != hk
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            y = ops.scaled_dot_product_attention(*xs, **kwargs)
            g = torch.randn_like(y) * 0.1
            actual = torch.autograd.grad(y, xs, g)
        torch.cuda.current_stream().wait_stream(stream)
        ref = reference(
            *cpu, None if mask is None else mask.cpu(), mode == "causal", scale
        )
        wanted = torch.autograd.grad(ref, cpu, g.cpu().double())
        tol = {
            torch.float32: (3e-5, 5e-4),
            torch.float16: (0.003, 0.008),
            torch.bfloat16: (0.03, 0.05),
        }[dtype]
        relative = []
        for a, r in [(y, ref), *zip(actual, wanted)]:
            self.assertEqual(a.cpu().double(), r, atol=tol[0], rtol=tol[1])
            error = float((a.cpu().double() - r).norm() / r.norm().clamp_min(1e-20))
            self.assertLess(
                error,
                {torch.float32: 5e-6, torch.float16: 8e-4, torch.bfloat16: 6e-3}[dtype],
            )
            relative.append(error)
        record(self._testMethodName, case=case, output_qkv_relative_l2=relative)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("mixed", [False, True])
    def test_autocast(self, device, dtype, mixed):
        xs = [
            torch.randn(
                1,
                heads,
                17,
                256,
                device=device,
                dtype=torch.float16 if mixed and i == 1 else torch.float32,
            ).requires_grad_()
            for i, heads in enumerate((4, 2, 2))
        ]
        refs = [x.detach().clone().requires_grad_() for x in xs]
        g = torch.randn(1, 4, 17, 256, device=device, dtype=dtype) * 0.1
        with torch.autocast("cuda", dtype=dtype):
            y = ops.scaled_dot_product_attention(*xs, is_causal=True, enable_gqa=True)
            with sdpa_kernel(SDPBackend.MATH):
                ref = F.scaled_dot_product_attention(
                    *refs, is_causal=True, enable_gqa=True
                )
        grads = torch.autograd.grad(y, xs, g)
        wanted = torch.autograd.grad(ref, refs, g)
        self.assertEqual(y.dtype, dtype)
        for a, b in [(y, ref), *zip(grads, wanted)]:
            self.assertEqual(
                a,
                b,
                atol=0.03 if dtype == torch.bfloat16 else 0.003,
                rtol=0.05 if dtype == torch.bfloat16 else 0.008,
            )
        record(self._testMethodName, mixed=mixed)

    @dtypes(torch.float32, torch.bfloat16)
    def test_checkpoint(self, device, dtype):
        xs = [
            torch.randn(1, heads, 33, 256, device=device, dtype=dtype).requires_grad_()
            for heads in (4, 2, 2)
        ]

        def fn(*values):
            return ops.scaled_dot_product_attention(
                *values, is_causal=True, enable_gqa=True
            )

        y = fn(*xs)
        g = torch.randn_like(y)
        expected = torch.autograd.grad(y, xs, g)
        z = checkpoint(fn, *xs, use_reentrant=False)
        actual = torch.autograd.grad(z, xs, g)
        self.assertEqual(y, z, atol=0, rtol=0)
        for a, b in zip(actual, expected):
            self.assertEqual(a, b, atol=0, rtol=0)
        record(self._testMethodName)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_empty(self, device, dtype):
        for m, n in ((3, 0), (0, 3), (0, 0)):
            xs = [
                torch.randn(
                    1, heads, length, 256, device=device, dtype=dtype
                ).requires_grad_()
                for heads, length in ((2, m), (1, n), (1, n))
            ]
            y = ops.scaled_dot_product_attention(*xs, enable_gqa=True)
            gradients = torch.autograd.grad(y.sum(), xs)
            self.assertEqual(y, torch.zeros_like(y), atol=0, rtol=0)
            for g in gradients:
                self.assertEqual(g, torch.zeros_like(g), atol=0, rtol=0)
        record(self._testMethodName, empty_geometries=3)

    @dtypes(torch.bfloat16)
    def test_long_geometry(self, device, dtype):
        xs = [
            (
                torch.randn(1, heads, 1024, 256, device=device, dtype=dtype) * 0.2
            ).requires_grad_()
            for heads in (8, 2, 2)
        ]
        refs = [x.detach().clone().requires_grad_() for x in xs]
        y = ops.scaled_dot_product_attention(*xs, is_causal=True, enable_gqa=True)
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(*refs, is_causal=True, enable_gqa=True)
        g = torch.randn_like(y) * 0.1
        actual = torch.autograd.grad(y, xs, g)
        expected = torch.autograd.grad(ref, refs, g)
        errors = []
        for a, b in [(y, ref), *zip(actual, expected)]:
            self.assertEqual(a, b, atol=0.03, rtol=0.05)
            error = float(
                (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)
            )
            self.assertLess(error, 0.006)
            errors.append(error)
        record(
            self._testMethodName,
            output_qkv_relative_l2=errors,
            shape=[1, 8, 2, 1024, 256],
        )

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    def test_inference(self, device, dtype):
        xs = [
            torch.randn(1, heads, 17, 256, device=device, dtype=dtype).requires_grad_()
            for heads in (4, 2, 2)
        ]
        cpu = [x.detach().cpu().double() for x in xs]
        mask = torch.ones(17, 17, device=device, dtype=torch.bool)
        mask[0] = False
        with (
            torch.no_grad(),
            mock.patch.object(
                F,
                "scaled_dot_product_attention",
                side_effect=AssertionError("Inference must stay tiled"),
            ),
        ):
            y = ops.scaled_dot_product_attention(*xs, attn_mask=mask, enable_gqa=True)
        ref = reference(*cpu, mask.cpu())
        self.assertEqual(
            y.cpu().double(),
            ref,
            atol=0.03
            if dtype == torch.bfloat16
            else 0.003
            if dtype == torch.float16
            else 3e-5,
            rtol=0.05
            if dtype == torch.bfloat16
            else 0.008
            if dtype == torch.float16
            else 5e-4,
        )
        self.assertEqual(y.requires_grad, False)
        record(self._testMethodName, fully_tiled_inference=True)

    @dtypes(torch.bfloat16)
    def test_hybrid_forward(self, device, dtype):
        xs = [
            torch.randn(1, heads, 17, 256, device=device, dtype=dtype).requires_grad_()
            for heads in (4, 2, 2)
        ]
        with mock.patch.object(
            F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
        ) as math_forward:
            y = ops.scaled_dot_product_attention(*xs, is_causal=True, enable_gqa=True)
            grads = torch.autograd.grad(y.sum(), xs)
        self.assertEqual(math_forward.call_count, 1)
        self.assertTrue(
            all(x.dtype == torch.float32 for x in math_forward.call_args.args)
        )
        self.assertTrue(all(bool(g.isfinite().all()) for g in grads))
        record(self._testMethodName, reference_forward_calls=1)

    @dtypes(torch.float32)
    def test_refusals(self, device, dtype):
        xs = [
            torch.randn(1, 2, 3, 256, device=device, requires_grad=True)
            for _ in range(3)
        ]
        with self.assertRaises(ValueError):
            ops.scaled_dot_product_attention(
                *(torch.randn(1, 2, 3, 257, device=device) for _ in range(3))
            )
        with self.assertRaises(NotImplementedError):
            ops.scaled_dot_product_attention(*xs, dropout_p=0.1)
        with self.assertRaises(NotImplementedError):
            ops.scaled_dot_product_attention(
                *xs, attn_mask=torch.zeros(3, 3, device=device, requires_grad=True)
            )
        with self.assertRaises(ValueError):
            ops.scaled_dot_product_attention(
                *xs,
                attn_mask=torch.ones(3, 3, device=device, dtype=torch.bool),
                is_causal=True,
            )
        record(self._testMethodName, refusals=4)


instantiate_device_type_tests(TestAttentionD256, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()
