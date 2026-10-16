# Owner(s): ["module: sparse"]
# Native gfx1013 scalar hipSPARSELt integration contracts.
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
    path = os.environ.get("BC250_A04_RESULT_PATH")
    if path:
        Path(path).write_text(json.dumps(ROWS, indent=2) + "\n")


def sparse_input(m, k, dtype):
    group = 2 if dtype == torch.float32 else 4
    kept = group // 2
    values = torch.randint(-3, 4, (m, k)).to(dtype)
    mask = torch.zeros(m, k // group, group, dtype=torch.bool)
    start = torch.arange(m * (k // group)).reshape(m, k // group) % group
    for i in range(kept):
        mask.scatter_(2, ((start + i) % group).unsqueeze(-1), True)
    return values.masked_fill(~mask.reshape(m, k), 0)


class TestBC250SparseLt(TestCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1013"
        )
        torch.manual_seed(250404)

    @dtypes(torch.float32, torch.float16, torch.bfloat16, torch.int8)
    @parametrize("geometry", [(7, 9, 12), (32, 16, 32)])
    @parametrize("column_major", [False, True])
    def test_pack_mm(self, device, dtype, geometry, column_major):
        m, n, k = geometry
        cpu_a = sparse_input(m, k, dtype)
        cpu_b = torch.randint(-3, 4, (k, n)).to(dtype)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = cpu_a.to(device)
            b = cpu_b.to(device)
            if column_major:
                b = b.t().contiguous().t()
            packed = torch._cslt_compress(a)
            # Decode the native value/index format independently of the GEMM.
            raw = packed.view(torch.uint8).flatten().cpu()
            group = 2 if dtype == torch.float32 else 4
            kept = group // 2
            value_bytes = m * k // 2 * a.element_size()
            values = raw[:value_bytes].view(dtype).reshape(m, k // group, kept)
            metadata = (
                raw[value_bytes : value_bytes + m * k // group]
                .reshape(m, k // group)
                .long()
            )
            decoded = torch.zeros(m, k // group, group, dtype=dtype)
            for i in range(kept):
                index = (metadata >> (2 * i)) & 3
                self.assertTrue(bool((index < group).all()))
                decoded.scatter_(2, index.unsqueeze(-1), values[:, :, i : i + 1])
            self.assertEqual(decoded.reshape(m, k), cpu_a, atol=0, rtol=0)
            product = cpu_a.double() @ cpu_b.double()
            executions = 0
            for epilogue in ["plain", "scalar_gpu", "vector", "typed_bias"]:
                if epilogue == "typed_bias" and dtype == torch.int8:
                    continue
                bias_cpu = torch.arange(m).float() / 8 - 1
                kwargs = {}
                expected = product
                if epilogue == "scalar_gpu":
                    kwargs = dict(
                        alpha=torch.tensor(-0.75, device=device),
                        bias=bias_cpu.to(device),
                    )
                    expected = -0.75 * product + bias_cpu.double()[:, None]
                elif epilogue == "vector":
                    alpha_cpu = (torch.arange(m) % 4 + 1).float() / 4
                    kwargs = dict(alpha=alpha_cpu.to(device), bias=bias_cpu.to(device))
                    expected = (
                        alpha_cpu.double()[:, None] * product
                        + bias_cpu.double()[:, None]
                    )
                elif epilogue == "typed_bias":
                    typed_bias = bias_cpu.to(dtype)
                    kwargs = dict(bias=typed_bias.to(device))
                    expected = product + typed_bias.double()[:, None]
                if dtype == torch.int8:
                    expected = expected.round().clamp(-128, 127).to(dtype)
                else:
                    expected = expected.to(dtype)
                for transpose in [False, True]:
                    actual = torch._cslt_sparse_mm(
                        packed, b, transpose_result=transpose, **kwargs
                    )
                    self.assertEqual(
                        actual.cpu(), expected.t() if transpose else expected
                    )
                    executions += 1
            self.assertEqual(a.cpu(), cpu_a, atol=0, rtol=0)
            self.assertEqual(b.cpu(), cpu_b, atol=0, rtol=0)
        record(
            self._testMethodName,
            dtype=str(dtype),
            geometry=geometry,
            column_major=column_major,
            matmul_executions=executions,
            metadata_roundtrip=True,
        )

    @dtypes(torch.float16, torch.bfloat16)
    def test_public_wrapper(self, device, dtype):
        cpu_a = sparse_input(32, 32, dtype)
        cpu_b = torch.randn(32, 16).to(dtype)
        a = torch.sparse.to_sparse_semi_structured(cpu_a.to(device))
        result = torch.mm(a, cpu_b.to(device))
        self.assertEqual(result.cpu(), (cpu_a.double() @ cpu_b.double()).to(dtype))
        record(self._testMethodName, dtype=str(dtype))

    def test_search(self, device):
        a = sparse_input(7, 12, torch.float32).to(device)
        b = torch.randn(12, 9, device=device)
        packed = torch._cslt_compress(a)
        self.assertTrue(torch.backends.cusparselt.is_available())
        self.assertEqual(torch.backends.cusparselt.version(), 209)
        algorithm, split_k, split_mode, count = torch._C._cusparselt.mm_search(
            packed, b, None, None, None, False
        )
        self.assertEqual((algorithm, split_k, split_mode, count), (0, 1, 0, 1))
        with self.assertWarnsRegex(UserWarning, "deprecated"):
            algorithm = torch._cslt_sparse_mm_search(packed, b)
        self.assertEqual(algorithm, 0)
        self.assertEqual(
            torch._cslt_sparse_mm(packed, b, alg_id=algorithm).cpu(),
            (a.cpu().double() @ b.cpu().double()).float(),
        )
        record(self._testMethodName, algorithm=algorithm)

    def test_contracts(self, device):
        a = sparse_input(7, 12, torch.float32).to(device)
        b = torch.ones(12, 9, device=device)
        packed = torch._cslt_compress(a)
        bad = [
            lambda: torch._cslt_compress(a.t()),
            lambda: torch._cslt_compress(torch.empty(0, 12, device=device)),
            lambda: torch._cslt_sparse_mm(packed.flatten(), b),
            lambda: torch._cslt_sparse_mm(packed, b.half()),
            lambda: torch._cslt_sparse_mm(packed, torch.ones(14, 9, device=device)),
            lambda: torch._cslt_sparse_mm(
                packed, torch.ones(12, 18, device=device)[:, ::2]
            ),
            lambda: torch._cslt_sparse_mm(packed, b, alpha=torch.ones(7)),
            lambda: torch._cslt_sparse_mm(packed, b, alpha=torch.tensor(1.0)),
            lambda: torch._cslt_sparse_mm(
                packed, b, alpha=torch.ones(7, device=device, dtype=torch.float16)
            ),
            lambda: torch._cslt_sparse_mm(packed, b, bias=torch.ones(6, device=device)),
            lambda: torch._cslt_sparse_mm(packed, b, bias=torch.ones(7)),
            lambda: torch._cslt_sparse_mm(packed, b, split_k=2),
            lambda: torch._cslt_sparse_mm(packed, b, alg_id=5),
        ]
        for call in bad:
            with self.assertRaises(RuntimeError):
                call()
        q = torch._cslt_compress(a.to(torch.int8))
        with self.assertRaisesRegex(RuntimeError, "same-dtype"):
            torch._cslt_sparse_mm(q, b.to(torch.int8), out_dtype=torch.int32)
        self.assertEqual(
            torch._cslt_sparse_mm(packed, b).cpu(),
            (a.cpu().double() @ b.cpu().double()).float(),
        )
        record(self._testMethodName, refusal_contracts=len(bad) + 1)


instantiate_device_type_tests(TestBC250SparseLt, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
