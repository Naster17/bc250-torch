# Owner(s): ["module: sparse"]
"""N2.13 bounded public sparse triangular solve matrix and dense CPU references."""

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


CHECK = TestCase()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--group",
        choices=("matrix", "edge", "extended", "unit", "trace"),
        default="matrix",
    )
    args = ap.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(1013213)
    CHECK.assertTrue(
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1013"
    )
    report = dict(
        status="running", group=args.group, checks=[], torch_path=torch.__file__
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def check(name, fn):
        start = time.monotonic()
        try:
            result = fn()
            torch.cuda.synchronize()
            report["checks"].append(
                dict(
                    name=name,
                    status="passed",
                    seconds=time.monotonic() - start,
                    **result,
                )
            )
        except Exception as e:
            report["checks"].append(
                dict(
                    name=name,
                    status="failed",
                    error=str(e),
                    seconds=time.monotonic() - start,
                )
            )
        save()

    def run(dtype, layout, index, nrhs, upper, unit, transpose, n_override=None):
        n = n_override if n_override is not None else (18 if layout == "bsr" else 17)
        block = 3 if layout == "bsr" and n_override is not None else 2
        raw = torch.randn(n, n, dtype=dtype) * 0.025
        # Poison the ignored triangle, including entries inside diagonal blocks.
        raw += (
            torch.tril(torch.ones_like(raw), -1)
            if upper
            else torch.triu(torch.ones_like(raw), 1)
        ) * 7
        raw.diagonal().fill_(5 if unit else 2)
        dense = (torch.triu(raw) if upper else torch.tril(raw)).to(
            torch.complex128 if dtype.is_complex else torch.float64
        )
        if unit:
            dense.diagonal().fill_(1)
        matrix = dense.mT if transpose else dense
        sparse = (
            raw.to_sparse_bsr((block, block))
            if layout == "bsr"
            else raw.to_sparse_csr()
        )
        values = sparse.values().cuda()
        value_layout = "row"
        if layout == "bsr" and transpose:
            values = values.mT.contiguous().mT
            value_layout = "column"
        constructor = (
            torch.sparse_bsr_tensor if layout == "bsr" else torch.sparse_csr_tensor
        )
        a = constructor(
            sparse.crow_indices().to(index).cuda(),
            sparse.col_indices().to(index).cuda(),
            values,
            size=raw.shape,
            check_invariants=True,
        )
        saved = [x.clone() for x in (a.crow_indices(), a.col_indices(), a.values())]
        known = torch.randn(n, nrhs, dtype=matrix.dtype) * 0.2
        rhs = (matrix @ known).to(dtype)
        ref = torch.linalg.solve_triangular(
            matrix, rhs.to(matrix.dtype), upper=upper != transpose, unitriangular=unit
        )
        variant = (int(upper) + int(unit) + int(transpose)) % 3
        backing = torch.full(
            (2 * n + 2, 2 * nrhs + 3),
            911 - 237j if dtype.is_complex else 911,
            dtype=dtype,
            device="cuda",
        )
        x = backing[1 : 2 * n + 1 : 2, 1 : 2 * nrhs + 1 : 2]
        mask = torch.zeros_like(backing, dtype=torch.bool)
        mask[1 : 2 * n + 1 : 2, 1 : 2 * nrhs + 1 : 2] = True
        untouched = backing[~mask].clone()
        if variant == 0:
            b = rhs.cuda()
        elif variant == 1:
            b = rhs.mT.contiguous().cuda().mT
        else:
            b = torch.empty((2 * n, 2 * nrhs), dtype=dtype, device="cuda")[::2, ::2]
            b.copy_(rhs.cuda())
        before = b.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = torch.triangular_solve(
                b,
                a,
                upper=upper,
                transpose=transpose,
                unitriangular=unit,
                out=(x, torch.empty(0, dtype=dtype, device="cuda")),
            )
        stream.synchronize()
        CHECK.assertTrue(
            result.solution.data_ptr() == x.data_ptr()
            and result.cloned_coefficient.numel() == 0
        )
        tol = 3e-5 if dtype in (torch.float32, torch.complex64) else 2e-12
        torch.testing.assert_close(x.cpu().to(matrix.dtype), ref, atol=tol, rtol=tol)
        residual = float(
            (matrix @ x.cpu().to(matrix.dtype) - rhs.to(matrix.dtype)).norm()
            / (
                matrix.norm() * x.cpu().to(matrix.dtype).norm()
                + rhs.to(matrix.dtype).norm()
            )
        )
        CHECK.assertTrue(residual < tol, residual)
        torch.testing.assert_close(b, before, rtol=0, atol=0)
        torch.testing.assert_close(backing[~mask], untouched, rtol=0, atol=0)
        for original, old in zip(
            (a.crow_indices(), a.col_indices(), a.values()), saved
        ):
            torch.testing.assert_close(original, old, rtol=0, atol=0)
        return dict(
            residual=residual,
            order=n,
            nrhs=nrhs,
            index_dtype=str(index),
            format=layout,
            value_layout=value_layout,
            rhs_layout=variant,
            output_strides=list(x.stride()),
            nondefault_stream=True,
            input_and_output_guards=True,
        )

    if args.group == "matrix":
        for dtype, layout, index, nrhs, upper, unit, transpose in itertools.product(
            (torch.float32, torch.float64, torch.complex64, torch.complex128),
            ("csr", "bsr"),
            (torch.int32, torch.int64),
            (1, 3),
            (False, True),
            (False, True),
            (False, True),
        ):
            name = f"{dtype}_{layout}_{index}_rhs{nrhs}_upper{upper}_unit{unit}_trans{transpose}"
            check(name, lambda: run(dtype, layout, index, nrhs, upper, unit, transpose))
    elif args.group == "extended":
        for dtype, layout, nrhs, size in itertools.product(
            (torch.float32, torch.float64, torch.complex64, torch.complex128),
            ("csr", "bsr"),
            (1, 3),
            (65, 257),
        ):
            n = size if layout == "csr" else size + 1
            check(
                f"{dtype}_{layout}_rhs{nrhs}_n{n}",
                lambda: run(
                    dtype,
                    layout,
                    torch.int64,
                    nrhs,
                    nrhs == 3,
                    nrhs == 1,
                    size == 257,
                    n,
                ),
            )
    elif args.group == "unit":
        for dtype, layout, nrhs, upper, trans in itertools.product(
            (torch.float32, torch.float64, torch.complex64, torch.complex128),
            ("csr", "bsr"),
            (1, 3),
            (False, True),
            (False, True),
        ):

            def unit(dtype=dtype, layout=layout, nrhs=nrhs, upper=upper, trans=trans):
                dense = torch.zeros(6, 6, dtype=dtype)
                dense[0, 2] = 0.1 if upper else 0
                dense[2, 0] = 0 if upper else 0.1
                a = (
                    dense.to_sparse_bsr((2, 2))
                    if layout == "bsr"
                    else dense.to_sparse_csr()
                )
                matrix = dense.to(
                    torch.complex128 if dtype.is_complex else torch.float64
                )
                matrix.diagonal().fill_(1)
                if trans:
                    matrix = matrix.mT
                rhs = torch.randn(6, nrhs, dtype=dtype)
                expected = torch.linalg.solve_triangular(
                    matrix,
                    rhs.to(matrix.dtype),
                    upper=upper != trans,
                    unitriangular=True,
                )
                actual = torch.triangular_solve(
                    rhs.cuda(),
                    a.cuda(),
                    upper=upper,
                    transpose=trans,
                    unitriangular=True,
                ).solution
                tol = 3e-5 if dtype in (torch.float32, torch.complex64) else 2e-12
                torch.testing.assert_close(
                    actual.cpu().to(matrix.dtype), expected, atol=tol, rtol=tol
                )
                return dict(implicit_unit_diagonal=True, order=6, nrhs=nrhs)

            check(
                f"{dtype}_{layout}_rhs{nrhs}_upper{upper}_trans{trans}_implicit-unit",
                unit,
            )
    elif args.group == "trace":
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            result = run(torch.complex64, "csr", torch.int64, 3, False, False, False)
            torch.cuda.synchronize()
        trace = args.output.with_name("trace.json")
        prof.export_chrome_trace(str(trace))
        kernels = [
            e["name"]
            for e in json.loads(trace.read_text())["traceEvents"]
            if e.get("cat") == "kernel"
            and "sparse_triangular_pivot_mask_kernel" in e.get("name", "")
        ]
        CHECK.assertEqual(len(kernels), 1)
        report["checks"].append(
            dict(name="native-pivot-mask", status="passed", kernels=kernels, **result)
        )
    else:
        for dtype, layout, nrhs, kind in itertools.product(
            (torch.float32, torch.float64, torch.complex64, torch.complex128),
            ("csr", "bsr"),
            (1, 3),
            ("missing", "zero", "missing-block", "empty", "empty-unit"),
        ):

            def edge(dtype=dtype, layout=layout, nrhs=nrhs, kind=kind):
                dense = torch.eye(6, dtype=dtype) * 2
                dense[2, 0] = 0.1
                if kind == "missing":
                    dense[3, 3] = 0
                if kind == "zero":
                    dense[3, 3] = 0
                if kind == "missing-block":
                    dense[2:4, 2:4] = 0
                if kind.startswith("empty"):
                    dense.zero_()
                a = (
                    dense.to_sparse_bsr((2, 2))
                    if layout == "bsr"
                    else dense.to_sparse_csr()
                )
                if kind == "zero" and layout == "csr":
                    # Retain an explicitly stored numerical zero diagonal.
                    a = (
                        dense
                        + torch.diag(torch.tensor([0, 0, 0, 1, 0, 0], dtype=dtype))
                    ).to_sparse_csr()
                    row = a.crow_indices()
                    cols = a.col_indices()
                    vals = a.values().clone()
                    vals[
                        (torch.arange(len(cols)) >= row[3])
                        & (torch.arange(len(cols)) < row[4])
                        & (cols == 3)
                    ] = 0
                    a = torch.sparse_csr_tensor(
                        row, cols, vals, size=(6, 6), check_invariants=True
                    )
                ga = a.cuda()
                old_indices = [
                    z.clone()
                    for z in (ga.crow_indices(), ga.col_indices(), ga.values())
                ]
                backing = torch.full(
                    (14, 2 * nrhs + 3),
                    711 - 29j if dtype.is_complex else 711,
                    dtype=dtype,
                    device="cuda",
                )
                out = backing[1:13:2, 1 : 2 * nrhs + 1 : 2]
                mask = torch.zeros_like(backing, dtype=torch.bool)
                mask[1:13:2, 1 : 2 * nrhs + 1 : 2] = True
                guards = backing[~mask].clone()
                rhs = torch.ones(6, nrhs, dtype=dtype, device="cuda")
                saved_rhs = rhs.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    result = torch.triangular_solve(
                        rhs,
                        ga,
                        upper=False,
                        unitriangular=kind == "empty-unit",
                        out=(out, torch.empty(0, dtype=dtype, device="cuda")),
                    ).solution
                stream.synchronize()
                CHECK.assertTrue(result.data_ptr() == out.data_ptr())
                if kind.startswith("empty"):
                    CHECK.assertTrue(bool(result.isnan().all()))
                else:
                    pivot = 2 if kind == "missing-block" else 3
                    CHECK.assertTrue(bool(result[pivot].isnan().all()), result)
                torch.testing.assert_close(backing[~mask], guards, rtol=0, atol=0)
                torch.testing.assert_close(rhs, saved_rhs, rtol=0, atol=0)
                for original, saved in zip(
                    (ga.crow_indices(), ga.col_indices(), ga.values()), old_indices
                ):
                    torch.testing.assert_close(original, saved, rtol=0, atol=0)
                return dict(
                    nan=int(result.isnan().sum()),
                    inf=int(result.isinf().sum()),
                    numel=result.numel(),
                    behavior="nan-at-pivot-or-legacy-empty",
                    output_guards=True,
                    nondefault_stream=True,
                )

            check(f"{dtype}_{layout}_rhs{nrhs}_{kind}", edge)
    failed = [r["name"] for r in report["checks"] if r["status"] != "passed"]
    report.update(status="failed" if failed else "passed", failed=failed)
    save()
    print("N213_SUMMARY", len(report["checks"]), "failed", len(failed), flush=True)
    for row in report["checks"]:
        if row["status"] == "failed":
            print("FAIL", row["name"], row["error"][:450], flush=True)
    CHECK.assertTrue(not failed)


class TestBC250SparseTriangular(TestCase):
    def test_reference_groups(self):
        import torch

        if torch.version.hip is None or not torch.cuda.is_available():
            self.skipTest("Requires native ROCm GPU")
        for group in ("matrix", "edge", "extended", "unit", "trace"):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as folder:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--output",
                        str(Path(folder) / "result.json"),
                        "--group",
                        group,
                    ],
                    check=True,
                    timeout=120,
                )


if __name__ == "__main__":
    if "--output" in sys.argv:
        main()
    else:
        run_tests()
