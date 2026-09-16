# BC-250 gfx1013 — PyTorch branch

> **REQUIRED: the r1 kernel + native prefixes.** This branch builds, but
> the build only succeeds against the native gfx1013 ROCm stack and only
> runs on [`Naster17/bc250-linux`](https://github.com/Naster17/bc250-linux)
> branch `bc250-r1` (one-shot boot). You need:
>
> - Native prefixes: rocBLAS, hipBLAS (**solver rebuild**, see below),
>   hipblas-common, hipblaslt (headers+lib), roctracer, rocFFT/hipFFT,
>   rocRAND/hiprand, rocSPARSE/hipSPARSE, rocSOLVER/hipSOLVER,
>   rocPRIM/hipCUB/rocThrust headers, MIOpen — all from
>   [`Naster17/bc250-rocm`](https://github.com/Naster17/bc250-rocm) branch
>   `bc250-gfx1013` (see its README for build order).
> - Full recipe with exact mounts/env: [`Naster17/bc250-stack`](https://github.com/Naster17/bc250-stack).

## What this branch is

Fork of `pytorch/pytorch` at tag `v2.9.1`, plus one BC-250 commit:

- `96a79fe` — **ROCm gfx1013 prefix-aware build fixes** (7 files, +55/−3).
  The default build assumes distro ROCm layout; ours lives in isolated
  native prefixes. Everything else is upstream `v2.9.1`.

## The commit, hunk by hunk

1. `tools/setup_helpers/cmake.py` — forward `Python_*` hints
   (`Python_INCLUDE_DIR/LIBRARY`, `Python_NumPy_INCLUDE_DIR(S)`,
   `Python_ROOT_DIR`, `Python_FIND_VIRTUALENV/STRATEGY`) from the
   environment into the CMake configure. Without this, `setup.py` drops
   them and `find_package(Python COMPONENTS Development.Module NumPy)`
   fails against the `/opt/pydev` overlay.
2. `cmake/public/LoadHIP.cmake` — roctx/roctracer prefix fallback: search
   `CMAKE_PREFIX_PATH` for `include/roctracer/roctx.h`, append it to
   `ROCM_INCLUDE_DIRS`, use `lib/libroctx64.so`. Stock `find_library`
   only looks in `${ROCM_PATH}/lib`, where our layout has no roctracer.
3. `cmake/Dependencies.cmake` (math libs) — tolerate headers+lib-only
   hipblaslt: upstream `v2.9.1` unconditionally links `roc::hipblaslt`;
   drop it when the package is absent.
4. `caffe2/CMakeLists.txt` — add the roctracer prefix include to
   `torch_hip` (fixes `roctracer/roctx.h` not found in profiler/nvtx
   sources).
5. `torch/CMakeLists.txt` + `torch/csrc/dynamo/cpython_includes.h` —
   roctracer include for `torch_python`, and compile dynamo C sources
   (`cpython_defs.c`, `eval_frame.c`) as **C** with `-std=gnu11`
   (amdclang++ chokes on `-std=c++17` for C, and C++ compilation breaks
   on `Py_BUILD_CORE` internals like `_PyThreadState_GET`).
6. `functorch/CMakeLists.txt` — same LANGUAGE C treatment for
   `dim_opcode.c`.
7. `cmake/Dependencies.cmake` (ABI) — host `CMAKE_CXX_FLAGS +=
   -fclang-abi-compat=17` to match HIP host mangling (both compilers are
   amdclang++ here; without it `libtorch_hip.so` misses Half symbols).

## Build (differs from upstream)

Upstream: `python setup.py bdist_wheel`. For gfx1013, GPU-free +
network-free container, all deps as read-only prefix mounts:

```sh
# On the board, with all native prefixes + python overlay staged:
doas ./build-torch-gfx1013.sh   # full script in bc250-stack/pytorch/
```

Key env (see script for the full list):

```sh
USE_ROCM=1 USE_CUDA=0 PYTORCH_ROCM_ARCH=gfx1013
USE_DISTRIBUTED=0 USE_NCCL=0 USE_RCCL=0 USE_GLOO=0 USE_MKLDNN=0
USE_FBGEMM=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0
USE_ROCM_CK_GEMM=0 USE_ROCM_CK_SDPA=0 BLAS=Eigen
CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
Python_EXECUTABLE=/venv/bin/python Python_INCLUDE_DIR=/opt/pydev/include/python3.12
Python_LIBRARY=/opt/pydev/lib/x86_64-linux-gnu/libpython3.12.a
MAX_JOBS=16   # ~2h full build
```

Python dev overlay must contain headers + `libpython3.12.a` **and** the
multiarch `pyconfig.h` at both `include/x86_64-linux-gnu/...` and
`include/python3.12/x86_64-linux-gnu/...`, plus `CFLAGS/CPPFLAGS` with
the pydev + numpy include dirs (stub compile). Eigen 3.4.0 is copied
into `third_party/eigen`; other submodules are pre-staged.

Wheel packaging (same mounts + `USE_SYSTEM_LIBS=1`, nccl off):

```sh
doas ./build-wheel-gfx1013.sh
# -> torch-2.9.1a0+gitunknown-cp312-cp312-linux_x86_64.whl (~133 MB)
```

## Run (differs from upstream)

```py
import torch
torch.cuda.is_available()   # True
torch.cuda.device_count()   # 1
torch.cuda.get_arch_list()  # ['gfx1013']
```

Runtime needs (all mandatory, baked into the r1.2 image via
`ld.so.conf.d/bc250-r12.conf` + `ld.so.preload`, or set manually):

- r1 kernel one-shot boot, `HSA_ENABLE_SDMA=1`, `OMP_NUM_THREADS=4`.
- `LD_PRELOAD=<omp>/libomp.so` (LLVM libomp from the build-tools image).
- `LD_LIBRARY_PATH` with syslibs (openblas/gfortran) + every native prefix.
- `numpy<2` in the venv (built against 1.x API).
- No distributed/NCCL, flash-attn, or CK paths — single-GPU train/infer
  only. Standard SDPA falls back to the math backend (fine ≤2k context;
  use llama.cpp for long context — it has the RDNA1 FA fix).

Validated: 64x64 GEMM OK; 50-step MLP train bit-identical to CPU
(`1.029231 / 1.004733 / 0.985372`). Launcher: `run-torch.sh` in
`bc250-stack/images/` (also in the profile repo as
`userspace/run-torch.sh`).

## Upstream sync

`main` tracks `pytorch/pytorch`. To rebase `bc250-gfx1013`:

```sh
git fetch origin
git checkout bc250-gfx1013
git rebase origin/v2.9.1   # or newer tag; re-check the hipblaslt hunk,
                           # upstream refactors this area often
# rebuild wheel + rerun GEMM/train gates before pushing
```
