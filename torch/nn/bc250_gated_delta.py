# SPDX-License-Identifier: MIT
"""Opt-in gfx1013 gated-delta preparation and eager chunked GPU attention.

Preparation uses scalar Triton forward/backward kernels. Triangular solves,
matmuls and the chunk scan use ATen. This is not the FLA package/backend.
Only floating eager first-order training is supported; chunk size is 64.
"""

import triton
import triton.language as tl

import torch
from torch.autograd.function import once_differentiable


@triton.jit
def _prepare(
    Q,
    K,
    V,
    BETA,
    OQ,
    OK,
    KB,
    VB,
    T: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    TP: tl.constexpr,
    NORM: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    t = row % TP
    h = (row // TP) % H
    b = row // (TP * H)
    token = (b * T + t) * H + h
    d = tl.arange(0, D)
    q = tl.load(Q + token * DK + d, (t < T) & (d < DK), 0).to(tl.float32)
    k = tl.load(K + token * DK + d, (t < T) & (d < DK), 0).to(tl.float32)
    v = tl.load(V + token * DV + d, (t < T) & (d < DV), 0).to(tl.float32)
    beta = tl.load(BETA + token, t < T, 0).to(tl.float32)
    if NORM:
        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6)
        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
    tl.store(OQ + row * DK + d, q * DK**-0.5, d < DK)
    tl.store(OK + row * DK + d, k, d < DK)
    tl.store(KB + row * DK + d, k * beta, d < DK)
    tl.store(VB + row * DV + d, v * beta, d < DV)


@triton.jit
def _prepare_backward(
    Q,
    K,
    V,
    BETA,
    GQ,
    GK,
    GKB,
    GVB,
    DQ,
    DKOUT,
    DVOUT,
    DB,
    T: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    TP: tl.constexpr,
    NORM: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    t = row % T
    h = (row // T) % H
    b = row // (T * H)
    token = (b * T + t) * H + h
    padded = (b * H + h) * TP + t
    d = tl.arange(0, D)
    q = tl.load(Q + token * DK + d, d < DK, 0).to(tl.float32)
    k = tl.load(K + token * DK + d, d < DK, 0).to(tl.float32)
    v = tl.load(V + token * DV + d, d < DV, 0).to(tl.float32)
    beta = tl.load(BETA + token).to(tl.float32)
    gq = tl.load(GQ + padded * DK + d, d < DK, 0) * DK**-0.5
    gkb = tl.load(GKB + padded * DK + d, d < DK, 0)
    gvb = tl.load(GVB + padded * DV + d, d < DV, 0)
    gk = tl.load(GK + padded * DK + d, d < DK, 0) + gkb * beta
    kn = k
    if NORM:
        rq = tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6)
        rk = tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
        qn = q * rq
        kn = k * rk
        gq = (gq - qn * tl.sum(gq * qn, 0)) * rq
        gk = (gk - kn * tl.sum(gk * kn, 0)) * rk
    tl.store(DQ + token * DK + d, gq, d < DK)
    tl.store(DKOUT + token * DK + d, gk, d < DK)
    tl.store(DVOUT + token * DV + d, gvb * beta, d < DV)
    tl.store(DB + token, tl.sum(gkb * kn, 0) + tl.sum(gvb * v, 0))


@triton.jit
def _decay(
    G,
    C,
    T: tl.constexpr,
    H: tl.constexpr,
    TP: tl.constexpr,
    BACKWARD: tl.constexpr,
    CHUNK: tl.constexpr,
):
    block = tl.program_id(0)
    chunks: tl.constexpr = TP // CHUNK
    chunk = block % chunks
    h = (block // chunks) % H
    b = block // (chunks * H)
    i = tl.arange(0, CHUNK)
    if BACKWARD:
        i = CHUNK - 1 - i
        x = tl.load(G + block * CHUNK + i)
        y = tl.cumsum(x, 0)
        t = chunk * CHUNK + i
        tl.store(C + (b * T + t) * H + h, y, t < T)
    else:
        t = chunk * CHUNK + i
        x = tl.load(G + (b * T + t) * H + h, t < T, 0).to(tl.float32)
        tl.store(C + block * CHUNK + i, tl.cumsum(x, 0))


class _Preparation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, normalize):
        b, t, h, dk = q.shape
        dv = v.shape[-1]
        tp = triton.cdiv(t, 64) * 64
        shape = (b, h, tp // 64, 64)
        oq = torch.empty((*shape, dk), device=q.device, dtype=torch.float32)
        ok, kb = torch.empty_like(oq), torch.empty_like(oq)
        vb = torch.empty((*shape, dv), device=q.device, dtype=torch.float32)
        decay = torch.empty(shape, device=q.device, dtype=torch.float32)
        with torch.cuda.device(q.device):
            _prepare[(b * h * tp,)](
                q,
                k,
                v,
                beta,
                oq,
                ok,
                kb,
                vb,
                t,
                h,
                dk,
                dv,
                tp,
                normalize,
                triton.next_power_of_2(max(dk, dv)),
                num_warps=4,
                enable_fp_fusion=False,
            )
            _decay[(b * h * (tp // 64),)](
                g,
                decay,
                t,
                h,
                tp,
                False,
                64,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(q, k, v, beta)
        ctx.normalize, ctx.tp, ctx.g_dtype = normalize, tp, g.dtype
        ctx.set_materialize_grads(False)
        return oq, ok, kb, vb, decay

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gk, gkb, gvb, gc):
        q, k, v, beta = ctx.saved_tensors
        b, t, h, dk = q.shape
        dv, tp = v.shape[-1], ctx.tp
        shape = (b, h, tp // 64, 64)
        grads = []
        for grad, width in ((gq, dk), (gk, dk), (gkb, dk), (gvb, dv)):
            grads.append(
                grad.contiguous()
                if grad is not None
                else torch.zeros((*shape, width), device=q.device, dtype=torch.float32)
            )
        dq, dkout, dvout, db = [torch.empty_like(x) for x in (q, k, v, beta)]
        dg = None
        with torch.cuda.device(q.device):
            _prepare_backward[(b * h * t,)](
                q,
                k,
                v,
                beta,
                *grads,
                dq,
                dkout,
                dvout,
                db,
                t,
                h,
                dk,
                dv,
                tp,
                ctx.normalize,
                triton.next_power_of_2(max(dk, dv)),
                num_warps=4,
                enable_fp_fusion=False,
            )
            if gc is not None:
                dg = torch.empty((b, t, h), device=q.device, dtype=ctx.g_dtype)
                _decay[(b * h * (tp // 64),)](
                    gc.contiguous(),
                    dg,
                    t,
                    h,
                    tp,
                    True,
                    64,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dq, dkout, dvout, dg, db, None


def prepare_gated_delta(query, key, value, g, beta, *, normalize=True):
    """Return FP32 padded/chunked (Q/sqrt(D), K, beta*K, beta*V, cumsum(g)).

    Inputs are BLHD Q/K/V and BLH log-decay/beta on actual gfx1013; Q/K have
    equal dimensions, all heads already expanded. Q/K normalization uses eps1e-6.
    Valid strided tensors are copied on GPU. Chunk width is 64, Dk/Dv <=128.
    Only eager first-order gradients are qualified, including both gates.
    """
    tensors = (query, key, value, g, beta)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("Expected Q/K/V/log-decay/beta tensors")
    if not all(x.is_cuda and x.device == query.device for x in tensors):
        raise ValueError("All tensors must share one gfx1013 device")
    if (
        torch.cuda.get_device_properties(query.device).gcnArchName.split(":")[0]
        != "gfx1013"
    ):
        raise ValueError("Native gfx1013 is required")
    types = (torch.float32, torch.float16, torch.bfloat16)
    if (
        query.dtype not in types
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise ValueError("Q/K/V must have the same FP32/FP16/BF16 dtype")
    if g.dtype not in types or beta.dtype not in types:
        raise ValueError("Log-decay and beta must be FP32/FP16/BF16")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("Expected equal BLHD Q/K and BLHV values")
    b, t, h, dk = query.shape
    if value.shape[:3] != (b, t, h) or g.shape != (b, t, h) or beta.shape != g.shape:
        raise ValueError("Batch, sequence and expanded heads must match")
    if (
        min(b, t, h, dk, value.shape[-1]) < 1
        or max(dk, value.shape[-1]) > 128
        or b * h * triton.cdiv(t, 64) * 64 > 1048576
    ):
        raise ValueError("Unsupported empty, head-width or padded token geometry")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be boolean")
    return _Preparation.apply(*(x.contiguous() for x in tensors), normalize)


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Native preparation plus ATen triangular solves and sequential chunk scan.

    Explicit API; does not install a Transformers hook or advertise FLA support.
    Initial state is FP32 [B,H,Dk,Dv] on the same device. Output has query dtype;
    returned final state is FP32. Log-decay is expected nonpositive, as in FLA.
    """
    if chunk_size != 64:
        raise ValueError("Only chunk_size=64 is supported")
    q, k, kb, vb, decay = prepare_gated_delta(
        query, key, value, g, beta, normalize=use_qk_l2norm_in_kernel
    )
    b, t, h, dk = query.shape
    dv = value.shape[-1]
    if initial_state is not None and (
        initial_state.shape != (b, h, dk, dv)
        or initial_state.device != query.device
        or initial_state.dtype != torch.float32
    ):
        raise ValueError("Initial state must be same-device FP32 [B,H,Dk,Dv]")
    mask = torch.ones((64, 64), device=query.device, dtype=torch.bool).triu(1)
    pair = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2))
        .masked_fill(mask, -float("inf"))
        .exp()
    )
    system = (kb @ k.transpose(-1, -2)) * pair
    attention = (q @ k.transpose(-1, -2)) * pair
    exp_decay = decay.exp().unsqueeze(-1)
    u = torch.linalg.solve_triangular(system, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(
        system, kb * exp_decay, upper=False, unitriangular=True
    )
    q = q * exp_decay
    k = k * (decay[..., -1:] - decay).exp().unsqueeze(-1)
    state = (
        torch.zeros((b, h, dk, dv), device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state
    )
    outputs = []
    for i in range(q.shape[2]):
        delta = u[:, :, i] - w[:, :, i] @ state
        outputs.append(q[:, :, i] @ state + attention[:, :, i] @ delta)
        state = (
            state * decay[:, :, i, -1].exp()[..., None, None]
            + k[:, :, i].transpose(-1, -2) @ delta
        )
    out = torch.stack(outputs, dim=2).flatten(2, 3)[:, :, :t].transpose(1, 2)
    return out.to(query.dtype).contiguous(), state if output_final_state else None
