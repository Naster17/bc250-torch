# SPDX-License-Identifier: MIT
"""Opt-in gfx1013 gated-delta preparation and eager chunked GPU attention.

Standalone preparation uses scalar Triton kernels. The full operation uses
reference-ordered ATen preparation for FP16/BF16 and native preparation for FP32.
Triangular solves, matmuls and the chunk scan use ATen by default; the explicit
``use_fused_solve_scan=True`` option replaces the two batched triangular solves
with one fused in-register unit-triangular inverse and the sequential Python
chunk scan with a fused linear-recurrence scan (both with analytic backward).
This is not the FLA package/backend.
Only floating eager first-order training is supported; chunk size is 64.
"""

import triton
import triton.language as tl

import torch
import torch.nn.functional as F
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


def _validate_inputs(query, key, value, g, beta, normalize):
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


def prepare_gated_delta(query, key, value, g, beta, *, normalize=True):
    """Return FP32 padded/chunked (Q/sqrt(D), K, beta*K, beta*V, cumsum(g)).

    Inputs are BLHD Q/K/V and BLH log-decay/beta on actual gfx1013; Q/K have
    equal dimensions, all heads already expanded. Q/K normalization uses eps1e-6.
    Valid strided tensors are copied on GPU. Chunk width is 64, Dk/Dv <=128.
    Only eager first-order gradients are qualified, including both gates.
    """
    _validate_inputs(query, key, value, g, beta, normalize)
    return _Preparation.apply(
        *(x.contiguous() for x in (query, key, value, g, beta)), normalize
    )


def _reference_ordered_prepare(query, key, value, g, beta, normalize):
    _validate_inputs(query, key, value, g, beta, normalize)
    b, t, h, dk = query.shape
    query, key, value, beta, g = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if normalize:
        # Preserve reference operation order: small rounding changes can affect
        # gradients after multiple BF16 layers. square() has a different VJP.
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query * dk**-0.5
    pad = (-t) % 64
    query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, pad)) for x in (beta, g)]
    vb, kb = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, kb, vb = [
        x.reshape(b, h, -1, 64, x.shape[-1]) for x in (query, key, kb, vb)
    ]
    return query, key, kb, vb, g.reshape(b, h, -1, 64).cumsum(-1)


# Saved-state buffer cap for the fused scan (forward saves every chunk state;
# backward adds transient buffers of similar size). Explicit refusal, not a
# silent fallback, so callers can choose the default ATen path instead.
FUSED_SCAN_STATE_BYTES_LIMIT = 256 * 1024 * 1024
# Measured crossover (D68 decomposition, gfx1013): at one chunk every fusion
# variant regresses (0.79-0.94x) because the reformulation adds more launches
# than the single loop iteration costs; from two chunks up the fused path wins
# at every measured geometry (1.09-1.81x). Single-chunk calls therefore keep
# the default ATen path even when the fused option is requested.
FUSED_SCAN_MIN_CHUNKS = 2


@triton.jit
def _tril_inverse(A, M, BLOCK: tl.constexpr):
    """In-register forward substitution: M = (I + tril(A, -1))^-1.

    One program per matrix; the diagonal is treated as 1 and the upper triangle
    is ignored, exactly like solve_triangular(unitriangular=True). Row i of the
    inverse is e_i - sum_j A[i, j] M[j, :] with earlier rows already final.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    base = pid * BLOCK * BLOCK
    a = tl.load(
        A + base + rows[:, None] * BLOCK + cols[None, :],
        rows[:, None] > cols[None, :],
        0.0,
    )
    m = (rows[:, None] == cols[None, :]).to(tl.float32)
    for i in range(1, BLOCK):
        a_row = tl.sum(tl.where(rows[:, None] == i, a, 0.0), 0)
        new_row = -tl.sum(a_row[:, None] * m, 0)
        new_row = tl.where(cols == i, 1.0, new_row)
        m = tl.where(rows[:, None] == i, new_row[None, :], m)
    tl.store(M + base + rows[:, None] * BLOCK + cols[None, :], m)


class _FusedTrilInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, system):
        flat = system.reshape(-1, 64, 64).contiguous()
        m = torch.empty_like(flat)
        with torch.cuda.device(system.device):
            _tril_inverse[(flat.shape[0],)](
                flat, m, 64, num_warps=4, enable_fp_fusion=False
            )
        ctx.save_for_backward(m)
        return m.reshape(system.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dm):
        (m,) = ctx.saved_tensors
        mt = m.transpose(-1, -2)
        # d(A^-1) = -A^-1 dA A^-1; only the strict lower triangle of the input
        # participates (unit diagonal, masked upper), so mask the gradient.
        grad = -(mt @ dm.reshape(m.shape) @ mt).tril(-1)
        return grad.reshape(dm.shape)


@triton.jit
def _chunk_scan_fwd(
    M,
    N,
    P,
    R,
    S0,
    OUT,
    SALL,
    SFINAL,
    C,
    DK,
    DV,
    DKP: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
):
    """Fused linear-recurrence scan, one program per (batch*head, value block).

    S_{i+1} = M_i S_i + N_i and out_i = P_i S_i + R_i are column-separable, so
    each program owns a [DK, BV] state block across all chunks. Every entering
    state is saved for the analytic backward.
    """
    bh = tl.program_id(0)
    vv = tl.program_id(1) * BV + tl.arange(0, BV)
    dk = tl.arange(0, DKP)
    dkm = dk < DK
    vm = vv < DV
    if HAS_INITIAL:
        state = tl.load(
            S0 + bh * DK * DV + dk[:, None] * DV + vv[None, :],
            dkm[:, None] & vm[None, :],
            0.0,
        )
    else:
        state = tl.zeros((DKP, BV), tl.float32)
    for i in range(C):
        soff = (bh * C + i) * DK * DV
        tl.store(
            SALL + soff + dk[:, None] * DV + vv[None, :],
            state,
            dkm[:, None] & vm[None, :],
        )
        for r in range(64):
            p = tl.load(P + ((bh * C + i) * 64 + r) * DK + dk, dkm, 0.0)
            acc = tl.sum(p[:, None] * state, 0)
            rr = tl.load(R + ((bh * C + i) * 64 + r) * DV + vv, vm, 0.0)
            tl.store(OUT + ((bh * C + i) * 64 + r) * DV + vv, acc + rr, vm)
        new = tl.zeros((DKP, BV), tl.float32)
        for r in range(DK):
            m = tl.load(M + (bh * C + i) * DK * DK + r * DK + dk, dkm, 0.0)
            row = tl.sum(m[:, None] * state, 0)
            n = tl.load(N + soff + r * DV + vv, vm, 0.0)
            new = tl.where(dk[:, None] == r, (row + n)[None, :], new)
        state = new
    tl.store(
        SFINAL + bh * DK * DV + dk[:, None] * DV + vv[None, :],
        state,
        dkm[:, None] & vm[None, :],
    )


@triton.jit
def _chunk_scan_bwd(
    M,
    P,
    DOUT,
    GFINAL,
    GALL,
    C,
    DK,
    DV,
    DKP: tl.constexpr,
    BV: tl.constexpr,
    HAS_FINAL: tl.constexpr,
):
    """Adjoint scan: G_i = M_i^T G_{i+1} + P_i^T dout_i, reverse chunk order.

    Emits every G_i; the remaining input gradients (dM, dN, dP, dR) are batched
    matmuls against the saved entering states in the Function backward.
    """
    bh = tl.program_id(0)
    vv = tl.program_id(1) * BV + tl.arange(0, BV)
    dk = tl.arange(0, DKP)
    dkm = dk < DK
    vm = vv < DV
    if HAS_FINAL:
        g = tl.load(
            GFINAL + bh * DK * DV + dk[:, None] * DV + vv[None, :],
            dkm[:, None] & vm[None, :],
            0.0,
        )
    else:
        g = tl.zeros((DKP, BV), tl.float32)
    t = tl.arange(0, 64)
    for i in range(C - 1, -1, -1):
        dout = tl.load(
            DOUT + ((bh * C + i) * 64 + t[:, None]) * DV + vv[None, :],
            vm[None, :],
            0.0,
        )
        new = tl.zeros((DKP, BV), tl.float32)
        for r in range(DK):
            p = tl.load(P + ((bh * C + i) * 64 + t) * DK + r)
            m = tl.load(M + (bh * C + i) * DK * DK + dk * DK + r, dkm, 0.0)
            row = tl.sum(p[:, None] * dout, 0) + tl.sum(m[:, None] * g, 0)
            new = tl.where(dk[:, None] == r, row[None, :], new)
        tl.store(
            GALL + (bh * C + i) * DK * DV + dk[:, None] * DV + vv[None, :],
            new,
            dkm[:, None] & vm[None, :],
        )
        g = new


class _FusedChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mm, nn, pp, rr, initial_state):
        b, h, c, dk, _ = mm.shape
        dv = nn.shape[-1]
        out = torch.empty((b, h, c, 64, dv), device=mm.device, dtype=torch.float32)
        s_all = torch.empty((b, h, c, dk, dv), device=mm.device, dtype=torch.float32)
        s_final = torch.empty((b, h, dk, dv), device=mm.device, dtype=torch.float32)
        bv = min(64, triton.next_power_of_2(dv))
        dkp = triton.next_power_of_2(dk)
        grid = (b * h, triton.cdiv(dv, bv))
        with torch.cuda.device(mm.device):
            _chunk_scan_fwd[grid](
                mm,
                nn,
                pp,
                rr,
                initial_state,
                out,
                s_all,
                s_final,
                c,
                dk,
                dv,
                dkp,
                bv,
                initial_state is not None,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(mm, pp, s_all)
        ctx.has_initial = initial_state is not None
        ctx.set_materialize_grads(False)
        return out, s_final

    @staticmethod
    @once_differentiable
    def backward(ctx, dout, ds_final):
        mm, pp, s_all = ctx.saved_tensors
        b, h, c, dk, _ = mm.shape
        dv = s_all.shape[-1]
        if dout is None:
            dout = torch.zeros(
                (b, h, c, 64, dv), device=mm.device, dtype=torch.float32
            )
        dout = dout.contiguous()
        g_all = torch.empty_like(s_all)
        has_final = ds_final is not None
        bv = min(64, triton.next_power_of_2(dv))
        dkp = triton.next_power_of_2(dk)
        grid = (b * h, triton.cdiv(dv, bv))
        with torch.cuda.device(mm.device):
            _chunk_scan_bwd[grid](
                mm,
                pp,
                dout,
                ds_final.contiguous() if has_final else dout,
                g_all,
                c,
                dk,
                dv,
                dkp,
                bv,
                has_final,
                num_warps=4,
                enable_fp_fusion=False,
            )
        if has_final:
            g_next = torch.cat(
                [g_all[:, :, 1:], ds_final.contiguous().unsqueeze(2)], 2
            )
        else:
            g_next = torch.cat(
                [
                    g_all[:, :, 1:],
                    torch.zeros(
                        (b, h, 1, dk, dv), device=mm.device, dtype=torch.float32
                    ),
                ],
                2,
            )
        s_t = s_all.transpose(-1, -2)
        dmm = g_next @ s_t
        dpp = dout @ s_t
        dinitial = g_all[:, :, 0] if ctx.has_initial else None
        return dmm, g_next, dpp, dout, dinitial


def _fused_solve_scan(q, k, kb, vb, decay, system, attention, initial_state, dk, dv):
    """Fused inverse + linear-recurrence scan on the prepared FP32 tensors.

    Mathematically identical to the default solve+loop path: with
    delta_i = u_i - w_i S_i, the recurrence becomes
    S_{i+1} = (d_i I - k_i^T w_i) S_i + k_i^T u_i and
    out_i = (q_i - attn_i w_i) S_i + attn_i u_i. The precompute stays in
    autograd-visible ATen; only the inverse and the scan are fused.
    """
    inv = _FusedTrilInverse.apply(system)
    u = inv @ vb
    w = inv @ (kb * decay.exp().unsqueeze(-1))
    q = q * decay.exp().unsqueeze(-1)
    k = k * (decay[..., -1:] - decay).exp().unsqueeze(-1)
    chunk_decay = decay[..., -1].exp()[..., None, None]
    eye = torch.eye(dk, device=q.device, dtype=torch.float32)
    mm = chunk_decay * eye - k.transpose(-1, -2) @ w
    nn = k.transpose(-1, -2) @ u
    pp = q - attention @ w
    rr = attention @ u
    return _FusedChunkScan.apply(mm, nn, pp, rr, initial_state)


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
    use_fused_solve_scan=False,
):
    """Precision-aware GPU preparation, ATen triangular solves and chunk scan.

    Explicit API; does not install a Transformers hook or advertise FLA support.
    FP16/BF16 use reference-ordered ATen preparation for model-gradient parity;
    FP32 outside autocast uses native preparation. The standalone native API remains
    explicit and experimental for low-precision model integration.
    Initial state is FP32 [B,H,Dk,Dv] on the same device. Output has query dtype;
    returned final state is FP32. Log-decay is expected nonpositive, as in FLA.
    ``use_fused_solve_scan=True`` opts into the fused Triton unit-triangular
    inverse and linear-recurrence chunk scan (analytic backward, same math,
    different FP32 rounding) when the sequence spans at least
    FUSED_SCAN_MIN_CHUNKS chunks; single-chunk calls keep the default path
    (measured D68 crossover). It refuses when the saved-state buffer would
    exceed FUSED_SCAN_STATE_BYTES_LIMIT. The default path is unchanged.
    """
    if chunk_size != 64:
        raise ValueError("Only chunk_size=64 is supported")
    if not isinstance(use_fused_solve_scan, bool):
        raise TypeError("use_fused_solve_scan must be boolean")
    prepare = (
        prepare_gated_delta
        if isinstance(query, torch.Tensor)
        and query.dtype == torch.float32
        and not torch.is_autocast_enabled("cuda")
        else _reference_ordered_prepare
    )
    q, k, kb, vb, decay = prepare(
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
    if use_fused_solve_scan and q.shape[2] >= FUSED_SCAN_MIN_CHUNKS:
        if b * h * q.shape[2] * dk * dv * 4 > FUSED_SCAN_STATE_BYTES_LIMIT:
            raise ValueError(
                "Fused scan saved-state buffer exceeds "
                f"{FUSED_SCAN_STATE_BYTES_LIMIT} bytes; use the default path"
            )
        out, state = _fused_solve_scan(
            q, k, kb, vb, decay, system, attention, initial_state, dk, dv
        )
    else:
        decayed_k_beta = kb * decay.exp().unsqueeze(-1)
        u = torch.linalg.solve_triangular(
            system, vb, upper=False, unitriangular=True
        )
        w = torch.linalg.solve_triangular(
            system, decayed_k_beta, upper=False, unitriangular=True
        )
        q = q * decay.exp().unsqueeze(-1)
        k = k * (decay[..., -1:] - decay).exp().unsqueeze(-1)
        chunk_decay = decay[..., -1].exp()[..., None, None]
        state = (
            torch.zeros((b, h, dk, dv), device=query.device, dtype=torch.float32)
            if initial_state is None
            else initial_state
        )
        out = torch.zeros_like(u)
        for i in range(q.shape[2]):
            delta = u[:, :, i] - w[:, :, i] @ state
            inter_chunk = q[:, :, i] @ state
            out[:, :, i] = inter_chunk + attention[:, :, i] @ delta
            state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ delta
    out = out.flatten(2, 3)[:, :, :t].transpose(1, 2)
    return out.to(query.dtype).contiguous(), state if output_final_state else None
