# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in scalar, tiled SDPA for BC-250. No global dispatcher registration.

Eager first-order autograd only. Finite Q/K/V, no dropout, head dimension <=256,
sequence lengths <=4096. Boolean and finite/-inf additive masks may broadcast to
[batch, query_heads, queries, keys]. Mask gradients are unsupported. This is a
memory-saving prototype, not a performance-qualified FlashAttention backend.
"""

import math

import triton
import triton.language as tl

import torch
from torch.autograd.function import once_differentiable


@triton.jit
def _mask_scores(
    score,
    Mask,
    b,
    h,
    rows,
    cols,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BOOL_MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
    valid,
):
    if HAS_MASK:
        bias = tl.load(
            Mask + b * S0 + h * S1 + rows * S2 + cols * S3, mask=valid, other=0
        )
        if BOOL_MASK:
            score = tl.where(bias, score, -float("inf"))
        else:
            score += bias.to(tl.float32)
    if CAUSAL:
        valid = valid & (rows >= cols)
    return tl.where(valid, score, -float("inf"))


@triton.jit
def _forward(
    Q,
    K,
    V,
    Mask,
    O,
    LSE,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BOOL_MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BD: tl.constexpr,
    T: tl.constexpr,
    LSE_ONLY: tl.constexpr,
):
    row = tl.program_id(0)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    kh = h // (H // HK)
    d = tl.arange(0, BD)
    q = tl.load(Q + (bh * M + row) * D + d, d < D, 0).to(tl.float32)
    maximum = -float("inf")
    normalizer = 0.0
    acc = tl.full((BD,), 0, tl.float32)
    # Upper-left causal attention cannot read keys beyond this query row.
    end = tl.minimum(N, row + 1) if CAUSAL else N
    for start in range(tl.cdiv(end, T)):
        cols = start * T + tl.arange(0, T)
        k = tl.load(
            K + ((b * HK + kh) * N + cols[:, None]) * D + d[None, :],
            (cols[:, None] < N) & (d[None, :] < D),
            0,
        ).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * SCALE
        score = _mask_scores(
            score,
            Mask,
            b,
            h,
            row,
            cols,
            S0,
            S1,
            S2,
            S3,
            HAS_MASK,
            BOOL_MASK,
            CAUSAL,
            cols < N,
        )
        next_max = tl.maximum(maximum, tl.max(score, 0))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.exp(score - safe_max)
        if not LSE_ONLY:
            v = tl.load(
                V + ((b * HK + kh) * N + cols[:, None]) * D + d[None, :],
                (cols[:, None] < N) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        normalizer = normalizer * alpha + tl.sum(p, 0)
        maximum = next_max
    denominator = tl.where(normalizer > 0, normalizer, 1.0)
    if not LSE_ONLY:
        tl.store(O + (bh * M + row) * D + d, acc / denominator, d < D)
    tl.store(
        LSE + bh * M + row,
        tl.where(normalizer > 0, maximum + tl.log(denominator), -float("inf")),
    )


@triton.jit
def _backward_q(
    Q,
    K,
    V,
    Mask,
    O,
    LSE,
    DO,
    DQ,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BOOL_MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BD: tl.constexpr,
    T: tl.constexpr,
):
    row, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    kh = h // (H // HK)
    d = tl.arange(0, BD)
    offset = (bh * M + row) * D + d
    q = tl.load(Q + offset, d < D, 0).to(tl.float32)
    go = tl.load(DO + offset, d < D, 0).to(tl.float32)
    o = tl.load(O + offset, d < D, 0)
    delta = tl.sum(go * o, 0)
    lse = tl.load(LSE + bh * M + row)
    acc = tl.full((BD,), 0, tl.float32)
    end = tl.minimum(N, row + 1) if CAUSAL else N
    for start in range(tl.cdiv(end, T)):
        cols = start * T + tl.arange(0, T)
        offsets = ((b * HK + kh) * N + cols[:, None]) * D + d[None, :]
        valid = (cols[:, None] < N) & (d[None, :] < D)
        k = tl.load(K + offsets, valid, 0).to(tl.float32)
        v = tl.load(V + offsets, valid, 0).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * SCALE
        score = _mask_scores(
            score,
            Mask,
            b,
            h,
            row,
            cols,
            S0,
            S1,
            S2,
            S3,
            HAS_MASK,
            BOOL_MASK,
            CAUSAL,
            cols < N,
        )
        p = tl.exp(score - tl.where(lse == -float("inf"), 0.0, lse))
        ds = p * (tl.sum(v * go[None, :], 1) - delta)
        acc += tl.sum(ds[:, None] * k, 0) * SCALE
    tl.store(DQ + offset, acc, d < D)


@triton.jit
def _backward_kv(
    Q,
    K,
    V,
    Mask,
    O,
    LSE,
    DO,
    DK,
    DV,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BOOL_MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BD: tl.constexpr,
    T: tl.constexpr,
):
    col, bkh = tl.program_id(0), tl.program_id(1)
    b, kh = bkh // HK, bkh % HK
    d = tl.arange(0, BD)
    offset = (bkh * N + col) * D + d
    k = tl.load(K + offset, d < D, 0).to(tl.float32)
    v = tl.load(V + offset, d < D, 0).to(tl.float32)
    dk = tl.full((BD,), 0, tl.float32)
    dv = tl.full((BD,), 0, tl.float32)
    # Each KV row owns its gradients, including all grouped query heads. No atomics.
    for group in range(H // HK):
        h = kh * (H // HK) + group
        bh = b * H + h
        # This key contributes only to query rows >=col. Keep the boundary
        # tile masked; keys beyond all query rows naturally produce zero.
        first = col // T if CAUSAL else 0
        for start in range(first, tl.cdiv(M, T)):
            rows = start * T + tl.arange(0, T)
            offsets = (bh * M + rows[:, None]) * D + d[None, :]
            valid = (rows[:, None] < M) & (d[None, :] < D)
            q = tl.load(Q + offsets, valid, 0).to(tl.float32)
            go = tl.load(DO + offsets, valid, 0).to(tl.float32)
            o = tl.load(O + offsets, valid, 0)
            delta = tl.sum(go * o, 1)
            lse = tl.load(LSE + bh * M + rows, rows < M, -float("inf"))
            score = tl.sum(q * k[None, :], 1) * SCALE
            score = _mask_scores(
                score,
                Mask,
                b,
                h,
                rows,
                col,
                S0,
                S1,
                S2,
                S3,
                HAS_MASK,
                BOOL_MASK,
                CAUSAL,
                rows < M,
            )
            p = tl.exp(score - tl.where(lse == -float("inf"), 0.0, lse))
            ds = p * (tl.sum(go * v[None, :], 1) - delta)
            dk += tl.sum(ds[:, None] * q, 0) * SCALE
            dv += tl.sum(p[:, None] * go, 0)
    tl.store(DK + offset, dk, d < D)
    tl.store(DV + offset, dv, d < D)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, causal, scale, reference_forward):
        b, h, m, d = q.shape
        hk, n = k.shape[1:3]
        # FP32 output is retained for the softmax gradient's row reduction;
        # storing only rounded BF16 output would degrade that reduction.
        if reference_forward:
            from torch.nn.attention import sdpa_kernel, SDPBackend

            reference_mask = mask if mask.numel() else None
            if reference_mask is not None and reference_mask.dtype != torch.bool:
                reference_mask = reference_mask.float()
            with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                o = torch.nn.functional.scaled_dot_product_attention(
                    q.float(),
                    k.float(),
                    v.float(),
                    attn_mask=reference_mask,
                    is_causal=causal,
                    scale=scale,
                    enable_gqa=h != hk,
                )
        else:
            o = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        lse = torch.empty((b, h, m), device=q.device, dtype=torch.float32)
        strides = mask.stride() if mask.numel() else (0, 0, 0, 0)
        params = (
            m,
            n,
            h,
            hk,
            d,
            *strides,
            scale,
            mask.numel() > 0,
            mask.dtype == torch.bool,
            causal,
            triton.next_power_of_2(d),
            32 if causal and 64 <= d <= 128 else 16,
        )
        with torch.cuda.device(q.device):
            if b * h * m:
                _forward[(m, b * h)](
                    q,
                    k,
                    v,
                    mask,
                    o,
                    lse,
                    *params,
                    LSE_ONLY=reference_forward,
                    num_warps=4,
                )
        ctx.save_for_backward(q, k, v, mask, o, lse)
        ctx.params = params
        return o.to(q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, mask, o, lse = ctx.saved_tensors
        b, h, m, d = q.shape
        hk, n = k.shape[1:3]
        grad = grad.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        with torch.cuda.device(q.device):
            if b * h * m:
                _backward_q[(m, b * h)](
                    q, k, v, mask, o, lse, grad, dq, *ctx.params, num_warps=4
                )
            if b * hk * n:
                _backward_kv[(n, b * hk)](
                    q, k, v, mask, o, lse, grad, dk, dv, *ctx.params, num_warps=4
                )
        return dq, dk, dv, None, None, None, None


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    """Explicit BC-250 tiled attention; same leading layout as functional SDPA.

    Unsupported options raise; this function never silently falls back. Importing
    it does not change functional SDPA, build capabilities, or backend flags.
    No higher-order gradients, torch.compile, CUDA graphs, or dropout support is
    claimed. BF16 head widths >128 in training use a reference math forward
    result plus native tiled backward to limit model-level rounding drift.
    That forward has quadratic temporary storage; saved state stays linear.
    CUDA autocast casts eligible floating inputs to its target dtype,
    as functional SDPA does, while preserving gradients through those casts.
    """
    if torch.compiler.is_compiling():
        raise NotImplementedError("gfx1013 tiled SDPA currently supports eager only")
    tensors = (query, key, value)
    if any(t.layout != torch.strided or t.ndim != 4 for t in tensors):
        raise ValueError("Expected strided [B,H,L,D] Q/K/V")
    if any(t.device != query.device for t in tensors):
        raise ValueError("Q/K/V must share device")
    if torch.is_autocast_enabled("cuda") and all(
        t.dtype in (torch.float32, torch.float16, torch.bfloat16) for t in tensors
    ):
        target = torch.get_autocast_dtype("cuda")
        tensors = tuple(t.to(target) for t in tensors)
        query, key, value = tensors
        if attn_mask is not None and attn_mask.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            attn_mask = attn_mask.to(target)
    if any(t.dtype != query.dtype for t in tensors):
        raise ValueError("Q/K/V must share dtype outside autocast")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Only FP32/FP16/BF16 Q/K/V are supported")
    if query.device.type != "cuda" or not torch.version.hip:
        raise ValueError("Requires native gfx1013 ROCm")
    if (
        torch.cuda.get_device_properties(query.device).gcnArchName.split(":")[0]
        != "gfx1013"
    ):
        raise ValueError("This prototype is restricted to gfx1013")
    b, h, m, d = query.shape
    bk, hk, n, dk = key.shape
    if (
        bk != b
        or value.shape != key.shape
        or dk != d
        or not 1 <= d <= 256
        or not h
        or not hk
        or h % hk
        or (h != hk and not enable_gqa)
    ):
        raise ValueError("Unsupported batch/head/dimension combination")
    if max(m, n) > 4096 or any(t.numel() >= 2**31 for t in tensors):
        raise ValueError("Prototype bound: sequence <=4096 and tensor elements <2**31")
    if dropout_p != 0:
        raise NotImplementedError("Dropout is not implemented")
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    if attn_mask is not None:
        if is_causal:
            raise ValueError("Explicit mask and is_causal cannot be combined")
        if (
            attn_mask.device != query.device
            or attn_mask.layout != torch.strided
            or attn_mask.dtype not in (torch.bool, torch.float32, query.dtype)
        ):
            raise ValueError(
                "Mask must be boolean or compatible floating point on the same device"
            )
        if attn_mask.requires_grad:
            raise NotImplementedError("Mask gradients are not implemented")
        attn_mask = attn_mask.expand(b, h, m, n)
    else:
        attn_mask = query.new_empty((0,))
    with torch.cuda.device(query.device):
        return _Attention.apply(
            *(t.contiguous() for t in tensors),
            attn_mask,
            bool(is_causal),
            scale,
            d > 128
            and query.dtype == torch.bfloat16
            and torch.is_grad_enabled()
            and any(t.requires_grad for t in tensors),
        )


def register_transformers_attention():
    """Explicitly register ``bc250_tiled`` with the installed Transformers package.

    Select with ``attn_implementation="bc250_tiled"`` after calling this helper.
    Requires zero attention dropout; output attention weights are unavailable.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        is_causal=None,
        **kwargs,
    ):
        # Reject optional semantics instead of quietly producing ordinary SDPA.
        for name in ("softcap", "sliding_window", "head_mask"):
            if kwargs.get(name) is not None:
                raise NotImplementedError(f"bc250_tiled does not support {name}")
        causal = (
            (
                getattr(module, "is_causal", True)
                and query.shape[-2] > 1
                and attention_mask is None
            )
            if is_causal is None
            else is_causal
        )
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
            dropout,
            causal,
            scale=scaling,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return output.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register("bc250_tiled", attention)
    # Custom attention names also need a mask builder; without this registration
    # Transformers may omit padding/packed-position masks for the unknown name.
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "bc250_tiled", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    )
