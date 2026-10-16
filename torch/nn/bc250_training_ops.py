# SPDX-License-Identifier: MIT
"""Opt-in BC-250 eager first-order causal convolution and chunked causal loss.

Native scalar Triton convolution avoids MIOpen search for this bounded operation.
Chunked loss recomputes FP32 probabilities in backward; no full FP32 logits copy.
No fused gated-delta/FLA implementation, graph or higher-order gradient contract.
"""

import functools

import triton
import triton.language as tl

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


@functools.lru_cache(None)
def _native(device: torch.device) -> bool:
    return (
        getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":")[
            0
        ]
        == "gfx1013"
    )


@triton.jit
def _conv(
    X,
    W,
    B,
    Y,
    Z,
    L: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    XB: tl.constexpr,
    XC: tl.constexpr,
    XT: tl.constexpr,
    WC: tl.constexpr,
    WK: tl.constexpr,
    BS: tl.constexpr,
    K: tl.constexpr,
    BIAS: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = i % L
    c = (i // L) % C
    b = i // (L * C)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(K):
        source = t + k - (K - 1)
        x = tl.load(
            X + b * XB + c * XC + source * XT, (i < N) & (source >= 0), other=0
        ).to(tl.float32)
        w = tl.load(W + c * WC + k * WK, i < N, other=0).to(tl.float32)
        acc = acc + x * w
    if BIAS:
        acc = acc + tl.load(B + c * BS, i < N, other=0).to(tl.float32)
    # Match conv-output rounding before the separate activation in the reference.
    z = acc.to(Y.dtype.element_ty).to(tl.float32)
    if ACT:
        tl.store(Z + i, z, i < N)
        z = z * tl.sigmoid(z)
    tl.store(Y + i, z, i < N)


@triton.jit
def _conv_dx(
    G,
    W,
    Z,
    DX,
    L: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    WC: tl.constexpr,
    WK: tl.constexpr,
    K: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = i % L
    c = (i // L) % C
    base = i - t
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(K):
        target = t + (K - 1) - k
        valid = (i < N) & (target < L)
        g = tl.load(G + base + target, valid, other=0).to(tl.float32)
        if ACT:
            z = tl.load(Z + base + target, valid, other=0).to(tl.float32)
            sigmoid = tl.sigmoid(z)
            g = (
                (g * sigmoid * (1 + z * (1 - sigmoid)))
                .to(G.dtype.element_ty)
                .to(tl.float32)
            )
        w = tl.load(W + c * WC + k * WK, i < N, other=0).to(tl.float32)
        acc = acc + g * w
    tl.store(DX + i, acc, i < N)


@triton.jit
def _conv_dparams(
    X,
    G,
    Z,
    P,
    L: tl.constexpr,
    BL: tl.constexpr,
    C: tl.constexpr,
    XB: tl.constexpr,
    XC: tl.constexpr,
    XT: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c, tap, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    b, t = index // L, index % L
    offset = b * C * L + c * L + t
    g = tl.load(G + offset, index < BL, other=0).to(tl.float32)
    if ACT:
        z = tl.load(Z + offset, index < BL, other=0).to(tl.float32)
        sigmoid = tl.sigmoid(z)
        g = (
            (g * sigmoid * (1 + z * (1 - sigmoid)))
            .to(G.dtype.element_ty)
            .to(tl.float32)
        )
    if tap < K:
        source = t + tap - (K - 1)
        x = tl.load(
            X + b * XB + c * XC + source * XT, (index < BL) & (source >= 0), other=0
        ).to(tl.float32)
        g = g * x
    tl.store(P + (c * (K + 1) + tap) * TILES + tile, tl.sum(g, 0))


@triton.jit
def _conv_dparams_reduce(
    P, DW, DB, K: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr
):
    c, tap = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, BLOCK)
    value = tl.sum(tl.load(P + (c * (K + 1) + tap) * TILES + i, i < TILES, other=0), 0)
    if tap < K:
        tl.store(DW + c * K + tap, value)
    else:
        tl.store(DB + c, value)


class _CausalConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, activation):
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        z = torch.empty_like(y) if activation else y
        _conv[(triton.cdiv(x.numel(), 256),)](
            x,
            w,
            b if b is not None else w,
            y,
            z,
            x.shape[2],
            x.shape[1],
            x.numel(),
            *x.stride(),
            *w.stride(),
            b.stride(0) if b is not None else 0,
            w.shape[1],
            b is not None,
            activation,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, w, z)
        ctx.has_bias = b is not None
        ctx.activation = activation
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, w, z = ctx.saved_tensors
        g = dy.contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(g)
            _conv_dx[(triton.cdiv(g.numel(), 256),)](
                g,
                w,
                z,
                dx,
                g.shape[2],
                g.shape[1],
                g.numel(),
                *w.stride(),
                w.shape[1],
                ctx.activation,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        dw = db = None
        if ctx.needs_input_grad[1] or (ctx.has_bias and ctx.needs_input_grad[2]):
            tiles = triton.cdiv(g.shape[0] * g.shape[2], 512)
            # Fixed partial reductions avoid floating-point atomic accumulation.
            partial = torch.empty(
                (w.shape[0], w.shape[1] + 1, tiles),
                device=x.device,
                dtype=torch.float32,
            )
            dw = torch.empty(w.shape, device=w.device, dtype=w.dtype)
            db = torch.empty((w.shape[0],), device=w.device, dtype=w.dtype)
            _conv_dparams[(w.shape[0], w.shape[1] + 1, tiles)](
                x,
                g,
                z,
                partial,
                x.shape[2],
                x.shape[0] * x.shape[2],
                x.shape[1],
                *x.stride(),
                w.shape[1],
                tiles,
                ctx.activation,
                512,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _conv_dparams_reduce[(w.shape[0], w.shape[1] + 1)](
                partial,
                dw,
                db,
                w.shape[1],
                tiles,
                triton.next_power_of_2(tiles),
                num_warps=4,
            )
        return dx, dw, db if ctx.has_bias else None, None


def supports_causal_conv(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, activation: str | None
) -> bool:
    return (
        x.is_cuda
        and _native(x.device)
        and x.ndim == 3
        and w.ndim == 2
        and w.device == x.device
        and w.dtype == x.dtype
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and 0 < x.numel() < 2**31
        and x.shape[1] <= 65535
        and x.shape[0] * x.shape[2] <= 1048576
        and x.shape[1] == w.shape[0]
        and 1 <= w.shape[1] <= 4
        and (
            b is None
            or (
                b.device == x.device and b.dtype == x.dtype and b.shape == (x.shape[1],)
            )
        )
        and activation in (None, "silu", "swish")
    )


def causal_conv(x, w, b=None, activation=None):
    """Depthwise causal cross-correlation, with optional SiLU and first derivatives.

    Shapes are x[B,C,L], w[C,K], b[C]. Native gfx1013 K<=4 uses scalar
    Triton; other valid calls use ordinary ATen. No global routing is installed.
    """
    if activation not in (None, "silu", "swish"):
        raise ValueError("activation must be None, silu or swish")
    if x.ndim != 3 or w.ndim != 2 or w.shape[0] != x.shape[1]:
        raise ValueError("Expected x[B,C,L] and w[C,K]")
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        x, w = [t.to(dtype) if t.dtype == torch.float32 else t for t in (x, w)]
        if b is not None and b.dtype == torch.float32:
            b = b.to(dtype)
    if supports_causal_conv(x, w, b, activation):
        return _CausalConv.apply(x, w, b, activation is not None)
    y = F.conv1d(x, w.unsqueeze(1), b, padding=w.shape[1] - 1, groups=x.shape[1])[
        ..., : x.shape[2]
    ]
    return F.silu(y) if activation is not None else y


class _ChunkCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, denominator, chunk):
        flat = logits.view(-1, logits.shape[-1])
        target = targets.reshape(-1)
        loss = torch.zeros((), dtype=torch.float32, device=logits.device)
        for start in range(0, flat.shape[0], chunk):
            loss = loss + F.cross_entropy(
                flat[start : start + chunk].float(),
                target[start : start + chunk],
                ignore_index=-100,
                reduction="sum",
            )
        ctx.save_for_backward(logits, target, denominator)
        ctx.chunk = chunk
        return loss / denominator

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        logits, target, denominator = ctx.saved_tensors
        flat = logits.view(-1, logits.shape[-1])
        grad = torch.empty_like(logits)
        gf = grad.view_as(flat)
        for start in range(0, flat.shape[0], ctx.chunk):
            labels = target[start : start + ctx.chunk]
            valid = labels != -100
            probabilities = torch.softmax(
                flat[start : start + ctx.chunk].float(), dim=-1
            )
            rows = torch.arange(labels.numel(), device=logits.device)
            probabilities[rows, labels.masked_fill(~valid, 0)] -= valid.float()
            probabilities.mul_(dout.float() / denominator)
            probabilities.masked_fill_(~valid[:, None], 0)
            gf[start : start + ctx.chunk].copy_(probabilities)
        return grad, None, None, None


def chunked_causal_loss(
    logits,
    labels,
    vocab_size=None,
    num_items_in_batch=None,
    ignore_index=-100,
    shift_labels=None,
    chunk_tokens=64,
):
    """First-order FP32 chunked causal CE; contiguous f32/f16/bf16 logits.

    Labels are shifted unless shift_labels is supplied. Only ignore_index=-100
    is supported. All-ignored mean returns NaN with zero gradients, like ATen.
    Full input logits and their gradient remain allocated.
    """
    if logits.ndim < 2 or logits.shape[-1] < 1 or not logits.is_contiguous():
        raise ValueError(
            "Chunked loss requires contiguous logits with token and vocabulary dimensions"
        )
    if (
        logits.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or ignore_index != -100
    ):
        raise ValueError("Unsupported chunked loss dtype/ignore index")
    if vocab_size is not None and logits.shape[-1] != vocab_size:
        raise ValueError("Vocab size mismatch")
    if (
        not isinstance(chunk_tokens, int)
        or isinstance(chunk_tokens, bool)
        or not 1 <= chunk_tokens <= 256
    ):
        raise ValueError("Chunk size must be1..256")
    if shift_labels is None:
        if labels.shape != logits.shape[:-1]:
            raise ValueError("Labels must match logit token shape")
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    target = shift_labels.to(logits.device)
    if target.shape != logits.shape[:-1] or target.dtype != torch.long:
        raise ValueError("Labels must match logit token shape and use int64")
    if not bool(
        ((target == -100) | ((target >= 0) & (target < logits.shape[-1]))).all()
    ):
        raise ValueError("Labels must be ignored or valid vocabulary indices")
    if num_items_in_batch is None:
        denominator = (target != -100).sum().float()
    else:
        denominator = torch.as_tensor(
            num_items_in_batch, device=logits.device, dtype=torch.float32
        )
        if (
            denominator.numel() != 1
            or denominator.requires_grad
            or not bool(torch.isfinite(denominator).all() & (denominator > 0).all())
        ):
            raise ValueError("Explicit item count must be a positive scalar")
    return _ChunkCE.apply(logits, target, denominator, chunk_tokens)
