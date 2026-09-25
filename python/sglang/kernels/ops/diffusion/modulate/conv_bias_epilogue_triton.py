# SPDX-License-Identifier: Apache-2.0
"""Bit-exact conv bias epilogue (optionally with a residual) for channels_last
VAE convolutions.

``F.conv2d`` / ``F.conv3d`` on CUDA run ``cudnn_convolution`` without the bias
and then ``output.add_(bias.view(1, C, 1, ...))`` as a separate broadcast
kernel. On a channels_last output that broadcast is not on aten's vectorised
path, and in a VAE residual block it is followed by yet another full pass for
``conv(x) + h``. This kernel replaces the whole tail of the block with one
read of each operand and one write, reproducing aten's arithmetic exactly:

- ``conv_bias_epilogue(x, bias)``: ``x.dtype(fp32(x) + fp32(bias))`` per
  element, i.e. ``x.add_(bias)`` (fp32 opmath, one rounding).
- ``conv_bias_epilogue(x, bias, residual)``: the eager ``conv(x) + h`` chain,
  ``x.dtype(x.dtype(x + bias) + h)``; the biased conv output is rounded to
  ``x.dtype`` before the residual add, so the result is bitwise equal to the
  two-kernel chain.
- ``conv_bias_epilogue(x, bias, residual, residual_bias)``: the residual is
  itself a raw conv output (a ``conv_shortcut``) whose bias is folded in the
  same way, ``x.dtype(x.dtype(x + bias) + x.dtype(h + residual_bias))``.

Biases may be fp32 while ``x`` is half precision; they are cast to ``x.dtype``
first, which is what autocast does before the conv call. The output is
``empty_like(x)`` and so keeps ``x``'s dense channels_last(_3d) strides,
exactly like the in-place aten ``add_``. Support is a predicate
(:func:`can_use_conv_bias_epilogue`); the kernel raises on unsupported input.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_BLOCK = 2048
_INT32_LIMIT = 2**31 - 1


@triton.jit
def _conv_bias_epilogue_kernel(
    x_ptr,
    bias_ptr,
    res_ptr,
    res_bias_ptr,
    out_ptr,
    numel,
    channels,
    HAS_RES: tl.constexpr,
    HAS_RES_BIAS: tl.constexpr,
    INT64_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if INT64_INDEX:
        start = tl.program_id(0).to(tl.int64) * BLOCK
        offsets = start + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    # Dense channels_last: the channel index is the fastest-varying one, so a
    # flat traversal keeps every load and store coalesced and the bias lookup
    # is a single modulo.
    channel = offsets % channels
    out_dtype = out_ptr.dtype.element_ty

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    # aten ``add_``: fp32 opmath, one rounding to the tensor dtype.
    y = (x + bias).to(out_dtype)
    if HAS_RES:
        h = tl.load(res_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RES_BIAS:
            res_bias = tl.load(res_bias_ptr + channel, mask=mask, other=0.0)
            h = (h + res_bias.to(tl.float32)).to(out_dtype).to(tl.float32)
        # ``conv(x) + h``: both operands are already rounded tensors, so the
        # add is one more fp32 op with one more rounding.
        y = (y.to(tl.float32) + h).to(out_dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _dense_channels_last(x: torch.Tensor) -> bool:
    """Canonical dense NHWC / NDHWC strides on every dim, with ``C > 1`` (a
    single channel is also NCHW-contiguous, and the flat channel modulo
    needs ``stride(C) == 1`` to be unambiguous)."""
    if x.dim() == 4:
        _, c, h, w = x.shape
        return c > 1 and x.stride() == (h * w * c, 1, w * c, c)
    if x.dim() == 5:
        _, c, t, h, w = x.shape
        return c > 1 and x.stride() == (t * h * w * c, 1, h * w * c, w * c, c)
    return False


def _bias_ok(x: torch.Tensor, bias: object) -> bool:
    return (
        isinstance(bias, torch.Tensor)
        and bias.is_cuda
        and bias.device == x.device
        and bias.numel() == x.shape[1]
        and (bias.dtype == x.dtype or bias.dtype == torch.float32)
    )


def can_use_conv_bias_epilogue(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
) -> bool:
    """True when ``x.add_(bias)`` (optionally ``+ residual``, whose own bias
    may be folded too) can run as the fused kernel with identical values and
    layout. Never raises."""
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        # no autograd here: refuse whenever a gradient could be recorded
        and not (torch.is_grad_enabled() and x.requires_grad)
        and x.dtype in _SUPPORTED_DTYPES
        and x.numel() > 0
        and _dense_channels_last(x)
        and _bias_ok(x, bias)
    ):
        return False
    if residual is None:
        return residual_bias is None
    return (
        isinstance(residual, torch.Tensor)
        and residual.device == x.device
        and residual.dtype == x.dtype
        and residual.shape == x.shape
        and residual.stride() == x.stride()
        and not (torch.is_grad_enabled() and residual.requires_grad)
        and (residual_bias is None or _bias_ok(x, residual_bias))
    )


def conv_bias_epilogue(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor | None = None,
    residual_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x.dtype(x + bias)``, or ``x.dtype(x.dtype(x + bias) + residual)`` with
    the residual's own bias folded the same way, on a dense channels_last
    tensor; bit-exact vs the aten chain. Guard with
    :func:`can_use_conv_bias_epilogue`."""
    if not can_use_conv_bias_epilogue(x, bias, residual, residual_bias):
        raise ValueError(
            "unsupported input for conv_bias_epilogue: needs a CUDA dense "
            "channels_last(_3d) bf16/fp16/fp32 tensor with C > 1, a bias of "
            "C elements, and a residual of the same shape, dtype and strides; "
            f"got shape {tuple(x.shape)} strides {tuple(x.stride())} {x.dtype}"
        )
    channels = x.shape[1]
    # Same cast autocast applies to the bias before the conv call.
    bias = bias.reshape(channels).to(x.dtype).contiguous()
    if residual_bias is not None:
        residual_bias = residual_bias.reshape(channels).to(x.dtype).contiguous()
    out = torch.empty_like(x)
    numel = out.numel()
    grid = (triton.cdiv(numel, _BLOCK),)
    with torch.get_device_module().device(x.device):
        _conv_bias_epilogue_kernel[grid](
            x,
            bias,
            x if residual is None else residual,
            bias if residual_bias is None else residual_bias,
            out,
            numel,
            channels,
            HAS_RES=residual is not None,
            HAS_RES_BIAS=residual_bias is not None,
            INT64_INDEX=numel > _INT32_LIMIT,
            BLOCK=_BLOCK,
        )
    return out


__all__ = ["can_use_conv_bias_epilogue", "conv_bias_epilogue"]
