# SPDX-License-Identifier: Apache-2.0
"""``conv_bias_epilogue``: bit-exact vs the aten ``conv_out.add_(bias)`` and
``conv(x) + h`` chains it replaces on channels_last VAE conv outputs."""

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_conv_bias_epilogue,
    conv_bias_epilogue,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# (shape, memory format): a 2D conv output (Qwen-Image 2.1 VAE, T squeezed)
# and a causal 3D one (Wan VAE); odd channel counts and a ragged tail.
SHAPES = [
    pytest.param((1, 144, 40, 56), torch.channels_last, id="nhwc_144"),
    pytest.param((2, 1152, 9, 7), torch.channels_last, id="nhwc_1152"),
    pytest.param((1, 96, 3, 20, 22), torch.channels_last_3d, id="ndhwc_96"),
]


def _conv_out(shape, memory_format, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    return x.contiguous(memory_format=memory_format)


def _aten_chain(x, bias, residual=None, residual_bias=None):
    """The eager ops, in eager order: ``add_`` on the conv output, the
    shortcut conv's own ``add_``, then ``conv(x) + h``."""
    view = (1, -1) + (1,) * (x.dim() - 2)
    y = x.clone().add_(bias.to(x.dtype).view(view))
    if residual is None:
        return y
    h = residual
    if residual_bias is not None:
        h = h.clone().add_(residual_bias.to(x.dtype).view(view))
    return y + h


_BITS = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
}


def _assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    bits = _BITS[actual.dtype]
    assert torch.equal(actual.contiguous().view(bits), expected.contiguous().view(bits))


@torch.no_grad()
@pytest.mark.parametrize("shape,memory_format", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_dtype", ["same", torch.float32])
def test_bias_only_matches_aten_add(shape, memory_format, dtype, bias_dtype):
    torch.manual_seed(0)
    x = _conv_out(shape, memory_format, dtype)
    bias = torch.randn(
        shape[1], device="cuda", dtype=dtype if bias_dtype == "same" else bias_dtype
    )
    original = x.clone()
    assert can_use_conv_bias_epilogue(x, bias)
    _assert_bits_equal(conv_bias_epilogue(x, bias), _aten_chain(x, bias))
    # the input is left untouched (the aten add_ mutates; this one allocates)
    assert torch.equal(x, original)


@torch.no_grad()
@pytest.mark.parametrize("shape,memory_format", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_residual_bias", [False, True])
def test_residual_matches_two_rounding_chain(
    shape, memory_format, dtype, with_residual_bias
):
    torch.manual_seed(1)
    x = _conv_out(shape, memory_format, dtype)
    bias = torch.randn(shape[1], device="cuda", dtype=dtype)
    residual = _conv_out(shape, memory_format, dtype)
    residual_bias = (
        torch.randn(shape[1], device="cuda", dtype=dtype)
        if with_residual_bias
        else None
    )
    assert can_use_conv_bias_epilogue(x, bias, residual, residual_bias)
    actual = conv_bias_epilogue(x, bias, residual, residual_bias)
    _assert_bits_equal(actual, _aten_chain(x, bias, residual, residual_bias))


@torch.no_grad()
def test_half_precision_intermediate_rounding_is_preserved():
    # Values chosen so that fusing the two adds into one fp32 add would round
    # differently from aten's ``bf16(bf16(x + b) + h)``: x + b lands exactly
    # halfway between two bf16 values (ties-to-even drops it back to x), so a
    # single rounding of x + b + h would land on the other side.
    x = torch.full((1, 8, 2, 2), 1.0, device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last)
    bias = torch.full((8,), 2**-8, device="cuda", dtype=torch.bfloat16)
    residual = torch.full_like(x, 2**-9)
    fused_once = (x.float() + bias.float().view(1, -1, 1, 1) + residual.float()).to(
        torch.bfloat16
    )
    expected = _aten_chain(x, bias, residual)
    assert not torch.equal(fused_once, expected), "test values do not discriminate"
    _assert_bits_equal(conv_bias_epilogue(x, bias, residual), expected)


@torch.no_grad()
def test_predicate_rejects_unsupported_inputs():
    x = torch.randn(1, 16, 4, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)
    nhwc = x.contiguous(memory_format=torch.channels_last)
    assert can_use_conv_bias_epilogue(nhwc, bias)
    # NCHW-contiguous: the flat channel modulo would be wrong
    assert not can_use_conv_bias_epilogue(x, bias)
    # a single channel is both NCHW- and NHWC-contiguous; stay out
    one = torch.randn(1, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
    assert not can_use_conv_bias_epilogue(one, bias[:1])
    # bias shape / dtype
    assert not can_use_conv_bias_epilogue(nhwc, bias[:8])
    assert not can_use_conv_bias_epilogue(nhwc, bias.half())
    # residual must match shape, dtype and strides exactly
    assert not can_use_conv_bias_epilogue(nhwc, bias, x)
    assert not can_use_conv_bias_epilogue(nhwc, bias, nhwc.float())
    assert not can_use_conv_bias_epilogue(nhwc, bias, nhwc[:, :, :2].contiguous())
    # a residual bias without a residual makes no sense
    assert not can_use_conv_bias_epilogue(nhwc, bias, None, bias)
    # 3D / 6D tensors are not conv outputs this kernel understands
    assert not can_use_conv_bias_epilogue(nhwc.flatten(2), bias)
    with pytest.raises(ValueError):
        conv_bias_epilogue(x, bias)


def test_predicate_rejects_grad_tensors():
    x = torch.randn(1, 16, 4, 4, device="cuda", requires_grad=True)
    x = x.contiguous(memory_format=torch.channels_last)
    bias = torch.randn(16, device="cuda")
    assert not can_use_conv_bias_epilogue(x, bias)
    with torch.no_grad():
        assert can_use_conv_bias_epilogue(x, bias)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
