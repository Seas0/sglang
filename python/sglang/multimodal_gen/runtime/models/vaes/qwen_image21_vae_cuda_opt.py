# SPDX-License-Identifier: Apache-2.0
"""CUDA fast path for the Qwen-Image 2.1 VAE encoder and decoder.

With the gate on (``quality="extra-high"`` or ``"high"``) every convolution
takes channels_last input and lets cuDNN apply the zero padding itself, so the
explicit ``F.pad`` copy and the NCHW/NHWC transposes cuDNN otherwise inserts
around each conv disappear; every ``RMS_norm -> SiLU`` chain runs the fused
channels_last_3d Triton kernel; the 2D nearest upsample runs the NHWC gather;
and no conv adds its bias on its own. aten runs ``cudnn_convolution`` and then
a separate broadcast ``add_`` over the whole activation, so inside a residual
block the bias of ``conv1`` is deferred into the fused RMSNorm+SiLU that
consumes it, and the bias of ``conv2``, the bias of ``conv_shortcut`` and the
residual add run as one bit-exact epilogue pass; the up blocks' upsample conv
defers its bias into the fused ``main + DupUp3D(src)`` add, and the encoder's
down blocks pad in channels_last and run ``x + AvgDown3D(x_copy)`` as one fused
pass with the downsample conv's bias folded in. With the gate off the original
module code runs bit-for-bit. Installed once at VAE load;
all-or-nothing and fail-closed like the Wan-family path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
    QwenImage21AvgDown3D,
    QwenImage21CausalConv3d,
    QwenImage21DupUp3D,
    QwenImage21ResidualBlock,
    QwenImage21ResidualDownBlock,
    QwenImage21ResidualUpBlock,
    QwenImage21RMS_norm,
    QwenImage21Upsample,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.wan_vae_cuda_opt import (
    GatedChannelsLastUpsample,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from sglang.kernels.ops.diffusion import (
        avg_down3d_add,
        can_use_conv_bias_epilogue,
        can_use_wan_rmsnorm_silu,
        conv_bias_epilogue,
        dup_up3d_add,
        wan_rmsnorm_silu,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


def _as_nhwc(x: torch.Tensor) -> torch.Tensor:
    """Canonical channels_last strides; a pure re-labelling when the elements
    are already laid out NHWC (size-1 dims may carry stray strides), else one
    conversion copy at a layout boundary."""
    n, c, h, w = x.shape
    canonical = (h * w * c, 1, w * c, c)
    if x.stride() == canonical:
        return x
    if x.is_contiguous(memory_format=torch.channels_last):
        return x.as_strided(x.shape, canonical)
    return x.contiguous(memory_format=torch.channels_last)


def _nhwc_weight(conv: nn.Conv2d) -> torch.Tensor:
    """channels_last copy of the conv weight, rebuilt only when the parameter
    storage or version changes (LoRA merges, weight reloads)."""
    weight = conv.weight
    key = (weight.data_ptr(), weight._version, weight.dtype)
    cached = conv.__dict__.get("_sgl_nhwc_weight")
    if cached is None or cached[0] != key:
        cached = (key, weight.detach().contiguous(memory_format=torch.channels_last))
        conv.__dict__["_sgl_nhwc_weight"] = cached
    return cached[1]


def _channel_view(bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The broadcast view aten adds a conv bias through, cast like autocast."""
    return bias.to(x.dtype).view(1, -1, *([1] * (x.dim() - 2)))


def _add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``x.add_(bias)`` as aten runs it after ``cudnn_convolution`` (fp32
    opmath, one rounding), on the vectorised path when the layout allows."""
    if can_use_conv_bias_epilogue(x, bias):
        return conv_bias_epilogue(x, bias)
    return x + _channel_view(bias, x)


def _conv_epilogue(
    y: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    residual_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Finish a raw (bias-free) 4D channels_last conv output ``y``.

    Reproduces the eager chain ``y.add_(bias)``; ``h = residual.add_(
    residual_bias)``; ``y + h`` with the same operation order and rounding,
    in one Triton pass when the operands allow and with the aten ops
    otherwise. ``bias=None`` returns ``y`` raw so a consumer (the fused norm,
    the DupUp3D add) can absorb the bias itself.
    """
    if residual is not None:
        residual = _as_nhwc(residual.squeeze(2) if residual.dim() == 5 else residual)
    if bias is not None and can_use_conv_bias_epilogue(
        y, bias, residual, residual_bias
    ):
        return conv_bias_epilogue(y, bias, residual, residual_bias)
    if bias is not None:
        y = y.add_(_channel_view(bias, y))
    if residual is None:
        return y
    if residual_bias is not None:
        residual = residual + _channel_view(residual_bias, residual)
    return y + residual


class _GatedCausalConv3d(QwenImage21CausalConv3d):
    """``QwenImage21CausalConv3d`` whose padding moves into cuDNN and whose
    input runs channels_last while the gate is on. ``skip_bias`` returns the
    raw conv output for a consumer that folds the bias; ``residual`` (and the
    bias of the conv that produced it) are added in the same epilogue pass."""

    def forward(
        self,
        x,
        cache_x=None,
        *,
        residual: torch.Tensor | None = None,
        residual_bias: torch.Tensor | None = None,
        skip_bias: bool = False,
    ):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            y = QwenImage21CausalConv3d.forward(self, x, cache_x)
            if residual is not None:
                if residual_bias is not None:
                    residual = residual + _channel_view(residual_bias, residual)
                y = y + residual
            return y
        assert cache_x is None
        # _padding is (w, w, h, h) for F.pad; the spatial pad is symmetric
        padding = (self._padding[2], self._padding[0])
        y = F.conv2d(
            _as_nhwc(x.squeeze(2)),
            _nhwc_weight(self),
            None,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )
        y = _conv_epilogue(y, None if skip_bias else self.bias, residual, residual_bias)
        return y.unsqueeze(2)


class _GatedConv2d(nn.Conv2d):
    """Plain ``nn.Conv2d`` (Resample and attention projections) running
    channels_last while the gate is on, with its bias applied by the
    vectorised epilogue instead of aten's broadcast ``add_``. A conv marked
    ``_sgl_defer_bias`` (an up block's upsample conv, whose output only feeds
    the DupUp3D residual add) returns its raw output and the owning
    :class:`_GatedResidualUpBlock` folds the bias there."""

    def forward(self, x):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            return nn.Conv2d.forward(self, x)
        y = F.conv2d(
            _as_nhwc(x),
            _nhwc_weight(self),
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        bias = None if self.__dict__.get("_sgl_defer_bias") else self.bias
        return _conv_epilogue(y, bias, None, None)


class _GatedZeroPad2d(nn.ZeroPad2d):
    """``nn.ZeroPad2d`` (the downsamplers' asymmetric ``(0, 1, 0, 1)`` pad)
    that pads a channels_last input in channels_last while the gate is on.
    aten's ``constant_pad_nd`` writes an NCHW-contiguous result, so on an NHWC
    activation it is a strided copy with 16x read amplification, followed by
    the conv's own conversion back to NHWC; this touches the border and the
    interior once and keeps the layout. Pure data movement, bit-exact."""

    def forward(self, x):
        if (
            not self._sgl_gate.enabled
            or torch.compiler.is_compiling()
            or x.dim() != 4
            or not x.is_contiguous(memory_format=torch.channels_last)
            or min(self.padding) < 0
        ):
            return nn.ZeroPad2d.forward(self, x)
        left, right, top, bottom = self.padding
        n, c, h, w = x.shape
        out = torch.empty(
            (n, c, h + top + bottom, w + left + right),
            dtype=x.dtype,
            device=x.device,
            memory_format=torch.channels_last,
        )
        if top:
            out[:, :, :top].zero_()
        if bottom:
            out[:, :, top + h :].zero_()
        if left:
            out[:, :, :, :left].zero_()
        if right:
            out[:, :, :, left + w :].zero_()
        out[:, :, top : top + h, left : left + w].copy_(x)
        return out


class _GatedRMSNormSiLU(QwenImage21RMS_norm):
    """``QwenImage21RMS_norm`` that is always followed by SiLU. With the gate
    on it applies the SiLU itself (fused when the layout allows) and the
    paired :class:`_GatedSiLU` steps aside. ``pre_bias`` is the deferred bias
    of the conv that produced ``x``; the fused kernel adds it with the
    rounding of ``x.add_(bias)`` before taking the statistics."""

    def forward(self, x, pre_bias: torch.Tensor | None = None):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            if pre_bias is not None:
                x = x + _channel_view(pre_bias, x)
            return QwenImage21RMS_norm.forward(self, x)
        if (
            self.channel_first
            and isinstance(self.bias, float)
            and self.bias == 0
            and can_use_wan_rmsnorm_silu(x, self.gamma, None, pre_bias)
        ):
            return wan_rmsnorm_silu(
                x, self.gamma, None, rms_scale=self.scale, pre_bias=pre_bias
            )
        if pre_bias is not None:
            x = _add_bias(x, pre_bias)
        return F.silu(QwenImage21RMS_norm.forward(self, x))


class _GatedSiLU(nn.SiLU):
    def forward(self, x):
        if self._sgl_gate.enabled and not torch.compiler.is_compiling():
            return x
        return nn.SiLU.forward(self, x)


class _GatedResidualBlock(QwenImage21ResidualBlock):
    """``QwenImage21ResidualBlock`` whose convs never add their own bias while
    the gate is on: ``conv1``'s bias goes into the fused ``norm2`` + SiLU, and
    ``conv2``'s bias, ``conv_shortcut``'s bias and the residual add are one
    epilogue pass. Same arithmetic order and rounding as the eager block."""

    def forward(self, x, feat_cache=None, feat_idx=None):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            return QwenImage21ResidualBlock.forward(self, x, feat_cache, feat_idx)
        shortcut = self.conv_shortcut
        if isinstance(shortcut, _GatedCausalConv3d):
            h, h_bias = shortcut(x, skip_bias=True), shortcut.bias
        else:
            h, h_bias = shortcut(x), None
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x, skip_bias=self.conv1.bias is not None)
        x = self.norm2(x, pre_bias=self.conv1.bias)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return self.conv2(x, residual=h, residual_bias=h_bias)


class _GatedResidualUpBlock(QwenImage21ResidualUpBlock):
    """``QwenImage21ResidualUpBlock`` whose upsample conv defers its bias into
    the fused ``main + DupUp3D(src)`` add while the gate is on (the conv output
    is the largest activation of its level, so its separate bias pass was the
    costliest one left). Same arithmetic order and rounding as the eager
    block."""

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            return QwenImage21ResidualUpBlock.forward(
                self, x, feat_cache, feat_idx, first_chunk
            )
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        x = self.upsampler(x)  # raw: its conv carries ``_sgl_defer_bias``
        bias = self.upsampler.resample[1].bias
        shortcut = self.avg_shortcut
        fused = dup_up3d_add(
            x,
            x_copy,
            shortcut.factor_t,
            shortcut.factor_s,
            shortcut.repeats,
            first_chunk,
            main_bias=bias,
        )
        if fused is not None:
            return fused
        x = _add_bias(x, bias)
        return x + shortcut(x_copy, first_chunk=first_chunk)


class _GatedResidualDownBlock(QwenImage21ResidualDownBlock):
    """``QwenImage21ResidualDownBlock`` whose ``x + AvgDown3D(x_copy)`` runs
    as one fused pass over the channels_last activation while the gate is on,
    with the downsample conv's bias deferred into it: the encoder mirror of
    :class:`_GatedResidualUpBlock`. The eager shortcut materialises a padded
    and a pixel-unshuffled copy, both read with 16x sector amplification on an
    NHWC input."""

    def forward(self, x, feat_cache=None, feat_idx=None):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            return QwenImage21ResidualDownBlock.forward(self, x, feat_cache, feat_idx)
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        bias = None
        if self.downsampler is not None:
            x = self.downsampler(x)  # raw: its conv carries ``_sgl_defer_bias``
            bias = self.downsampler.resample[1].bias
        shortcut = self.avg_shortcut
        fused = avg_down3d_add(
            x,
            x_copy,
            shortcut.factor_t,
            shortcut.factor_s,
            shortcut.out_channels,
            main_bias=bias,
        )
        if fused is not None:
            return fused
        if bias is not None:
            x = _add_bias(x, bias)
        return x + shortcut(x_copy)


def _norm_silu_pairs(part: nn.Module) -> list[tuple[nn.Module, str, nn.Module, str]]:
    """(owner, norm attribute, owner, activation attribute) for every
    ``RMS_norm -> SiLU`` chain of an encoder or decoder; ``[]`` if any chain
    is non-standard so the install fails closed."""
    pairs = []
    blocks = [m for m in part.modules() if isinstance(m, QwenImage21ResidualBlock)]
    for block in blocks:
        if type(block) not in (QwenImage21ResidualBlock, _GatedResidualBlock):
            return []
        if type(block.conv_shortcut) not in (
            nn.Identity,
            QwenImage21CausalConv3d,
            _GatedCausalConv3d,
        ):
            return []
        for name in ("norm1", "norm2"):
            pairs.append((block, name, block, "nonlinearity"))
    pairs.append((part, "norm_out", part, "nonlinearity"))
    for owner, norm_name, act_owner, act_name in pairs:
        norm, act = getattr(owner, norm_name), getattr(act_owner, act_name)
        if not (
            type(norm) in (QwenImage21RMS_norm, _GatedRMSNormSiLU)
            and norm.channel_first
            and isinstance(norm.gamma, torch.Tensor)
            and isinstance(norm.bias, float)
            and type(act) in (nn.SiLU, _GatedSiLU)
            and not act.inplace
        ):
            return []
    return pairs


def _install(part: nn.Module, gate: VaeFastPathGate) -> tuple[int, int, int] | None:
    pairs = _norm_silu_pairs(part)
    if not pairs:
        return None
    for owner, norm_name, act_owner, act_name in pairs:
        norm, act = getattr(owner, norm_name), getattr(act_owner, act_name)
        # class swaps keep every parameter registered under its original name
        norm.__class__ = _GatedRMSNormSiLU
        norm._sgl_gate = gate
        act.__class__ = _GatedSiLU
        act._sgl_gate = gate
    convs = 0
    for m in part.modules():
        if type(m) is QwenImage21CausalConv3d:
            m.__class__ = _GatedCausalConv3d
        elif type(m) is nn.Conv2d:
            m.__class__ = _GatedConv2d
        elif type(m) is QwenImage21ResidualBlock:
            m.__class__ = _GatedResidualBlock
            m._sgl_gate = gate
            continue
        else:
            continue
        m._sgl_gate = gate
        convs += 1
    upsamples = 0
    for m in part.modules():
        seq = getattr(m, "resample", None)
        if isinstance(seq, nn.Sequential) and type(seq[0]) is QwenImage21Upsample:
            seq[0] = GatedChannelsLastUpsample(seq[0], gate)
            upsamples += 1
    for m in part.modules():
        if not (
            type(m) is QwenImage21ResidualUpBlock
            and m.upsampler is not None
            and type(m.avg_shortcut) is QwenImage21DupUp3D
        ):
            continue
        seq = getattr(m.upsampler, "resample", None)
        if not (
            isinstance(seq, nn.Sequential)
            and len(seq) == 2
            and type(seq[1]) is _GatedConv2d
            and seq[1].bias is not None
        ):
            continue
        seq[1]._sgl_defer_bias = True
        m.__class__ = _GatedResidualUpBlock
        m._sgl_gate = gate
    for m in part.modules():
        if not (
            type(m) is QwenImage21ResidualDownBlock
            and type(m.avg_shortcut) is QwenImage21AvgDown3D
        ):
            continue
        if m.downsampler is not None:
            seq = getattr(m.downsampler, "resample", None)
            if not (
                isinstance(seq, nn.Sequential)
                and len(seq) == 2
                and type(seq[0]) is nn.ZeroPad2d
                and type(seq[1]) is _GatedConv2d
                and seq[1].bias is not None
            ):
                continue
            seq[0].__class__ = _GatedZeroPad2d
            seq[0]._sgl_gate = gate
            seq[1]._sgl_defer_bias = True
        m.__class__ = _GatedResidualDownBlock
        m._sgl_gate = gate
    return len(pairs), convs, upsamples


def maybe_optimize_qwen_image21_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA fast path on a Qwen-Image 2.1 VAE."""
    if not isinstance(vae, AutoencoderKLQwenImage21):
        return vae
    if vae.spatial_parallel:
        logger.info("Qwen-Image 2.1 VAE: spatial-parallel decode; skipping fast path.")
        return vae
    if not _HAS_TRITON:
        logger.warning("Qwen-Image 2.1 VAE: Triton unavailable; skipping fast path.")
        return vae
    parts = [
        getattr(vae, name) for name in ("encoder", "decoder") if hasattr(vae, name)
    ]
    gate = VaeFastPathGate()
    counts = [_install(part, gate) for part in parts]
    if any(count is None for count in counts):
        logger.warning("Qwen-Image 2.1 VAE: non-standard blocks; skipping fast path.")
        return vae
    register_vae_fast_path_gate(vae, gate)
    logger.info(
        "Qwen-Image 2.1 VAE: installed quality-gated fast path (%d RMSNorm+SiLU "
        "fusions, %d channels_last convs with fused bias epilogues, "
        "%d channels_last upsamples).",
        sum(c[0] for c in counts),
        sum(c[1] for c in counts),
        sum(c[2] for c in counts),
    )
    return vae
