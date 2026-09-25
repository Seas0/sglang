# SPDX-License-Identifier: Apache-2.0
"""Bit-exact data-movement kernels for the Wan causal VAE.

Both kernels only move values (plus zero fill / one same-order addition), so
their outputs are bitwise identical to the aten op chains they replace:

- :func:`cat_pad_channels_last_3d` builds a causal Conv3d input directly in
  ``channels_last_3d`` layout from a strided hidden state and an optional
  temporal feature cache, replacing ``cat + F.pad + contiguous`` (three full
  tensor passes plus the cache ``clone``/``cat`` bookkeeping) with one pass.
- :func:`dup_up3d_add` evaluates ``main + DupUp3D(src)`` in one pass,
  replacing ``repeat_interleave + permute().contiguous() + add`` (each a full
  tensor pass over the upsampled tensor).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_MAX_INT32 = 2**31 - 1


@triton.jit
def _cat_pad_cl3d_kernel(
    x_ptr,
    cache_ptr,
    out_ptr,
    keep_ptr,
    total,
    C,
    T,
    H,
    W,
    cache_t,
    out_t,
    out_h,
    out_w,
    pad_t_zero,
    pad_h,
    pad_w,
    sxb,
    sxc,
    sxt,
    sxh,
    sxw,
    scb,
    scc,
    sct,
    sch,
    scw,
    HAS_CACHE: tl.constexpr,
    KEEP_T: tl.constexpr,
    IDX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if IDX64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Output is channels_last_3d contiguous: linear index = (((b*T+t)*H+h)*W+w)*C+c
    oc = offs % C
    rest = offs // C
    ow = rest % out_w
    rest = rest // out_w
    oh = rest % out_h
    rest = rest // out_h
    o_t = rest % out_t
    ob = rest // out_t

    iw = ow - pad_w
    ih = oh - pad_h
    it = o_t - pad_t_zero

    spatial_ok = (iw >= 0) & (iw < W) & (ih >= 0) & (ih < H)
    from_cache = spatial_ok & (it >= 0) & (it < cache_t)
    from_x = spatial_ok & (it >= cache_t) & (it < cache_t + T)

    xt = it - cache_t
    x_off = ob * sxb + oc * sxc + xt * sxt + ih * sxh + iw * sxw
    vals = tl.load(x_ptr + x_off, mask=mask & from_x, other=0.0)
    if HAS_CACHE:
        c_off = ob * scb + oc * scc + it * sct + ih * sch + iw * scw
        c_vals = tl.load(cache_ptr + c_off, mask=mask & from_cache, other=0.0)
        vals = tl.where(from_cache, c_vals, vals)
    tl.store(out_ptr + offs, vals, mask=mask)
    if KEEP_T > 0:
        # Second output: the compact next-chunk feature cache = unpadded
        # interior of the last KEEP_T frames, written in the same pass
        # (channels_last_3d contiguous, laid out (B, C, KEEP_T, H, W)).
        ct = o_t - (out_t - KEEP_T)
        keep = mask & spatial_ok & (ct >= 0)
        k_off = (((ob * KEEP_T + ct) * H + ih) * W + iw) * C + oc
        tl.store(keep_ptr + k_off, vals, mask=keep)


def cat_pad_channels_last_3d(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    padding: list[int] | tuple[int, ...],
    keep_cache_t: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """``contiguous_cl3d(F.pad(cat([cache_x, x], dim=2), padding))`` in one pass.

    ``padding`` follows the ``WanCausalConv3d._padding`` convention
    ``(w_left, w_right, h_top, h_bottom, t_front, t_back)``; the temporal
    front padding is consumed by ``cache_x`` frames first and any remainder is
    zero filled (identical to the aten fallback). With ``keep_cache_t > 0``
    the same pass also emits the compact next-chunk feature cache (the
    unpadded interior of the last ``keep_cache_t`` frames) and returns the
    ``(conv_input, cache)`` pair. Returns ``None`` when the request is
    unsupported so callers can fall back.
    """
    pw_l, pw_r, ph_t, ph_b, pt_front, pt_back = padding
    if pw_l != pw_r or ph_t != ph_b or pt_back != 0:
        return None
    if x.dim() != 5 or x.device.type not in ("cuda", "xpu"):
        return None
    cache_t = 0
    if cache_x is not None:
        if (
            cache_x.dim() != 5
            or cache_x.dtype != x.dtype
            or cache_x.device != x.device
            or cache_x.shape[0] != x.shape[0]
            or cache_x.shape[1] != x.shape[1]
            or cache_x.shape[3:] != x.shape[3:]
        ):
            return None
        cache_t = cache_x.shape[2]
    pad_t_zero = pt_front - cache_t
    if pad_t_zero < 0:
        return None

    B, C, T, H, W = x.shape
    out_t = pt_front + T
    out_h = H + 2 * ph_t
    out_w = W + 2 * pw_l
    keep_t = min(keep_cache_t, out_t)
    out = torch.empty(
        (B, C, out_t, out_h, out_w),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )
    total = out.numel()
    if total == 0 or total > _MAX_INT32 * 4:
        return None
    if keep_t > 0:
        keep_arg = torch.empty(
            (B, C, keep_t, H, W),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
    else:
        keep_arg = out  # unused dummy pointer

    if cache_x is None:
        cache_arg = x  # unused dummy pointer
        scb = scc = sct = sch = scw = 0
    else:
        cache_arg = cache_x
        scb, scc, sct, sch, scw = cache_x.stride()
    sxb, sxc, sxt, sxh, sxw = x.stride()

    BLOCK = 512
    grid = (triton.cdiv(total, BLOCK),)
    with torch.get_device_module().device(x.device):
        _cat_pad_cl3d_kernel[grid](
            x,
            cache_arg,
            out,
            keep_arg,
            total,
            C,
            T,
            H,
            W,
            cache_t,
            out_t,
            out_h,
            out_w,
            pad_t_zero,
            ph_t,
            pw_l,
            sxb,
            sxc,
            sxt,
            sxh,
            sxw,
            scb,
            scc,
            sct,
            sch,
            scw,
            HAS_CACHE=cache_x is not None,
            KEEP_T=keep_t,
            IDX64=total >= _MAX_INT32,
            BLOCK=BLOCK,
        )
    if keep_cache_t > 0:
        return out, keep_arg
    return out


@triton.jit
def _dup_up3d_add_kernel(
    main_ptr,
    src_ptr,
    main_bias_ptr,
    out_ptr,
    total,
    C_out,
    out_t,
    out_h,
    out_w,
    t_offset,
    smb,
    smc,
    smt,
    smh,
    smw,
    ssb,
    ssc,
    sst,
    ssh,
    ssw,
    sob,
    soc,
    sot,
    soh,
    sow,
    FT: tl.constexpr,
    FS: tl.constexpr,
    REPEATS: tl.constexpr,
    CHANNELS_INNER: tl.constexpr,
    HAS_MAIN_BIAS: tl.constexpr,
    IDX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if IDX64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Logical (B, C_out, out_t, out_h, out_w) index; the output tensor keeps
    # ``main``'s stride order (``empty_like`` preserve), matching what the
    # aten add would produce, so downstream layout-sensitive reductions see
    # the exact same memory format. FT/FS/REPEATS are constexpr powers of two,
    # so the pixel-shuffle divisions compile to shifts. The traversal order
    # follows the output's memory order (channels innermost for NHWC-style
    # ``main``) so stores and ``main`` loads stay coalesced.
    if CHANNELS_INNER:
        oc = offs % C_out
        rest = offs // C_out
        ow = rest % out_w
        rest = rest // out_w
        oh = rest % out_h
        rest = rest // out_h
        o_t = rest % out_t
        ob = rest // out_t
    else:
        ow = offs % out_w
        rest = offs // out_w
        oh = rest % out_h
        rest = rest // out_h
        o_t = rest % out_t
        rest = rest // out_t
        oc = rest % C_out
        ob = rest // C_out

    # Undo the DupUp3D pixel-shuffle mapping (t_offset restores frames that
    # were sliced away for the first chunk).
    t2 = o_t + t_offset
    ti = t2 // FT
    rt = t2 % FT
    hi = oh // FS
    rh = oh % FS
    wi = ow // FS
    rw = ow % FS
    ch_rep = ((oc * FT + rt) * FS + rh) * FS + rw
    ci = ch_rep // REPEATS

    m_off = ob * smb + oc * smc + o_t * smt + oh * smh + ow * smw
    s_off = ob * ssb + ci * ssc + ti * sst + hi * ssh + wi * ssw
    o_off = ob * sob + oc * soc + o_t * sot + oh * soh + ow * sow
    m = tl.load(main_ptr + m_off, mask=mask, other=0.0).to(tl.float32)
    if HAS_MAIN_BIAS:
        # ``main`` is a raw conv output whose bias was deferred here: apply it
        # as aten's add_ would (fp32 opmath, one rounding to main's dtype)
        # before the residual add.
        b = tl.load(main_bias_ptr + oc, mask=mask, other=0.0).to(tl.float32)
        m = (m + b).to(out_ptr.dtype.element_ty).to(tl.float32)
    s = tl.load(src_ptr + s_off, mask=mask, other=0.0)
    # Accumulate in fp32 and round once on store, matching aten's opmath
    # behaviour for half-precision adds.
    vals = m + s.to(tl.float32)
    tl.store(out_ptr + o_off, vals, mask=mask)


def dup_up3d_add(
    main: torch.Tensor,
    src: torch.Tensor,
    factor_t: int,
    factor_s: int,
    repeats: int,
    drop_first_frames: bool,
    main_bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """``main + DupUp3D(src)`` in one pass (output layout follows ``main``).

    ``src`` is the DupUp3D input ``(B, C_in, T, H, W)``; ``main`` must match
    the DupUp3D output shape. ``drop_first_frames`` mirrors the
    ``first_chunk`` slicing (``x[:, :, factor_t - 1 :]``). ``main_bias`` (one
    value per output channel) is the deferred bias of the conv that produced
    ``main``, applied first with the rounding of ``main.add_(bias)``. Returns
    ``None`` when unsupported so callers can fall back.
    """
    if main.dim() != 5 or src.dim() != 5:
        return None
    # Power-of-two factors keep the constexpr pixel-shuffle math on the
    # shift/mask path (all Wan-family VAEs use ft in {1, 2}, fs = 2).
    if factor_t & (factor_t - 1) or factor_s & (factor_s - 1):
        return None
    if repeats <= 0 or repeats & (repeats - 1):
        return None
    if main.device.type not in ("cuda", "xpu") or src.device.type not in (
        "cuda",
        "xpu",
    ):
        return None
    if main.dtype != src.dtype or main.device != src.device:
        return None
    B, C_in, T, H, W = src.shape
    t_offset = factor_t - 1 if drop_first_frames else 0
    exp_shape = (
        B,
        C_in * repeats // (factor_t * factor_s * factor_s),
        T * factor_t - t_offset,
        H * factor_s,
        W * factor_s,
    )
    if tuple(main.shape) != exp_shape:
        return None
    if main_bias is not None:
        if (
            main_bias.device != main.device
            or main_bias.numel() != exp_shape[1]
            or main_bias.dtype not in (main.dtype, torch.float32)
        ):
            return None
        # Same cast autocast applies to a conv bias before the conv call.
        main_bias = main_bias.reshape(-1).to(main.dtype).contiguous()

    # ``empty_like`` preserves the stride order of the dense ``main`` view —
    # the same layout the aten ``main + dup`` would produce — so downstream
    # layout-sensitive reductions see the exact same memory format.
    out = torch.empty_like(main)
    total = out.numel()
    if total == 0 or total > _MAX_INT32 * 4:
        return None

    smb, smc, smt, smh, smw = main.stride()
    ssb, ssc, sst, ssh, ssw = src.stride()
    sob, soc, sot, soh, sow = out.stride()
    BLOCK = 512
    grid = (triton.cdiv(total, BLOCK),)
    with torch.get_device_module().device(main.device):
        _dup_up3d_add_kernel[grid](
            main,
            src,
            main if main_bias is None else main_bias,
            out,
            total,
            exp_shape[1],
            exp_shape[2],
            exp_shape[3],
            exp_shape[4],
            t_offset,
            smb,
            smc,
            smt,
            smh,
            smw,
            ssb,
            ssc,
            sst,
            ssh,
            ssw,
            sob,
            soc,
            sot,
            soh,
            sow,
            FT=factor_t,
            FS=factor_s,
            REPEATS=repeats,
            CHANNELS_INNER=out.stride(1) == 1 and exp_shape[1] > 1,
            HAS_MAIN_BIAS=main_bias is not None,
            IDX64=total >= _MAX_INT32,
            BLOCK=BLOCK,
        )
    return out


@triton.jit
def _avg_down3d_add_kernel(
    main_ptr,
    src_ptr,
    main_bias_ptr,
    out_ptr,
    total,
    C_out,
    out_t,
    out_h,
    out_w,
    pad_t,
    inv_group,
    smb,
    smc,
    smt,
    smh,
    smw,
    ssb,
    ssc,
    sst,
    ssh,
    ssw,
    sob,
    soc,
    sot,
    soh,
    sow,
    FT: tl.constexpr,
    FS: tl.constexpr,
    GROUP: tl.constexpr,
    CHANNELS_INNER: tl.constexpr,
    HAS_MAIN_BIAS: tl.constexpr,
    IDX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if IDX64:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Logical (B, C_out, out_t, out_h, out_w) index in the output's memory
    # order (channels innermost for an NHWC-style ``main``), as in
    # ``_dup_up3d_add_kernel``.
    if CHANNELS_INNER:
        oc = offs % C_out
        rest = offs // C_out
        ow = rest % out_w
        rest = rest // out_w
        oh = rest % out_h
        rest = rest // out_h
        o_t = rest % out_t
        ob = rest // out_t
    else:
        ow = offs % out_w
        rest = offs // out_w
        oh = rest % out_h
        rest = rest // out_h
        o_t = rest % out_t
        rest = rest // out_t
        oc = rest % C_out
        ob = rest // C_out

    # AvgDown3D: front-pad T to a multiple of FT with zero frames, pixel-
    # unshuffle (FT, FS, FS) into the channel dim, then average GROUP
    # consecutive shuffled channels. Shuffled channel k of output channel
    # ``oc`` decodes to (ci, rt, rh, rw) with k = ((ci * FT + rt) * FS + rh)
    # * FS + rw; FT/FS/GROUP are constexpr powers of two, so the div/mod
    # compile to shifts. aten's ``mean`` accumulates the group in fp32 in
    # index order and scales once, so do the same.
    FACTOR: tl.constexpr = FT * FS * FS
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for g in tl.static_range(GROUP):
        k = oc * GROUP + g
        ci = k // FACTOR
        rem = k % FACTOR
        rt = rem // (FS * FS)
        rh = (rem // FS) % FS
        rw = rem % FS
        ti = o_t * FT + rt - pad_t
        s_off = (
            ob * ssb + ci * ssc + ti * sst + (oh * FS + rh) * ssh + (ow * FS + rw) * ssw
        )
        s = tl.load(src_ptr + s_off, mask=mask & (ti >= 0), other=0.0)
        acc = acc + s.to(tl.float32)
    avg = (acc * inv_group).to(out_ptr.dtype.element_ty).to(tl.float32)

    m_off = ob * smb + oc * smc + o_t * smt + oh * smh + ow * smw
    o_off = ob * sob + oc * soc + o_t * sot + oh * soh + ow * sow
    m = tl.load(main_ptr + m_off, mask=mask, other=0.0).to(tl.float32)
    if HAS_MAIN_BIAS:
        # ``main`` is a raw conv output whose bias was deferred here (aten
        # add_: fp32 opmath, one rounding to main's dtype).
        b = tl.load(main_bias_ptr + oc, mask=mask, other=0.0).to(tl.float32)
        m = (m + b).to(out_ptr.dtype.element_ty).to(tl.float32)
    # One fp32 add, one rounding on store, like the aten ``main + avg``.
    tl.store(out_ptr + o_off, m + avg, mask=mask)


def avg_down3d_add(
    main: torch.Tensor,
    src: torch.Tensor,
    factor_t: int,
    factor_s: int,
    out_channels: int,
    main_bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """``main + AvgDown3D(src)`` in one pass (output layout follows ``main``).

    ``src`` is the AvgDown3D input ``(B, C_in, T, H, W)``; ``main`` must match
    the AvgDown3D output shape ``(B, out_channels, ceil(T / factor_t),
    H / factor_s, W / factor_s)``. The eager module front-pads T with zero
    frames, pixel-unshuffles ``(factor_t, factor_s, factor_s)`` into the
    channel dim and averages ``C_in * factor / out_channels`` consecutive
    shuffled channels; this reads ``src`` in place instead of materialising
    the padded and the shuffled copies (on a channels_last input the shuffle
    copy reads with 16x sector amplification). ``main_bias`` is the deferred
    bias of the conv that produced ``main``, applied first with the rounding
    of ``main.add_(bias)``. The group mean is accumulated in fp32 in index
    order and scaled once, as aten's ``mean`` does, so the values match the
    eager chain up to the reduction order. Returns ``None`` when unsupported
    so callers can fall back.
    """
    if main.dim() != 5 or src.dim() != 5:
        return None
    if factor_t & (factor_t - 1) or factor_s & (factor_s - 1):
        return None
    if main.device.type not in ("cuda", "xpu") or src.device.type not in (
        "cuda",
        "xpu",
    ):
        return None
    if main.dtype != src.dtype or main.device != src.device:
        return None
    B, C_in, T, H, W = src.shape
    factor = factor_t * factor_s * factor_s
    if H % factor_s or W % factor_s or (C_in * factor) % out_channels:
        return None
    group = C_in * factor // out_channels
    # Power-of-two groups keep the unshuffle math on shifts; 16 bounds the
    # unrolled accumulation (Wan 2.2 / Qwen-Image 2.1 use 1 and 4).
    if group & (group - 1) or group > 16:
        return None
    pad_t = (factor_t - T % factor_t) % factor_t
    exp_shape = (
        B,
        out_channels,
        (T + pad_t) // factor_t,
        H // factor_s,
        W // factor_s,
    )
    if tuple(main.shape) != exp_shape:
        return None
    if main_bias is not None:
        if (
            main_bias.device != main.device
            or main_bias.numel() != out_channels
            or main_bias.dtype not in (main.dtype, torch.float32)
        ):
            return None
        main_bias = main_bias.reshape(-1).to(main.dtype).contiguous()

    out = torch.empty_like(main)
    total = out.numel()
    if total == 0 or total > _MAX_INT32 * 4:
        return None

    smb, smc, smt, smh, smw = main.stride()
    ssb, ssc, sst, ssh, ssw = src.stride()
    sob, soc, sot, soh, sow = out.stride()
    BLOCK = 512
    grid = (triton.cdiv(total, BLOCK),)
    with torch.get_device_module().device(main.device):
        _avg_down3d_add_kernel[grid](
            main,
            src,
            main if main_bias is None else main_bias,
            out,
            total,
            exp_shape[1],
            exp_shape[2],
            exp_shape[3],
            exp_shape[4],
            pad_t,
            1.0 / group,
            smb,
            smc,
            smt,
            smh,
            smw,
            ssb,
            ssc,
            sst,
            ssh,
            ssw,
            sob,
            soc,
            sot,
            soh,
            sow,
            FT=factor_t,
            FS=factor_s,
            GROUP=group,
            CHANNELS_INNER=out.stride(1) == 1 and exp_shape[1] > 1,
            HAS_MAIN_BIAS=main_bias is not None,
            IDX64=total >= _MAX_INT32,
            BLOCK=BLOCK,
        )
    return out
