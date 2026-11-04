"""Quantized-weight dequantization for native MTP packages.

Customer GLM w8a8c8 packages store the large matmul weights as
``W8A8_DYNAMIC``: an int8 payload plus per-out-channel f32
``weight_scale`` (and optional f32 ``weight_offset``) siblings:

    float_weight = (int8_weight - offset) * scale

FLOAT tensors (norms, embeddings, ``lm_head``, ``shared_head.head``, the
``rot.weight`` sidecar) need no dequantization.  The folding rule below is
data-driven: any ``*.weight`` whose sibling ``*.weight_scale`` was loaded is
dequantized and the auxiliary keys are dropped from the state dict.
"""

from __future__ import annotations

import torch
from loguru import logger

__all__ = ["dequantize_w8a8", "fold_w8a8_siblings"]


def dequantize_w8a8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize a per-channel-quantized int8 weight as bfloat16.

    :param weight: int8 tensor ``[out, in]``.
    :param scale: f32 tensor ``[out, 1]`` (broadcast over the input dim).
    :param offset: optional f32 zero-point tensor ``[out, 1]``.
    """
    if weight.dtype != torch.int8:
        raise ValueError(
            f"dequantize_w8a8 expects an int8 payload, got {weight.dtype}."
        )
    result = weight.to(torch.float32)
    if offset is not None:
        result = result - offset.to(result.dtype)
    result = result * scale.to(result.dtype)
    return result.to(torch.bfloat16)


def fold_w8a8_siblings(weights: dict[str, torch.Tensor]) -> list[str]:
    """Dequantize ``*.weight`` entries that carry ``*.weight_scale`` siblings.

    Mutates ``weights`` in place: the quantized payload is replaced by its
    bfloat16 materialization and the ``weight_scale``/``weight_offset``
    siblings are removed.  Returns the list of consumed sibling keys.

    ``weight_scale_inv`` siblings (fp8 semantics) are intentionally left
    untouched -- they are handled (currently: rejected) elsewhere.
    """
    consumed: list[str] = []
    for key in sorted(weights):
        if not key.endswith(".weight"):
            continue
        scale_key = f"{key}_scale"
        scale = weights.get(scale_key)
        if scale is None:
            continue
        weight = weights[key]
        if weight.dtype != torch.int8:
            continue
        offset_key = f"{key}_offset"
        offset = weights.get(offset_key)
        weights[key] = dequantize_w8a8(weight, scale, offset)
        del weights[scale_key]
        consumed.append(scale_key)
        if offset is not None:
            del weights[offset_key]
            consumed.append(offset_key)
        logger.debug(
            f"Dequantized '{key}' (dropped {scale_key}"
            + (f", {offset_key})" if offset is not None else ")")
        )
    return consumed
