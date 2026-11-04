"""Internal reference helpers for the GLM DSA interleaved RoPE gate.

The GLM draft path reuses the upstream ``GlmMoeDsaRotaryEmbedding`` and
``apply_rotary_pos_emb_interleave``; these helpers exist only so the rotation
gate (plan section 22) can compare a native/reference path against a
dispatched kernel without duplicating the rotation in the model code.  They
are intentionally not exported from ``speculators.models.mtp``.
"""

from __future__ import annotations

import torch

__all__ = ["compare_rotary_paths", "reference_glm_interleaved_rope"]


def reference_glm_interleaved_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply GLM DSA interleaved RoPE to ``query``/``key``.

    Dimensions are rotated in interleaved pairs ``(x0, x1), (x2, x3), ...``

        ``y[2j]   = x[2j]   * cos[j] - x[2j+1] * sin[j]``
        ``y[2j+1] = x[2j+1] * cos[j] + x[2j]   * sin[j]``

    :param query: ``[..., rotary_dim]`` query tensor.
    :param key: ``[..., rotary_dim]`` key tensor.
    :param cos: per-pair cosine angles broadcastable to ``query[..., :rotary_dim]``.
    :param sin: per-pair sine angles with the same shape as ``cos``.
    :param rotary_dim: Number of leading dimensions to rotate; the tail is
        copied unchanged for partial rotary.  Defaults to the full last dim.
    """
    rotary_dim = query.shape[-1] if rotary_dim is None else rotary_dim
    if rotary_dim <= 0 or rotary_dim > query.shape[-1]:
        raise ValueError(
            f"rotary_dim must be in (0, {query.shape[-1]}], got {rotary_dim}."
        )
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}.")
    if key.shape[-1] < rotary_dim:
        raise ValueError(
            f"key last dim {key.shape[-1]} is smaller than rotary_dim {rotary_dim}."
        )

    q_rot = _rotate_interleaved(query[..., :rotary_dim], cos, sin)
    k_rot = _rotate_interleaved(key[..., :rotary_dim], cos, sin)
    if rotary_dim == query.shape[-1]:
        return q_rot, k_rot
    return (
        torch.cat([q_rot, query[..., rotary_dim:]], dim=-1),
        torch.cat([k_rot, key[..., rotary_dim:]], dim=-1),
    )


def _rotate_interleaved(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    if cos.shape[-1] != even.shape[-1] or sin.shape[-1] != even.shape[-1]:
        raise ValueError(
            "cos/sin must hold one angle per interleaved pair: "
            f"got {cos.shape[-1]}/{sin.shape[-1]} angles for "
            f"{even.shape[-1]} pairs."
        )
    return torch.cat([even * cos - odd * sin, odd * cos + even * sin], dim=-1)


def compare_rotary_paths(
    native_q: torch.Tensor,
    native_k: torch.Tensor,
    dispatched_q: torch.Tensor,
    dispatched_k: torch.Tensor,
) -> dict[str, float]:
    """Return the rotation-gate metrics for native vs dispatched Q/K.

    ``*_norm_ratio_native`` is ``native.norm() / dispatched.norm()`` and
    ``*_norm_ratio_dispatched`` is its reciprocal, so a clean path reports
    both as ``1.0`` and a global kernel scaling (the community ``~0.73``
    issue) shows up directly.
    """
    metrics: dict[str, float] = {}
    for name, native, dispatched in (
        ("q", native_q, dispatched_q),
        ("k", native_k, dispatched_k),
    ):
        if native.shape != dispatched.shape:
            raise ValueError(
                f"{name} shapes differ: {native.shape} != {dispatched.shape}."
            )
        if native.dtype != dispatched.dtype:
            raise ValueError(
                f"{name} dtypes differ: {native.dtype} != {dispatched.dtype}."
            )
        native_fp32 = native.float()
        dispatched_fp32 = dispatched.float()
        diff = (native_fp32 - dispatched_fp32).abs()
        denom = dispatched_fp32.abs().clamp_min(torch.finfo(torch.float32).tiny)
        native_norm = float(native_fp32.norm())
        dispatched_norm = float(dispatched_fp32.norm())
        metrics[f"{name}_max_abs"] = float(diff.max()) if diff.numel() else 0.0
        metrics[f"{name}_max_rel"] = (
            float((diff / denom).max()) if diff.numel() else 0.0
        )
        if native_norm == 0.0 and dispatched_norm == 0.0:
            metrics[f"{name}_norm_ratio_native"] = 1.0
            metrics[f"{name}_norm_ratio_dispatched"] = 1.0
        elif dispatched_norm == 0.0 or native_norm == 0.0:
            metrics[f"{name}_norm_ratio_native"] = float("inf")
            metrics[f"{name}_norm_ratio_dispatched"] = 0.0
        else:
            metrics[f"{name}_norm_ratio_native"] = native_norm / dispatched_norm
            metrics[f"{name}_norm_ratio_dispatched"] = dispatched_norm / native_norm
    return metrics
