"""Interleaved-RoPE reference gate for GLM DSA (plan section 22.3)."""

import pytest
import torch

pytest.importorskip("transformers.models.glm_moe_dsa")

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRotaryEmbedding,
    apply_rotary_pos_emb_interleave,
)

from speculators.models.mtp.rope_reference import (
    compare_rotary_paths,
    reference_glm_interleaved_rope,
)


def make_rotary(rotary_dim: int) -> GlmMoeDsaRotaryEmbedding:
    config = GlmMoeDsaConfig(
        hidden_size=rotary_dim,
        num_attention_heads=1,
        qk_rope_head_dim=rotary_dim,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    return GlmMoeDsaRotaryEmbedding(config)


def make_cos_sin(rotary_dim: int, batch: int = 2, seq: int = 8):
    rotary = make_rotary(rotary_dim)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    x = torch.zeros(batch, seq, rotary_dim)
    return rotary(x, position_ids)


@pytest.mark.parametrize("rotary_dim", [64, 96, 192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_matches_upstream_interleave(rotary_dim, dtype):
    torch.manual_seed(0)
    batch, heads, seq = 2, 4, 8
    query = torch.randn(batch, heads, seq, rotary_dim, dtype=dtype)
    key = torch.randn(batch, heads, seq, rotary_dim, dtype=dtype)
    cos, sin = make_cos_sin(rotary_dim, batch, seq)
    cos, sin = cos.to(dtype), sin.to(dtype)

    expected_q, expected_k = apply_rotary_pos_emb_interleave(query, key, cos, sin)
    actual_q, actual_k = reference_glm_interleaved_rope(
        query,
        key,
        cos[..., : rotary_dim // 2].unsqueeze(1),
        sin[..., : rotary_dim // 2].unsqueeze(1),
        rotary_dim,
    )

    assert actual_q.dtype == expected_q.dtype == dtype
    assert actual_k.dtype == expected_k.dtype == dtype
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


def test_partial_rotary_leaves_tail_unchanged():
    torch.manual_seed(0)
    batch, heads, seq, head_dim, rotary_dim = 1, 2, 6, 32, 16
    query = torch.randn(batch, heads, seq, head_dim)
    key = torch.randn(batch, heads, seq, head_dim)
    cos, sin = make_cos_sin(rotary_dim, batch, seq)

    out_q, out_k = reference_glm_interleaved_rope(
        query,
        key,
        cos[..., : rotary_dim // 2].unsqueeze(1),
        sin[..., : rotary_dim // 2].unsqueeze(1),
        rotary_dim,
    )

    assert torch.equal(out_q[..., rotary_dim:], query[..., rotary_dim:])
    assert torch.equal(out_k[..., rotary_dim:], key[..., rotary_dim:])
    assert not torch.equal(out_q[..., :rotary_dim], query[..., :rotary_dim])


def test_interleaved_pairing_differs_from_neox_half_pairing():
    torch.manual_seed(0)
    seq, head_dim = 1, 8
    query = torch.arange(head_dim, dtype=torch.float32).reshape(1, 1, 1, head_dim)
    key = query.clone()
    cos, sin = make_cos_sin(head_dim, 1, seq)
    cos = cos[..., : head_dim // 2].unsqueeze(1)
    sin = sin[..., : head_dim // 2].unsqueeze(1)

    interleaved_q, _ = reference_glm_interleaved_rope(query, key, cos, sin)

    half = head_dim // 2
    neox_q = torch.cat(
        [
            query[..., :half] * cos - query[..., half:] * sin,
            query[..., half:] * cos + query[..., :half] * sin,
        ],
        dim=-1,
    )
    assert not torch.equal(interleaved_q, neox_q)


def test_reference_rejects_bad_shapes():
    query = torch.randn(1, 1, 4, 64)
    key = torch.randn(1, 1, 4, 64)
    cos = torch.ones(1, 1, 4, 31)
    sin = torch.ones(1, 1, 4, 31)

    with pytest.raises(ValueError, match="even"):
        reference_glm_interleaved_rope(query, key, cos, sin, rotary_dim=31)
    with pytest.raises(ValueError, match="one angle per interleaved pair"):
        reference_glm_interleaved_rope(query, key, cos, sin, rotary_dim=64)


def test_compare_rotary_paths_clean_and_scaled():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 64)
    k = torch.randn(2, 4, 8, 64)

    clean = compare_rotary_paths(q, k, q.clone(), k.clone())
    for name in ("q", "k"):
        assert clean[f"{name}_max_abs"] == 0.0
        assert clean[f"{name}_max_rel"] == 0.0
        assert clean[f"{name}_norm_ratio_native"] == pytest.approx(1.0)
        assert clean[f"{name}_norm_ratio_dispatched"] == pytest.approx(1.0)

    scaled = compare_rotary_paths(q, k, q * 0.73, k * 0.73)
    assert scaled["q_norm_ratio_native"] == pytest.approx(1.0 / 0.73, rel=1e-5)
    assert scaled["q_norm_ratio_dispatched"] == pytest.approx(0.73, rel=1e-5)
    assert abs(scaled["q_norm_ratio_dispatched"] - 1.0) > 1e-2


def test_compare_rotary_paths_requires_matching_dtype():
    q = torch.randn(1, 1, 2, 8)
    with pytest.raises(ValueError, match="dtypes differ"):
        compare_rotary_paths(q, q, q.bfloat16(), q.bfloat16())


def test_compare_rotary_paths_requires_matching_shape():
    q = torch.randn(1, 1, 2, 8)
    with pytest.raises(ValueError, match="shapes differ"):
        compare_rotary_paths(q, q, torch.randn(1, 1, 3, 8), q)
