"""Unit tests for the thin GLM-MoE-DSA MTP layer."""

import pytest
import torch

pytest.importorskip("transformers.models.glm_moe_dsa")

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaRotaryEmbedding,
)

from speculators.models.mtp.glm_glue import GlmMTPLayer  # noqa: E402

BATCH = 1
SEQ_LEN = 128
HIDDEN = 64


def make_tiny_config() -> GlmMoeDsaConfig:
    config = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        routed_scaling_factor=2.5,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        rms_norm_eps=1e-5,
        first_k_dense_replace=1,
        index_topk=32,
        index_head_dim=16,
        index_n_heads=4,
        max_position_embeddings=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "sdpa"
    return config


@pytest.fixture
def config() -> GlmMoeDsaConfig:
    return make_tiny_config()


@pytest.fixture
def layer(config) -> GlmMTPLayer:
    torch.manual_seed(0)
    module = GlmMTPLayer(config, layer_idx=0)
    # The upstream experts store uninitialized ``torch.empty`` weights.
    module.mlp.experts.gate_up_proj.data.normal_(std=0.02)
    module.mlp.experts.down_proj.data.normal_(std=0.02)
    module.eval()
    return module


def make_inputs(config, layer):
    torch.manual_seed(1)
    hidden_states = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    token_embeddings = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    rotary = GlmMoeDsaRotaryEmbedding(config)
    position_embeddings = rotary(hidden_states, position_ids)
    return hidden_states, token_embeddings, position_ids, position_embeddings


def test_forward_shape_and_backward(layer, config):
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config, layer
    )
    hidden_states.requires_grad_(True)
    token_embeddings.requires_grad_(True)

    output = layer(
        hidden_states=hidden_states,
        token_embeddings=token_embeddings,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )

    assert output.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert torch.isfinite(output).all()

    output.sum().backward()
    assert hidden_states.grad is not None
    assert token_embeddings.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_mla_norm_eps_matches_config(layer, config):
    assert layer.self_attn.q_a_layernorm.variance_epsilon == config.rms_norm_eps
    assert layer.self_attn.kv_a_layernorm.variance_epsilon == config.rms_norm_eps
    assert config.rms_norm_eps != 1e-6


def test_indexer_is_frozen(layer):
    indexer = layer.self_attn.indexer
    assert indexer is not None
    assert all(not parameter.requires_grad for parameter in indexer.parameters())


def test_position_zero_embedding_is_ignored(layer, config):
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config, layer
    )
    perturbed = token_embeddings.clone()
    perturbed[:, 0] += 5.0

    with torch.no_grad():
        base = layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        changed = layer(
            hidden_states=hidden_states,
            token_embeddings=perturbed,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    torch.testing.assert_close(base, changed)


def test_cache_kwargs_are_forced_off(layer, config):
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config, layer
    )
    with torch.no_grad():
        output = layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=object(),
            use_cache=True,
            prev_topk_indices=torch.zeros(BATCH, 1, dtype=torch.long),
        )
    assert output.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert torch.isfinite(output).all()


def test_source_config_is_not_mutated(config):
    original_mlp = list(config.mlp_layer_types)
    GlmMTPLayer(config, layer_idx=0)
    assert config.mlp_layer_types == original_mlp
