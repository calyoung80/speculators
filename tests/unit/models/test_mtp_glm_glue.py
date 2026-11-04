"""Unit tests for the thin GLM-MoE-DSA MTP layer."""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers.models.glm_moe_dsa")

from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from transformers.masking_utils import create_causal_mask
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaExperts,
    GlmMoeDsaMoE,
    GlmMoeDsaRotaryEmbedding,
)

from speculators.models.mtp.glm_glue import GlmMTPLayer

BATCH = 1
SEQ_LEN = 128
HIDDEN = 64

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"


def load_tiny_config(name: str = "tiny_config.json") -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig.from_pretrained(FIXTURE_DIR / name)


@pytest.fixture
def config() -> GlmMoeDsaConfig:
    cfg = load_tiny_config()
    cfg._attn_implementation = "sdpa"
    return cfg


@pytest.fixture
def layer(config) -> GlmMTPLayer:
    torch.manual_seed(0)
    module = GlmMTPLayer(config, layer_idx=0)
    # The upstream experts store uninitialized ``torch.empty`` weights.
    module.mlp.experts.gate_up_proj.data.normal_(std=0.02)
    module.mlp.experts.down_proj.data.normal_(std=0.02)
    module.eval()
    return module


def make_inputs(config, hidden_states=None, token_embeddings=None):
    torch.manual_seed(1)
    if hidden_states is None:
        hidden_states = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    if token_embeddings is None:
        token_embeddings = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    rotary = GlmMoeDsaRotaryEmbedding(config)
    position_embeddings = rotary(hidden_states, position_ids)
    return hidden_states, token_embeddings, position_ids, position_embeddings


def test_forward_signature():
    params = inspect.signature(GlmMTPLayer.forward).parameters
    assert list(params)[1:6] == [
        "hidden_states",
        "token_embeddings",
        "attention_mask",
        "position_ids",
        "position_embeddings",
    ]
    assert params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_head_dim_matches_rope(config, layer):
    assert config.head_dim == config.qk_rope_head_dim
    layer_config = layer.self_attn.config
    assert layer_config.head_dim == layer_config.qk_rope_head_dim == 16


def test_nonzero_layer_idx_is_rejected(config):
    with pytest.raises(ValueError, match="layer_idx"):
        GlmMTPLayer(config, layer_idx=1)


def test_forward_shape_and_backward(layer, config):
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config
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


def test_mtp_modules_are_created(layer):
    for name in (
        "hidden_layernorm",
        "token_layernorm",
        "input_proj",
        "final_layernorm",
    ):
        assert hasattr(layer, name), name
    assert layer.input_proj.in_features == 2 * HIDDEN
    assert layer.input_proj.out_features == HIDDEN


def test_mode_is_full_indexer_and_sparse_moe(layer):
    assert layer.self_attn.config.indexer_types == ["full"]
    assert layer.self_attn.config.mlp_layer_types == ["sparse"]
    assert layer.self_attn.skip_topk is False
    assert layer.self_attn.indexer is not None
    assert isinstance(layer.mlp, GlmMoeDsaMoE)


def test_indexer_is_frozen(layer):
    indexer = layer.self_attn.indexer
    assert indexer is not None
    assert all(not parameter.requires_grad for parameter in indexer.parameters())


def test_sparse_topk_is_not_full_attention(layer, config, monkeypatch):
    captured = {}
    original_forward = layer.self_attn.indexer.forward

    def spy(*args, **kwargs):
        topk_indices = original_forward(*args, **kwargs)
        captured["topk_indices"] = topk_indices
        return topk_indices

    monkeypatch.setattr(layer.self_attn.indexer, "forward", spy)
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config
    )
    with torch.no_grad():
        layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    topk_indices = captured["topk_indices"]
    assert topk_indices.shape == (BATCH, SEQ_LEN, config.index_topk)
    assert config.index_topk < SEQ_LEN
    assert topk_indices.min() >= 0
    assert topk_indices.max() < SEQ_LEN
    # Not every key position is selected: the DSA path really is sparse.
    assert topk_indices.unique().numel() < SEQ_LEN


def test_attention_mask_none_matches_explicit_causal_mask(layer, config):
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config
    )
    additive = torch.full(
        (BATCH, 1, SEQ_LEN, SEQ_LEN),
        torch.finfo(hidden_states.dtype).min,
        dtype=hidden_states.dtype,
    )
    additive = torch.triu(additive, diagonal=1)

    with torch.no_grad():
        without_mask = layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        with_mask = layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=additive,
        )

    torch.testing.assert_close(without_mask, with_mask)


def test_create_causal_mask_none_and_4d_branches(config):
    inputs_embeds = torch.zeros(BATCH, SEQ_LEN, HIDDEN)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    no_padding = create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    assert no_padding is None

    padding = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    padding[:, -1] = 0
    with_padding = create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=padding,
        past_key_values=None,
        position_ids=position_ids,
    )
    assert with_padding is not None
    assert with_padding.ndim == 4


def test_position_zero_embedding_is_ignored(layer, config):
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config
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
        config
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


def test_return_dict_is_ignored(layer, config):
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config
    )
    with torch.no_grad():
        with_dict = layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            return_dict=True,
        )
        without_dict = layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            return_dict=False,
        )
    assert torch.is_tensor(with_dict)
    torch.testing.assert_close(with_dict, without_dict)


def test_eager_experts_dispatch(layer, config, monkeypatch):
    seen = []
    original = ALL_EXPERTS_FUNCTIONS.get_interface

    def spy(experts_implementation, default):
        seen.append(experts_implementation)
        return original(experts_implementation, default)

    monkeypatch.setattr(ALL_EXPERTS_FUNCTIONS, "get_interface", spy)
    hidden_states, token_embeddings, position_ids, position_embeddings = make_inputs(
        config
    )
    with torch.no_grad():
        layer(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    assert layer.mlp.experts.config._experts_implementation == "eager"
    assert "eager" in seen


def test_experts_parity_with_upstream(layer):
    reference = GlmMoeDsaExperts(layer.mlp.experts.config)
    reference.load_state_dict(layer.mlp.experts.state_dict())

    torch.manual_seed(2)
    hidden = torch.randn(16, HIDDEN)
    topk_index = torch.randint(0, 4, (16, 2), dtype=torch.long)
    topk_weights = torch.rand(16, 2)

    actual_hidden = hidden.clone().requires_grad_(True)
    reference_hidden = hidden.clone().requires_grad_(True)
    actual = layer.mlp.experts(actual_hidden, topk_index, topk_weights)
    expected = reference(reference_hidden, topk_index, topk_weights)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        actual_hidden.grad, reference_hidden.grad, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        layer.mlp.experts.gate_up_proj.grad,
        reference.gate_up_proj.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_rot_is_absent_in_upstream_layer(layer, config):
    # The community GLM-MoE-DSA checkpoint has ``has_rot_weight=false`` and the
    # upstream decoder layer has no ``rot`` projection. A rot-bearing native
    # checkpoint must be handled by the converter, not invented here.
    assert not hasattr(layer, "rot")
    assert not hasattr(config, "is_rot_used")


def test_hub_kernels_can_be_disabled_before_import():
    code = (
        "import inspect\n"
        "from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import "
        "GlmMoeDsaRMSNorm\n"
        "source = inspect.getsource(GlmMoeDsaRMSNorm.forward)\n"
        "assert 'variance_epsilon' in source, source\n"
        "print('ok')\n"
    )
    env = os.environ.copy()
    env["USE_HUB_KERNELS"] = "0"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_source_config_is_not_mutated(config):
    original_mlp = list(config.mlp_layer_types)
    original_indexer = list(config.indexer_types)
    GlmMTPLayer(config, layer_idx=0)
    assert config.mlp_layer_types == original_mlp
    assert config.indexer_types == original_indexer
