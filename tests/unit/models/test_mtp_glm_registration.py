"""Registration tests for the GLM-MoE-DSA MTP layer."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("transformers.models.glm_moe_dsa")

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding,
)

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.models.mtp import (
    MTPDraftModel,
    MTPSpeculatorConfig,
    model_definitions,
)
from speculators.models.mtp.capabilities import (
    MTP_CAPABILITIES,
    capability_ignore_keys,
)
from speculators.models.mtp.glm_glue import GlmMTPLayer
from speculators.proposals import GreedyTokenProposalConfig

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"


def load_tiny_config(name: str = "tiny_config.json") -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig.from_pretrained(FIXTURE_DIR / name)


def make_speculator_config(transformer_config: GlmMoeDsaConfig):
    return MTPSpeculatorConfig(
        transformer_layer_config=transformer_config,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["GlmMoeDsaForCausalLM"],
            ),
        ),
    )


def test_glm_moe_dsa_is_registered():
    assert "glm_moe_dsa" in model_definitions.mtp_model_classes
    assert model_definitions.resolve_model_type("glm_moe_dsa") == "glm_moe_dsa"

    components = model_definitions.mtp_model_classes["glm_moe_dsa"]
    assert components.first_layer_class is GlmMTPLayer
    assert components.decoder_layer_class is GlmMTPLayer
    assert components.norm_class is GlmMoeDsaRMSNorm
    assert components.rotary_emb_class is GlmMoeDsaRotaryEmbedding


def test_glm_moe_dsa_reports_minimum_transformers_version(monkeypatch):
    monkeypatch.delitem(model_definitions.mtp_model_classes, "glm_moe_dsa")
    with pytest.raises(ValueError, match="glm_moe_dsa.*5.14.1"):
        model_definitions.resolve_model_type("glm_moe_dsa")


def test_target_layer_ids_follow_num_hidden_layers():
    config = make_speculator_config(load_tiny_config())
    model = MTPDraftModel(config)
    assert model.target_layer_ids == [config.transformer_layer_config.num_hidden_layers]
    assert model.target_layer_ids == [2]


def test_builds_model_and_layer_from_registry():
    config = make_speculator_config(load_tiny_config())
    model = MTPDraftModel(config)
    assert isinstance(model.mtp_layers[0], GlmMTPLayer)
    assert isinstance(model.rotary_emb, GlmMoeDsaRotaryEmbedding)
    assert model.mtp_layers[0].self_attn.config.model_type == "glm_moe_dsa"


@pytest.mark.parametrize("fixture", ["tiny_config.json", "tiny_config_tied.json"])
def test_builds_tied_and_untied_models(fixture):
    transformer_config = load_tiny_config(fixture)
    model = MTPDraftModel(make_speculator_config(transformer_config))
    assert isinstance(model.mtp_layers[0], GlmMTPLayer)
    assert (
        model.config.transformer_layer_config.tie_word_embeddings
        is transformer_config.tie_word_embeddings
    )


def test_qwen_and_other_registrations_are_unchanged():
    assert "qwen3" in model_definitions.mtp_model_classes
    assert (
        model_definitions.mtp_model_classes["qwen3"].first_layer_class
        is not GlmMTPLayer
    )
    for model_type in ("qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"):
        if model_type in model_definitions.mtp_model_classes:
            assert (
                model_definitions.mtp_model_classes[model_type].first_layer_class
                is not GlmMTPLayer
            )

    assert {"mtp", "dflash", "dflash2", "dspark"} <= set(
        SpeculatorModelConfig.registry or {}
    )


def test_glm_capability_matches_probe_defaults():
    capability = MTP_CAPABILITIES["glm_moe_dsa"]
    assert capability.hidden_state_stage == "post_norm"
    assert capability.own_mtp_head is False
    assert capability.own_mtp_embed is False
    assert capability.own_mtp_final_norm is True
    assert capability.has_rot_weight is False
    assert capability.freeze_indexer is True
    assert capability.native_mtp_key_prefix == "model.layers.{num_hidden_layers}."
    assert capability.index_name == "quant_model_weights.safetensors.index.json"


def test_model_exposes_capability_and_keeps_generic_ignore_keys():
    model = MTPDraftModel(make_speculator_config(load_tiny_config()))
    assert model.mtp_capability is MTP_CAPABILITIES["glm_moe_dsa"]
    assert "lm_head.weight" in model._keys_to_ignore_on_save
    assert "embed_tokens.weight" in model._keys_to_ignore_on_save
    assert "verifier_norm.weight" in model._keys_to_ignore_on_save


def test_capability_ignore_keys_toggle_own_head_and_embed():
    capability = MTP_CAPABILITIES["glm_moe_dsa"]
    base = {"lm_head.weight", "embed_tokens.weight", "verifier_norm.weight"}

    assert capability_ignore_keys(base, capability) == base
    assert capability_ignore_keys(base, replace(capability, own_mtp_head=True)) == {
        "embed_tokens.weight",
        "verifier_norm.weight",
    }
    assert capability_ignore_keys(base, replace(capability, own_mtp_embed=True)) == {
        "lm_head.weight",
        "verifier_norm.weight",
    }
    assert capability_ignore_keys(
        base, replace(capability, own_mtp_head=True, own_mtp_embed=True)
    ) == {"verifier_norm.weight"}


def test_freeze_capability_parameters_refreezes_indexer():
    model = MTPDraftModel(make_speculator_config(load_tiny_config()))
    indexer = model.mtp_layers[0].self_attn.indexer
    assert indexer is not None
    for parameter in indexer.parameters():
        parameter.requires_grad = True

    model.freeze_capability_parameters()

    assert all(not parameter.requires_grad for parameter in indexer.parameters())


def test_shared_weight_loader_is_noop_for_default_glm(monkeypatch):
    calls = []
    capability = replace(
        MTP_CAPABILITIES["glm_moe_dsa"],
        shared_weight_loader=lambda model, path: calls.append((model, path)),
        shared_mtp_weights_path="/tmp/native",
    )
    monkeypatch.setitem(MTP_CAPABILITIES, "glm_moe_dsa", capability)
    model = MTPDraftModel(make_speculator_config(load_tiny_config()))

    model.load_shared_mtp_weights()

    assert calls == []


def test_shared_weight_loader_runs_when_own_weights_are_declared(monkeypatch):
    calls = []

    def loader(model, path):
        calls.append((model, path))

    capability = replace(
        MTP_CAPABILITIES["glm_moe_dsa"],
        own_mtp_head=True,
        own_mtp_embed=True,
        shared_weight_loader=loader,
        shared_mtp_weights_path="/tmp/native",
    )
    monkeypatch.setitem(MTP_CAPABILITIES, "glm_moe_dsa", capability)
    model = MTPDraftModel(make_speculator_config(load_tiny_config()))

    assert model._keys_to_ignore_on_save == {"verifier_norm.weight"}
    model.load_shared_mtp_weights()
    assert calls == [(model, "/tmp/native")]

    model.load_shared_mtp_weights("/tmp/other")
    assert calls[-1] == (model, "/tmp/other")


def test_shared_weight_loader_requires_loader_when_own_weights_declared(
    monkeypatch,
):
    capability = replace(MTP_CAPABILITIES["glm_moe_dsa"], own_mtp_head=True)
    monkeypatch.setitem(MTP_CAPABILITIES, "glm_moe_dsa", capability)
    model = MTPDraftModel(make_speculator_config(load_tiny_config()))

    with pytest.raises(NotImplementedError, match="no shared-weight loader"):
        model.load_shared_mtp_weights("/tmp/native")


def test_vllm_train_forward_is_rejected_for_post_norm_capability():
    model = MTPDraftModel(make_speculator_config(load_tiny_config()))
    input_ids = torch.zeros(1, 16, dtype=torch.long)
    hidden_states = torch.randn(1, 16, model.config.hidden_size)

    with pytest.raises(ValueError, match="double-normalize"):
        model(input_ids, hidden_states, train_forward="vllm")


def test_fastmtp_forward_runs_for_glm():
    torch.manual_seed(0)
    model = MTPDraftModel(make_speculator_config(load_tiny_config()))
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    # The upstream experts store uninitialized ``torch.empty`` weights and
    # ``MTPDraftModel._init_weights`` does not know the GLM expert class.
    nn.init.normal_(model.mtp_layers[0].mlp.experts.gate_up_proj, std=0.02)
    nn.init.normal_(model.mtp_layers[0].mlp.experts.down_proj, std=0.02)
    model.eval()

    input_ids = torch.randint(0, 128, (1, 16))
    hidden_states = torch.randn(1, 16, model.config.hidden_size)

    draft_tokens, loss, metrics = model(input_ids, hidden_states)

    assert len(draft_tokens) == 1
    assert draft_tokens[0].shape == (14,)  # seq_len - K - 1 supervised positions
    assert torch.isfinite(loss).all()
    assert "accept_pos_1_sum" in metrics
    assert "step_1_acc_sum" in metrics
    assert "loss_sum" in metrics
