"""CPU integration forward for the GLM MTP draft model (K=1 main path)."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers.models.glm_moe_dsa")

from torch import nn
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig, shift_batch_mtp
from speculators.proposals import GreedyTokenProposalConfig

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"
SEQ_LEN = 32


def load_tiny_config() -> GlmMoeDsaConfig:
    config = GlmMoeDsaConfig.from_pretrained(FIXTURE_DIR / "tiny_config.json")
    config._attn_implementation = "sdpa"
    return config


def make_model() -> MTPDraftModel:
    torch.manual_seed(0)
    transformer_config = load_tiny_config()
    config = MTPSpeculatorConfig(
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
    model = MTPDraftModel(config)
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    # Upstream expert params are ``torch.empty``; the generic init hook does
    # not know the GLM expert class.
    nn.init.normal_(model.mtp_layers[0].mlp.experts.gate_up_proj, std=0.02)
    nn.init.normal_(model.mtp_layers[0].mlp.experts.down_proj, std=0.02)
    model.eval()
    return model


def make_batch(model: MTPDraftModel):
    generator = torch.Generator().manual_seed(1)
    return {
        "input_ids": torch.randint(0, 128, (1, SEQ_LEN), generator=generator),
        "verifier_last_hidden_states": torch.randn(
            1, SEQ_LEN, model.config.hidden_size, generator=generator
        ),
        "loss_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
        "position_ids": torch.arange(SEQ_LEN).unsqueeze(0),
        "document_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
    }


def test_no_pack_forward_and_metrics():
    model = make_model()
    batch = shift_batch_mtp(make_batch(model))

    draft_tokens, loss, metrics = model(**batch, train_forward="fastmtp")

    assert draft_tokens[0].shape == (SEQ_LEN - 2,)
    assert torch.isfinite(loss)
    for key in (
        "loss_step_0",
        "step_1_acc_sum",
        "step_1_acc_total",
        "accept_pos_1_sum",
        "accept_pos_1_total",
        "loss_sum",
        "loss_total",
    ):
        assert key in metrics


def test_pre_norm_capture_is_normalized_exactly_once():
    model = make_model()
    batch = make_batch(model)
    hidden_states = batch["verifier_last_hidden_states"]

    with torch.no_grad():
        post_norm = model.verifier_norm(hidden_states)
        pre_tokens, pre_loss, _ = model(
            batch["input_ids"],
            hidden_states,
            position_ids=batch["position_ids"],
            loss_mask=batch["loss_mask"],
            hidden_state_stage="pre_norm",
        )
        post_tokens, post_loss, _ = model(
            batch["input_ids"],
            post_norm,
            position_ids=batch["position_ids"],
            loss_mask=batch["loss_mask"],
            hidden_state_stage="post_norm",
        )

    assert torch.equal(post_norm, model.verifier_norm(hidden_states))
    torch.testing.assert_close(pre_loss, post_loss)
    assert torch.equal(pre_tokens[0], post_tokens[0])


def test_post_norm_capture_never_calls_verifier_norm(monkeypatch):
    model = make_model()
    batch = make_batch(model)

    def fail(_tensor):
        raise AssertionError("post_norm captures must not be re-normalized")

    monkeypatch.setattr(model.verifier_norm, "forward", fail)

    _, loss, _ = model(
        batch["input_ids"],
        batch["verifier_last_hidden_states"],
        position_ids=batch["position_ids"],
        loss_mask=batch["loss_mask"],
        hidden_state_stage="post_norm",
    )
    assert torch.isfinite(loss)


def test_vllm_rejects_post_norm_capture():
    model = make_model()
    batch = make_batch(model)

    with pytest.raises(ValueError, match="double-normalize"):
        model(
            batch["input_ids"],
            batch["verifier_last_hidden_states"],
            position_ids=batch["position_ids"],
            hidden_state_stage="post_norm",
            train_forward="vllm",
        )


def test_vllm_guard_is_absolute_for_post_norm_capability():
    model = make_model()
    batch = make_batch(model)

    # §17.9: the vLLM-aligned path is a Phase 8 experiment for GLM; an
    # explicit pre_norm stage must not reopen it.
    with pytest.raises(ValueError, match="double-normalize"):
        model(
            batch["input_ids"],
            batch["verifier_last_hidden_states"],
            position_ids=batch["position_ids"],
            hidden_state_stage="pre_norm",
            train_forward="vllm",
        )


def test_unknown_stage_is_rejected():
    model = make_model()
    batch = make_batch(model)

    with pytest.raises(ValueError, match="hidden_state_stage"):
        model(
            batch["input_ids"],
            batch["verifier_last_hidden_states"],
            hidden_state_stage="final_norm",
        )


def test_backward_is_finite():
    model = make_model()
    model.train()
    batch = shift_batch_mtp(make_batch(model))

    _, loss, _ = model(**batch, train_forward="fastmtp")
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
