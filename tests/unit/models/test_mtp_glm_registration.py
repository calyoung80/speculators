"""Registration tests for the GLM-MoE-DSA MTP layer."""

import pytest

pytest.importorskip("transformers.models.glm_moe_dsa")

from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaRMSNorm,
    GlmMoeDsaRotaryEmbedding,
)

from speculators.models.mtp import model_definitions  # noqa: E402
from speculators.models.mtp.glm_glue import GlmMTPLayer  # noqa: E402


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
