"""pre-norm -> model.norm -> post-norm parity on CPU (plan section 10.3)."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers.models.glm_moe_dsa")

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRMSNorm,
)

from speculators.models.mtp.capture import (
    CaptureManifest,
    compare_capture_tensors,
    reference_post_norm,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"


def tiny_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig.from_pretrained(FIXTURE_DIR / "tiny_config.json")


def manifest(stage: str) -> CaptureManifest:
    data = {
        "hidden_state_stage": stage,
        "verifier_source_layer_ordinal": 2,
        "verifier_source_layer_index_zero_based": 1,
        "verifier_source_tensor_semantics": "decoder_output_before_model_norm",
        "verifier_source_indexer_type": "shared",
        "capture_tensor_semantics": (
            "model_norm_output"
            if stage == "post_norm"
            else "decoder_output_before_model_norm"
        ),
        "vllm_target_layer_id": 2,
        "native_mtp_layer_index": 2,
        "native_mtp_key_prefix": "model.layers.2.",
        "native_mtp_tensor_semantics": "native_mtp_head",
        "source_norm": "model.norm",
        "hidden_size": 64,
        "dtype": "bfloat16",
        "input_bytes_sha256": "a" * 64,
        "position_ids_sha256": "b" * 64,
        "capture_files": [{"path": "capture.safetensors", "sha256": "c" * 64}],
        "model_norm": {"eps": 1e-5, "weight_sha256": "d" * 64},
    }
    return CaptureManifest(data)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_post_norm_matches_upstream_norm(dtype):
    torch.manual_seed(0)
    config = tiny_config()
    norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(config.hidden_size, dtype=dtype) * 0.1 + 1.0)

    pre_norm = torch.randn(2, 16, config.hidden_size, dtype=dtype)
    expected = norm(pre_norm)
    actual = reference_post_norm(pre_norm, norm.weight, config.rms_norm_eps)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_post_norm_capture_parity_bypasses_norm():
    torch.manual_seed(0)
    config = tiny_config()
    norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    pre_norm = torch.randn(1, 8, config.hidden_size)
    reference = reference_post_norm(pre_norm, norm.weight, config.rms_norm_eps)

    metrics = compare_capture_tensors(
        {"hidden_states": reference}, {"hidden_states": reference.clone()}
    )
    assert metrics["determinism"] == "bitwise"
    assert metrics["max_abs"] == 0.0

    post = manifest("post_norm")
    assert torch.equal(post.training_hidden_states(reference, norm), reference)


def test_pre_norm_capture_is_normalized_once_and_matches_post_capture():
    torch.manual_seed(0)
    config = tiny_config()
    norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    pre_norm = torch.randn(1, 8, config.hidden_size)
    post_norm = norm(pre_norm)

    pre = manifest("pre_norm")
    trained = pre.training_hidden_states(pre_norm, norm)

    assert torch.equal(trained, post_norm)
    metrics = compare_capture_tensors(
        {"hidden_states": trained}, {"hidden_states": post_norm}
    )
    assert metrics["determinism"] == "bitwise"


def test_tolerance_gate_reports_pre_registered_metrics():
    torch.manual_seed(0)
    weight = torch.ones(64)
    pre_norm = torch.randn(1, 8, 64)
    reference = reference_post_norm(pre_norm, weight, 1e-5)
    noisy = reference + 1e-4

    metrics = compare_capture_tensors(
        {"hidden_states": noisy}, {"hidden_states": reference}, atol=1e-3, rtol=1e-2
    )
    assert metrics["determinism"] == "tolerance"
    assert metrics["max_abs"] < 1e-3
    assert metrics["mismatch_fraction"] == 0.0
