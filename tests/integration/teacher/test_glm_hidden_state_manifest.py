"""CPU gate for the frozen teacher capture manifest (plan sections 10/21.6)."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers.models.glm_moe_dsa")

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)

from speculators.models.mtp.capture import (
    CaptureManifest,
    compare_capture_files,
    compare_capture_tensors,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"


def tiny_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig.from_pretrained(FIXTURE_DIR / "tiny_config.json")


def make_manifest(**overrides) -> dict:
    manifest = {
        "hidden_state_stage": "post_norm",
        "verifier_source_layer_ordinal": 2,
        "verifier_source_layer_index_zero_based": 1,
        "verifier_source_tensor_semantics": "decoder_output_before_model_norm",
        "verifier_source_indexer_type": "shared",
        "capture_tensor_semantics": "model_norm_output",
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
    manifest.update(overrides)
    return manifest


def pre_norm_manifest(**overrides) -> dict:
    manifest = make_manifest(
        hidden_state_stage="pre_norm",
        capture_tensor_semantics="decoder_output_before_model_norm",
        model_norm={"weight_sha256": "d" * 64},
    )
    manifest.update(overrides)
    return manifest


def test_valid_post_norm_manifest_matches_config_and_keys():
    manifest = CaptureManifest(make_manifest())

    manifest.validate(
        config=tiny_config(),
        checkpoint_keys=["model.layers.2.eh_proj.weight", "model.embed_tokens.weight"],
    )

    assert manifest.hidden_state_stage == "post_norm"
    assert not manifest.uses_verifier_norm


def test_valid_pre_norm_manifest_requires_frozen_norm_hash():
    manifest = CaptureManifest(pre_norm_manifest())

    manifest.validate(config=tiny_config())

    assert manifest.uses_verifier_norm
    with pytest.raises(ValueError, match="weight_sha256"):
        CaptureManifest(pre_norm_manifest(model_norm={"eps": 1e-5}))
    # A pre_norm capture without any model_norm record cannot prove that the
    # frozen norm was persisted; it must be rejected too.
    no_norm = pre_norm_manifest()
    del no_norm["model_norm"]
    with pytest.raises(ValueError, match="weight_sha256"):
        CaptureManifest(no_norm)
    with pytest.raises(ValueError, match="source_norm"):
        CaptureManifest(make_manifest(source_norm="model.norm.final"))
    with pytest.raises(ValueError, match="post_norm captures must record"):
        CaptureManifest(make_manifest(model_norm=None))
    # The post-norm parity gate must be able to hash model.norm.weight, so a
    # capture that only records the epsilon is not enough either.
    with pytest.raises(ValueError, match="weight_sha256"):
        CaptureManifest(make_manifest(model_norm={"eps": 1e-5}))


def test_layer_naming_must_not_conflate_ordinal_and_index():
    with pytest.raises(ValueError, match="zero_based"):
        CaptureManifest(make_manifest(verifier_source_layer_index_zero_based=2))
    with pytest.raises(ValueError, match="native_mtp_key_prefix"):
        CaptureManifest(make_manifest(native_mtp_key_prefix="model.layers.3."))
    with pytest.raises(ValueError, match="native_mtp_tensor_semantics"):
        CaptureManifest(make_manifest(native_mtp_tensor_semantics="mtp"))


def test_required_fields_are_enforced():
    manifest = make_manifest()
    del manifest["hidden_state_stage"]
    with pytest.raises(ValueError, match="missing required fields"):
        CaptureManifest(manifest)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "verifier_source_layer_ordinal": 3,
                "verifier_source_layer_index_zero_based": 2,
                "native_mtp_layer_index": 3,
                "native_mtp_key_prefix": "model.layers.3.",
            },
            "does not match the verifier depth",
        ),
        (
            {
                "native_mtp_layer_index": 3,
                "native_mtp_key_prefix": "model.layers.3.",
            },
            "native MTP layer index",
        ),
        ({"hidden_size": 32}, "hidden_size"),
        ({"model_norm": {"eps": 1e-6, "weight_sha256": "d" * 64}}, "model_norm.eps"),
        ({"verifier_source_indexer_type": "full"}, "indexer type"),
    ],
)
def test_config_cross_checks(overrides, message):
    manifest = CaptureManifest(make_manifest(**overrides))
    with pytest.raises(ValueError, match=message):
        manifest.validate(config=tiny_config())


def test_checkpoint_prefix_must_exist():
    manifest = CaptureManifest(make_manifest())
    with pytest.raises(ValueError, match="No checkpoint key starts with"):
        manifest.validate(checkpoint_keys=["model.embed_tokens.weight"])


def test_capture_files_can_be_verified_on_disk(tmp_path):
    capture = tmp_path / "capture.safetensors"
    save_file({"hidden_states": torch.zeros(2, 4, 64)}, str(capture))
    from speculators.models.mtp.capture import sha256_file

    digest = sha256_file(capture)
    manifest = CaptureManifest(
        make_manifest(capture_files=[{"path": str(capture), "sha256": digest}])
    )
    manifest.validate(verify_files=True)

    bad = CaptureManifest(
        make_manifest(capture_files=[{"path": str(capture), "sha256": "0" * 64}])
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        bad.validate(verify_files=True)


def test_relative_capture_paths_resolve_against_manifest_dir(tmp_path, monkeypatch):
    capture = _write_capture(tmp_path / "capture.safetensors")
    from speculators.models.mtp.capture import sha256_file

    manifest_path = tmp_path / "capture_manifest.json"
    entry = {"path": "capture.safetensors", "sha256": sha256_file(capture)}
    manifest_path.write_text(json.dumps(make_manifest(capture_files=[entry])))
    # Relative entries must resolve next to the manifest, not the CWD.
    monkeypatch.chdir(tmp_path.parent)

    manifest = CaptureManifest.from_path(manifest_path)
    manifest.validate(verify_files=True)
    assert manifest.base_dir == tmp_path.resolve()


def test_training_hidden_states_applies_norm_exactly_once():
    calls = []

    def verifier_norm(tensor):
        calls.append(tensor)
        return tensor * 2

    hidden = torch.ones(1, 4, 64)

    post = CaptureManifest(make_manifest())
    assert torch.equal(post.training_hidden_states(hidden, verifier_norm), hidden)
    assert calls == []

    pre = CaptureManifest(pre_norm_manifest())
    assert torch.equal(pre.training_hidden_states(hidden, verifier_norm), hidden * 2)
    assert len(calls) == 1


def _write_capture(path: Path, scale: float = 1.0) -> Path:
    save_file(
        {
            "hidden_states": torch.linspace(0.5, 1.5, 64).reshape(1, 1, 64) * scale,
            "input_ids": torch.arange(4).reshape(1, 4),
        },
        str(path),
    )
    return path


def test_determinism_gate_bitwise_then_tolerance_then_fail(tmp_path):
    first = _write_capture(tmp_path / "a.safetensors")
    twin = tmp_path / "twin" / "a.safetensors"
    twin.parent.mkdir()
    _write_capture(twin)

    bitwise = compare_capture_files([first], [twin])
    assert bitwise["determinism"] == "bitwise"
    assert bitwise["bitwise"] is True

    noisy_dir = tmp_path / "noisy"
    noisy_dir.mkdir()
    noisy = _write_capture(noisy_dir / "a.safetensors", scale=1.00001)
    tolerance = compare_capture_files([first], [noisy])
    assert tolerance["determinism"] == "tolerance"
    assert tolerance["bitwise"] is False
    assert tolerance["within_tolerance"] is True
    assert tolerance["mismatch_fraction"] >= 0.0

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    broken = _write_capture(broken_dir / "a.safetensors", scale=2.0)
    failed = compare_capture_files([first], [broken])
    assert failed["determinism"] == "fail"
    assert failed["within_tolerance"] is False


def test_compare_tensors_reports_all_gate_metrics():
    first = {"hidden_states": torch.ones(1, 4, 8)}
    second = {"hidden_states": torch.ones(1, 4, 8)}

    result = compare_capture_tensors(first, second)

    for key in (
        "determinism",
        "bitwise",
        "within_tolerance",
        "tensor_count",
        "numel",
        "mismatch_count",
        "mismatch_fraction",
        "max_abs",
        "mean_abs",
        "max_rel",
        "atol",
        "rtol",
    ):
        assert key in result

    with pytest.raises(ValueError, match="tensor sets differ"):
        compare_capture_tensors(first, {})
