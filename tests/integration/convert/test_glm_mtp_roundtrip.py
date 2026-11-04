"""End-to-end GLM MTP convert -> train-shape -> stitch round trip (CPU)."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from speculators.cli.stitch import stitch
from speculators.convert.mtp.converter import MTPConverter
from speculators.convert.mtp.glm_layout import GlmMtpPolicy
from speculators.models.mtp import MTPDraftModel
from tests.unit.convert.glm_mtp_fixtures import (
    MTP_PREFIX,
    build_native_mtp,
    build_non_mtp_weights,
    write_checkpoint,
)

QUANT_INDEX = "quant_model_weights.safetensors.index.json"


def _convert(tmp_path: Path, native: dict[str, torch.Tensor], **converter_kwargs):
    native_dir = write_checkpoint(
        tmp_path / "native",
        {**native, **build_non_mtp_weights()},
        shards=3,
        index_name=QUANT_INDEX,
    )
    converted = tmp_path / "converted"
    MTPConverter(**converter_kwargs).convert(
        native_dir, converted, base_model=str(native_dir), validate=True
    )
    return native_dir, converted


def test_bf16_quant_index_roundtrip_is_bitwise(tmp_path):
    native = build_native_mtp(dtype=torch.bfloat16)
    native_dir, converted = _convert(tmp_path, native)

    manifest = json.loads((converted / "glm_native_manifest.json").read_text())
    key_map = json.loads((converted / "glm_mtp_key_map.json").read_text())
    assert manifest["index_name"] == QUANT_INDEX
    assert manifest["hidden_state_stage"] == "post_norm"
    # ``is_rot_used`` is null until the customer probe provides quant
    # metadata; the converter still succeeds for a no-rot source.
    assert manifest["is_rot_used"] is None
    assert manifest["rot_gate_consistent"] is None
    assert manifest["native_mtp_layer_index"] == 2
    assert manifest["native_mtp_key_prefix"] == MTP_PREFIX
    assert key_map["native_mtp_key_prefix"] == MTP_PREFIX

    model = MTPDraftModel.from_pretrained(str(converted))
    model.save_pretrained(str(tmp_path / "finetuned"))
    stitched_dir = tmp_path / "stitched"
    stitch(tmp_path / "finetuned", native_dir, stitched_dir)

    weight_map = json.loads((stitched_dir / QUANT_INDEX).read_text())["weight_map"]
    stitched: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        stitched.update(load_file(str(stitched_dir / shard)))

    for key, tensor in native.items():
        # own_mtp_head/embed=false tensors are copied through untouched; every
        # other MTP tensor must round-trip bitwise in bf16.
        assert torch.equal(stitched[key], tensor), key

    for key, tensor in build_non_mtp_weights().items():
        assert torch.equal(stitched[key], tensor), key
    assert not any(key.startswith("mtp_layers.") for key in stitched)


def test_zero_optional_norm_bias_is_dropped_and_nonzero_is_rejected(tmp_path):
    native = build_native_mtp()
    native[MTP_PREFIX + "enorm.bias"] = torch.zeros(64)
    _, converted = _convert(tmp_path, native)

    manifest = json.loads((converted / "glm_native_manifest.json").read_text())
    assert MTP_PREFIX + "enorm.bias" in manifest["source_keys_unused"]
    assert not any(
        "token_layernorm.bias" in key
        for key in load_file(str(converted / "model.safetensors"))
    )

    bad = build_native_mtp()
    bad[MTP_PREFIX + "enorm.bias"] = torch.ones(64)
    native_dir = write_checkpoint(
        tmp_path / "bad-native", {**bad, **build_non_mtp_weights()}
    )
    with pytest.raises(ValueError, match="Non-zero optional RMSNorm bias"):
        MTPConverter(policy=GlmMtpPolicy()).convert(
            native_dir, tmp_path / "bad-converted", base_model=str(native_dir)
        )


def test_rot_gate_consistency_blocks_conversion(tmp_path):
    native = build_native_mtp()
    native_dir = write_checkpoint(
        tmp_path / "native", {**native, **build_non_mtp_weights()}
    )

    with pytest.raises(ValueError, match="rot_gate_consistent=false"):
        MTPConverter(policy=GlmMtpPolicy(is_rot_used=True)).convert(
            native_dir, tmp_path / "converted", base_model=str(native_dir)
        )


def test_native_mtp_layer_index_must_match_depth(tmp_path):
    native = build_native_mtp()
    native_dir = write_checkpoint(
        tmp_path / "native", {**native, **build_non_mtp_weights()}
    )
    config = json.loads((native_dir / "config.json").read_text())
    config["num_hidden_layers"] = 3
    (native_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="num_hidden_layers"):
        MTPConverter(policy=GlmMtpPolicy()).convert(
            native_dir, tmp_path / "converted", base_model=str(native_dir)
        )


def test_manifest_files_are_written_next_to_checkpoint(tmp_path):
    native = build_native_mtp()
    _, converted = _convert(tmp_path, native)

    assert (converted / "glm_native_manifest.json").is_file()
    assert (converted / "glm_mtp_key_map.json").is_file()
    assert (converted / "glm_native_manifest.json").stat().st_size > 0


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"


def test_frozen_native_fixture_matches_stacked_reference(tmp_path, monkeypatch):
    native_dir = FIXTURES / "native_per_expert"
    converted = tmp_path / "converted"
    MTPConverter(policy=GlmMtpPolicy()).convert(
        native_dir, converted, base_model=str(native_dir), validate=True
    )

    # The frozen stacked fixture references its verifier with a repo-relative
    # path; from_pretrained resolves it against the working directory.
    monkeypatch.chdir(FIXTURES.parents[2])
    expected = MTPDraftModel.from_pretrained(str(FIXTURES / "native_stacked"))
    actual = MTPDraftModel.from_pretrained(str(converted))
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    assert set(actual_state) == set(expected_state)
    for key, tensor in expected_state.items():
        assert torch.equal(actual_state[key], tensor), key

    generated_manifest = json.loads(
        (converted / "glm_native_manifest.json").read_text()
    )
    frozen_manifest = json.loads((FIXTURES / "glm_native_manifest.json").read_text())
    assert generated_manifest == frozen_manifest
    assert json.loads((converted / "glm_mtp_key_map.json").read_text()) == (
        json.loads((FIXTURES / "glm_mtp_key_map.json").read_text())
    )
