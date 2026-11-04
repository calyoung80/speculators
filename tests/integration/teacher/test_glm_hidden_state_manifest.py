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
    build_capture_manifest,
    compare_capture_files,
    compare_capture_tensors,
    compare_post_norm_files,
    compare_post_norm_tensors,
    load_capture_norm,
    reference_post_norm,
    run_post_norm_gate,
    sha256_file,
    sha256_tensor,
    write_capture_manifest,
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


def _build_manifest(tmp_path: Path, **overrides) -> dict:
    capture = _write_capture(tmp_path / "capture.safetensors")
    kwargs = {
        "hidden_state_stage": "post_norm",
        "verifier_source_layer_ordinal": 2,
        "native_mtp_layer_index": 2,
        "native_mtp_key_prefix": "model.layers.2.",
        "vllm_target_layer_id": 2,
        "hidden_size": 64,
        "dtype": torch.bfloat16,
        "input_bytes_sha256": "a" * 64,
        "position_ids_sha256": "b" * 64,
        "capture_files": [capture],
        "model_norm_weight": torch.ones(64),
        "verifier_source_indexer_type": "shared",
        "model_norm_eps": 1e-5,
    }
    kwargs.update(overrides)
    return build_capture_manifest(**kwargs)


def test_build_capture_manifest_derives_stage_fields(tmp_path):
    data = _build_manifest(tmp_path)

    assert data["verifier_source_layer_index_zero_based"] == 1
    assert data["verifier_source_tensor_semantics"] == (
        "decoder_output_before_model_norm"
    )
    assert data["capture_tensor_semantics"] == "model_norm_output"
    assert data["native_mtp_tensor_semantics"] == "native_mtp_head"
    assert data["dtype"] == "bfloat16"
    assert data["capture_files"][0]["sha256"] == sha256_file(
        tmp_path / "capture.safetensors"
    )
    CaptureManifest(data).validate(config=tiny_config())


def test_build_capture_manifest_pre_norm_omits_eps(tmp_path):
    data = _build_manifest(
        tmp_path,
        hidden_state_stage="pre_norm",
        model_norm_eps=None,
    )

    assert data["capture_tensor_semantics"] == ("decoder_output_before_model_norm")
    assert "eps" not in data["model_norm"]
    assert data["model_norm"]["weight_sha256"] == sha256_tensor(torch.ones(64))


def test_build_capture_manifest_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="model_norm_eps"):
        _build_manifest(tmp_path, model_norm_eps=None)
    with pytest.raises(ValueError, match="extra fields"):
        _build_manifest(tmp_path, extra={"hidden_size": 32})
    with pytest.raises(ValueError, match="capture_files"):
        _build_manifest(tmp_path, capture_files=[])


def test_write_capture_manifest_verifies_files(tmp_path):
    data = _build_manifest(tmp_path)
    manifest_path = tmp_path / "capture_manifest.json"

    validated = write_capture_manifest(manifest_path, data)

    assert validated.base_dir == tmp_path.resolve()
    assert CaptureManifest.from_path(manifest_path).hidden_state_stage == "post_norm"

    (tmp_path / "capture.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_capture_manifest(manifest_path, data)


def test_post_norm_reference_gate_passes_and_fails():
    weight = torch.randn(64)
    hidden = torch.randn(2, 4, 64, dtype=torch.float32)
    reference = reference_post_norm(hidden, weight, 1e-5)

    passed = compare_post_norm_tensors(
        {"hidden_states": hidden},
        {"hidden_states": reference},
        weight,
        1e-5,
    )
    assert passed["gate"] == "pass"
    assert passed["determinism"] == "bitwise"
    assert passed["token_count"] == 8
    assert passed["weight_sha256"] == sha256_tensor(weight)

    failed = compare_post_norm_tensors(
        {"hidden_states": hidden},
        {"hidden_states": reference * 2},
        weight,
        1e-5,
    )
    assert failed["gate"] == "fail"
    assert failed["within_tolerance"] is False


def test_post_norm_reference_gate_rejects_misaligned_sets():
    weight = torch.randn(64)
    hidden = torch.randn(1, 4, 64)

    with pytest.raises(ValueError, match="tensor sets differ"):
        compare_post_norm_tensors(
            {"hidden_states": hidden},
            {"other": hidden},
            weight,
            1e-5,
        )


def test_post_norm_file_gate_records_hashes(tmp_path):
    weight = torch.randn(64)
    hidden = torch.randn(2, 4, 64)
    pre = tmp_path / "pre.safetensors"
    post = tmp_path / "post.safetensors"
    save_file({"hidden_states": hidden}, str(pre))
    save_file({"hidden_states": reference_post_norm(hidden, weight, 1e-5)}, str(post))

    result = compare_post_norm_files([pre], [post], weight, 1e-5)

    assert result["gate"] == "pass"
    assert result["files"]["pre_norm"][0]["sha256"] == sha256_file(pre)
    assert result["files"]["post_norm"][0]["sha256"] == sha256_file(post)

    duplicate = tmp_path / "duplicate.safetensors"
    save_file({"hidden_states": hidden}, str(duplicate))
    with pytest.raises(ValueError, match="repeat"):
        compare_post_norm_files([pre, duplicate], [post], weight, 1e-5)


def _hidden_and_tokens(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, 4, 64, generator=generator)
    tokens = torch.arange(8, dtype=torch.long).reshape(2, 4)
    return {
        "hidden_states": hidden,
        "input_ids": tokens,
        "token_ids": tokens.clone(),
        "position_ids": torch.arange(4).unsqueeze(0).expand(2, 4).clone(),
    }


def test_post_norm_gate_ignores_token_tensors():
    weight = torch.randn(64)
    pre_norm = _hidden_and_tokens()
    post_norm = {
        **pre_norm,
        "hidden_states": reference_post_norm(pre_norm["hidden_states"], weight, 1e-5),
    }

    result = compare_post_norm_tensors(pre_norm, post_norm, weight, 1e-5)

    assert result["gate"] == "pass"
    assert result["token_alignment"]["aligned"] is True
    assert result["token_count"] == 8
    assert result["compared_keys"] == ["hidden_states"]
    assert "input_ids" in result["token_alignment"]["keys"]


def test_post_norm_gate_rejects_token_misalignment():
    weight = torch.randn(64)
    pre_norm = _hidden_and_tokens()
    post_norm = {
        **pre_norm,
        "hidden_states": reference_post_norm(pre_norm["hidden_states"], weight, 1e-5),
        "token_ids": torch.zeros(8, dtype=torch.long).reshape(2, 4),
    }

    result = compare_post_norm_tensors(pre_norm, post_norm, weight, 1e-5)

    assert result["gate"] == "fail"
    assert result["token_alignment"]["aligned"] is False
    assert result["token_alignment"]["mismatched_keys"] == ["token_ids"]


def test_post_norm_gate_checks_expected_weight_hash():
    weight = torch.randn(64)
    pre_norm = _hidden_and_tokens()
    post_norm = {
        **pre_norm,
        "hidden_states": reference_post_norm(pre_norm["hidden_states"], weight, 1e-5),
    }

    result = compare_post_norm_tensors(
        pre_norm,
        post_norm,
        weight,
        1e-5,
        expected_weight_sha256="0" * 64,
    )

    assert result["gate"] == "fail"
    assert result["weight_hash_match"] is False


def test_post_norm_gate_rejects_fp8_capture():
    tensors = {
        "hidden_states": torch.zeros(2, 4, 64, dtype=torch.float8_e4m3fn),
        "hidden_states_scales": torch.ones(2, 1, 1),
    }
    with pytest.raises(NotImplementedError, match="FP8"):
        compare_post_norm_tensors(tensors, tensors, torch.ones(64), 1e-5)


def test_post_norm_gate_accepts_single_file_paths(tmp_path):
    weight = torch.randn(64)
    pre_norm = _hidden_and_tokens()
    pre = tmp_path / "pre.safetensors"
    post = tmp_path / "post.safetensors"
    save_file(pre_norm, str(pre))
    save_file(
        {
            **pre_norm,
            "hidden_states": reference_post_norm(
                pre_norm["hidden_states"], weight, 1e-5
            ),
        },
        str(post),
    )

    result = compare_post_norm_files(pre, post, weight, 1e-5)

    assert result["gate"] == "pass"


def test_chunked_capture_requires_semantics(tmp_path):
    with pytest.raises(ValueError, match="chunk_semantics"):
        CaptureManifest(_build_manifest(tmp_path, chunked_prefill=True))

    manifest = CaptureManifest(
        _build_manifest(
            tmp_path,
            chunked_prefill=True,
            chunk_semantics={"stitch": "kv-page-order", "padding": "none"},
        )
    )
    assert manifest.data["chunked_prefill"] is True


def test_manifest_requires_hex_hashes_and_indexer(tmp_path):
    with pytest.raises(ValueError, match="input_bytes_sha256"):
        CaptureManifest(make_manifest(input_bytes_sha256="not-a-hash"))
    # A trailing newline must not sneak past fullmatch ($ anchors before \n).
    with pytest.raises(ValueError, match="input_bytes_sha256"):
        CaptureManifest(make_manifest(input_bytes_sha256="a" * 64 + "\n"))
    # Uppercase is rejected: verify_files compares against the lowercase
    # digest sha256_file emits, so accepting it would be self-contradictory.
    with pytest.raises(ValueError, match="model_norm.weight_sha256"):
        CaptureManifest(
            make_manifest(model_norm={"eps": 1e-5, "weight_sha256": "D" * 64})
        )
    # A float zero-based index would pass the equality check and only blow up
    # later as a confusing sequence index.
    with pytest.raises(ValueError, match="verifier_source_layer_index_zero_based"):
        CaptureManifest(make_manifest(verifier_source_layer_index_zero_based=1.0))

    missing_indexer = make_manifest()
    del missing_indexer["verifier_source_indexer_type"]
    with pytest.raises(ValueError, match="verifier_source_indexer_type"):
        CaptureManifest(missing_indexer)
    with pytest.raises(ValueError, match="verifier_source_indexer_type"):
        CaptureManifest(make_manifest(verifier_source_indexer_type=None))

    with pytest.raises(ValueError, match="vllm_target_layer_id"):
        CaptureManifest(make_manifest(vllm_target_layer_id="2"))


def test_load_capture_norm_rejects_nonzero_bias(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_capture_norm(tmp_path / "missing-dir")

    weight_file = tmp_path / "norm.safetensors"
    save_file({"model.norm.weight": torch.ones(64)}, str(weight_file))
    assert torch.equal(load_capture_norm(weight_file), torch.ones(64))

    bias_file = tmp_path / "norm_bias.safetensors"
    save_file(
        {
            "model.norm.weight": torch.ones(64),
            "model.norm.bias": torch.ones(64),
        },
        str(bias_file),
    )
    with pytest.raises(ValueError, match="non-zero"):
        load_capture_norm(bias_file)


def test_run_post_norm_gate_end_to_end(tmp_path):
    weight = torch.randn(64)
    pre_norm = _hidden_and_tokens(seed=1)
    post_norm = {
        **pre_norm,
        "hidden_states": reference_post_norm(pre_norm["hidden_states"], weight, 1e-5),
    }
    pre_file = tmp_path / "pre.safetensors"
    post_file = tmp_path / "post.safetensors"
    weight_file = tmp_path / "norm.safetensors"
    save_file(pre_norm, str(pre_file))
    save_file(post_norm, str(post_file))
    save_file({"model.norm.weight": weight}, str(weight_file))

    def manifest(stage: str, capture: Path, **overrides) -> dict:
        kwargs = {
            "hidden_state_stage": stage,
            "verifier_source_layer_ordinal": 2,
            "native_mtp_layer_index": 2,
            "native_mtp_key_prefix": "model.layers.2.",
            "vllm_target_layer_id": 2,
            "hidden_size": 64,
            "dtype": torch.bfloat16,
            "input_bytes_sha256": "a" * 64,
            "position_ids_sha256": "b" * 64,
            "capture_files": [capture],
            "model_norm_weight": weight,
            "verifier_source_indexer_type": "shared",
            "model_norm_eps": 1e-5 if stage == "post_norm" else None,
        }
        kwargs.update(overrides)
        # Default provenance is shared by both captures; overrides may patch
        # single fields (e.g. a mismatched vllm_commit).
        kwargs["extra"] = {
            "container_image": "glm-mtp:latest",
            "vllm_commit": "abc123",
            **kwargs.get("extra", {}),
        }
        return build_capture_manifest(**kwargs)

    pre_manifest = write_capture_manifest(
        tmp_path / "pre_manifest.json", manifest("pre_norm", pre_file)
    )
    post_manifest = write_capture_manifest(
        tmp_path / "post_manifest.json", manifest("post_norm", post_file)
    )
    report_path = tmp_path / "report.json"

    result = run_post_norm_gate(
        pre_manifest,
        post_manifest,
        norm_source=weight_file,
        report_path=report_path,
    )

    assert result["gate"] == "pass"
    assert result["request"]["input_bytes_sha256"] == "a" * 64
    assert json.loads(report_path.read_text())["gate"] == "pass"

    mismatched = write_capture_manifest(
        tmp_path / "post_mismatch.json",
        manifest("post_norm", post_file, input_bytes_sha256="c" * 64),
    )
    with pytest.raises(ValueError, match="same request"):
        run_post_norm_gate(pre_manifest, mismatched, norm_source=weight_file)

    other_commit = write_capture_manifest(
        tmp_path / "post_commit.json",
        manifest("post_norm", post_file, extra={"vllm_commit": "other"}),
    )
    with pytest.raises(ValueError, match="vllm_commit"):
        run_post_norm_gate(pre_manifest, other_commit, norm_source=weight_file)
