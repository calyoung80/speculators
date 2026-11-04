"""Unit tests for the GLM-MoE-DSA native MTP converter."""

import json

import pytest
import torch

from speculators.convert import convert_model
from speculators.convert.mtp.converter import MTPConverter
from speculators.convert.mtp.glm_layout import (
    GlmMtpPolicy,
    build_glm_layout,
    detect_glm_native_prefix,
    verify_glm_policy,
)
from speculators.convert.mtp.layout import NativeLayout
from speculators.models.mtp import MTPDraftModel
from tests.unit.convert.glm_mtp_fixtures import (
    EXPERTS,
    HIDDEN,
    MTP_PREFIX,
    NUM_LAYERS,
    TINY_CONFIG,
    build_native_mtp,
    build_non_mtp_weights,
    expected_fused_experts,
    write_checkpoint,
)

QUANT_INDEX = "quant_model_weights.safetensors.index.json"


@pytest.fixture
def native() -> dict[str, torch.Tensor]:
    return build_native_mtp()


def test_detect_glm_native_prefix(native):
    assert detect_glm_native_prefix(native) == MTP_PREFIX
    with pytest.raises(ValueError, match="No GLM MTP head"):
        detect_glm_native_prefix(["model.embed_tokens.weight"])


@pytest.mark.parametrize(
    ("native_name", "target"),
    [
        ("eh_proj.weight", "mtp_layers.0.input_proj.weight"),
        ("enorm.weight", "mtp_layers.0.token_layernorm.weight"),
        ("hnorm.weight", "mtp_layers.0.hidden_layernorm.weight"),
        (
            "shared_head.norm.weight",
            "mtp_layers.0.final_layernorm.weight",
        ),
        ("input_layernorm.weight", "mtp_layers.0.input_layernorm.weight"),
        (
            "post_attention_layernorm.weight",
            "mtp_layers.0.post_attention_layernorm.weight",
        ),
        (
            "self_attn.q_a_proj.weight",
            "mtp_layers.0.self_attn.q_a_proj.weight",
        ),
        (
            "mlp.experts.0.gate_proj.weight",
            "mtp_layers.0.mlp.experts.0.gate_proj.weight",
        ),
        (
            "mlp.gate.e_score_correction_bias",
            "mtp_layers.0.mlp.gate.e_score_correction_bias",
        ),
        (
            "self_attn.indexer.k_norm.bias",
            "mtp_layers.0.self_attn.indexer.k_norm.bias",
        ),
    ],
)
def test_layout_remap(native_name, target):
    layout = build_glm_layout(MTP_PREFIX)
    assert layout.remap_key(MTP_PREFIX + native_name) == target
    assert layout.inverse_remap_key(target) == MTP_PREFIX + native_name


def test_layout_ignores_shared_head_and_quant_metadata():
    layout = build_glm_layout(MTP_PREFIX)
    assert layout.is_ignored(MTP_PREFIX + "shared_head.head.weight")
    assert layout.is_ignored(MTP_PREFIX + "embed_tokens.weight")
    assert layout.is_ignored(MTP_PREFIX + "mlp.experts.0.gate_proj.weight_scale_inv")
    assert not layout.is_ignored(MTP_PREFIX + "shared_head.norm.weight")


def test_policy_rot_mismatch_raises(native):
    keys = list(native)
    with pytest.raises(ValueError, match="rot.weight presence"):
        verify_glm_policy(GlmMtpPolicy(has_rot_weight=True), keys, MTP_PREFIX)


def test_rot_key_with_default_policy_raises():
    keys = list(build_native_mtp(include_rot=True))
    with pytest.raises(ValueError, match="rot.weight presence"):
        verify_glm_policy(GlmMtpPolicy(), keys, MTP_PREFIX)


def test_policy_rot_present_is_accepted():
    # Both native rot storage forms pass the gate now that GlmMTPLayer owns
    # the rot projection (plan section 8.4).
    prefixed = list(build_native_mtp(include_rot=True))
    verify_glm_policy(
        GlmMtpPolicy(has_rot_weight=True, is_rot_used=True),
        prefixed,
        MTP_PREFIX,
    )
    sidecar = [key for key in prefixed if key != MTP_PREFIX + "rot.weight"]
    sidecar.append("rot.weight")
    verify_glm_policy(
        GlmMtpPolicy(has_rot_weight=True, is_rot_used=True),
        sidecar,
        MTP_PREFIX,
    )


def test_policy_own_head_missing_raises(native):
    keys = [key for key in native if "shared_head.head" not in key]
    with pytest.raises(ValueError, match="own_mtp_head=True"):
        verify_glm_policy(GlmMtpPolicy(own_mtp_head=True), keys, MTP_PREFIX)


def test_policy_own_head_present_is_accepted(native):
    # Independent head loading is implemented end to end (plan section 8.2).
    verify_glm_policy(GlmMtpPolicy(own_mtp_head=True), native, MTP_PREFIX)


def test_policy_own_embed_missing_raises(native):
    keys = [key for key in native if key != MTP_PREFIX + "embed_tokens.weight"]
    with pytest.raises(ValueError, match="own_mtp_embed=True"):
        verify_glm_policy(GlmMtpPolicy(own_mtp_embed=True), keys, MTP_PREFIX)


def test_policy_own_embed_present_is_accepted(native):
    # Independent embedding loading is implemented end to end (plan 8.2).
    verify_glm_policy(GlmMtpPolicy(own_mtp_embed=True), native, MTP_PREFIX)


def test_layout_policy_mismatch_is_rejected(tmp_path, native):
    # Layout maps the head but the converter's (default) policy says the
    # checkpoint shares the verifier head -- the extracted head would be
    # silently dropped at save time, so this must fail loudly.
    layout = build_glm_layout(MTP_PREFIX, GlmMtpPolicy(own_mtp_head=True))
    write_checkpoint(tmp_path / "native", native)

    with pytest.raises(ValueError, match="disagrees with the GLM layout"):
        MTPConverter(layout=layout).convert_to_state_dict(tmp_path / "native")


def test_policy_without_layout_mapping_is_rejected(tmp_path, native):
    layout = build_glm_layout(MTP_PREFIX)
    write_checkpoint(tmp_path / "native", native)

    with pytest.raises(ValueError, match="disagrees with the GLM layout"):
        MTPConverter(
            layout=layout, policy=GlmMtpPolicy(own_mtp_head=True)
        ).convert_to_state_dict(tmp_path / "native")


def test_convert_to_state_dict_fuses_experts(tmp_path, native):
    write_checkpoint(tmp_path / "native", native)

    converter = MTPConverter(policy=GlmMtpPolicy())
    state = converter.convert_to_state_dict(tmp_path / "native")

    gate_up, down = expected_fused_experts(native)
    torch.testing.assert_close(state["mtp_layers.0.mlp.experts.gate_up_proj"], gate_up)
    torch.testing.assert_close(state["mtp_layers.0.mlp.experts.down_proj"], down)
    assert state["mtp_layers.0.input_proj.weight"].shape == (HIDDEN, 2 * HIDDEN)
    torch.testing.assert_close(
        state["mtp_layers.0.token_layernorm.weight"],
        native[MTP_PREFIX + "enorm.weight"],
    )
    torch.testing.assert_close(
        state["mtp_layers.0.self_attn.indexer.k_norm.bias"],
        native[MTP_PREFIX + "self_attn.indexer.k_norm.bias"],
    )
    assert not any("shared_head.head" in key for key in state)
    assert not any("embed_tokens" in key for key in state)


def test_convert_to_state_dict_reports_key_map(tmp_path, native):
    write_checkpoint(tmp_path / "native", native)

    converter = MTPConverter(policy=GlmMtpPolicy())
    converter.convert_to_state_dict(tmp_path / "native")

    key_map = converter.last_key_map
    assert key_map is not None
    assert key_map["native_mtp_key_prefix"] == MTP_PREFIX
    assert (
        key_map["fused_experts"]["gate_up_proj"]
        == "mtp_layers.0.mlp.experts.gate_up_proj"
    )
    unused = set(key_map["source_keys_unused"])
    assert MTP_PREFIX + "shared_head.head.weight" in unused
    assert MTP_PREFIX + "embed_tokens.weight" in unused
    assert MTP_PREFIX + "enorm.weight" in set(key_map["source_keys_used"])


def test_quant_index_is_discovered(tmp_path, native):
    write_checkpoint(
        tmp_path / "native",
        native,
        shards=3,
        index_name=QUANT_INDEX,
    )

    converter = MTPConverter(policy=GlmMtpPolicy())
    state = converter.convert_to_state_dict(tmp_path / "native")

    assert "mtp_layers.0.mlp.experts.gate_up_proj" in state
    assert converter.last_manifest is not None
    assert converter.last_manifest["index_name"] == QUANT_INDEX


def test_explicit_index_name(tmp_path, native):
    write_checkpoint(tmp_path / "native", native, shards=2, index_name=QUANT_INDEX)

    converter = MTPConverter(index_name=QUANT_INDEX)
    state = converter.convert_to_state_dict(tmp_path / "native")

    assert "mtp_layers.0.input_proj.weight" in state


def test_end_to_end_convert_writes_reports(tmp_path, native):
    # The verifier needs top-level embed/lm_head/norm keys, mirroring a real
    # GLM checkpoint (untied, lm_head.weight present).
    native_dir = write_checkpoint(
        tmp_path / "native", {**native, **build_non_mtp_weights()}
    )
    output = tmp_path / "converted"

    MTPConverter(policy=GlmMtpPolicy()).convert(
        native_dir, output, base_model=str(native_dir), validate=True
    )

    assert (output / "glm_mtp_key_map.json").exists()
    manifest = json.loads((output / "glm_native_manifest.json").read_text())
    assert manifest["model_type"] == "glm_moe_dsa"
    assert manifest["native_mtp_layer_index"] == NUM_LAYERS
    assert manifest["native_mtp_key_prefix"] == MTP_PREFIX
    assert manifest["hidden_size"] == HIDDEN
    assert manifest["num_experts"] == EXPERTS
    assert manifest["own_mtp_head"] is False
    assert manifest["own_mtp_embed"] is False
    assert manifest["has_rot_weight"] is False
    assert manifest["index_name"] == "model.safetensors"
    assert manifest["native_key_count"] == (
        manifest["source_keys_used_count"] + manifest["source_keys_unused_count"]
    )
    assert manifest["target_key_count"] > 0

    model = MTPDraftModel.from_pretrained(str(output))
    state = model.state_dict()
    torch.testing.assert_close(
        state["mtp_layers.0.input_proj.weight"],
        native[MTP_PREFIX + "eh_proj.weight"],
    )
    gate_up, _ = expected_fused_experts(native)
    torch.testing.assert_close(state["mtp_layers.0.mlp.experts.gate_up_proj"], gate_up)


def test_end_to_end_with_quant_index_verifier(tmp_path, native):
    native_dir = write_checkpoint(
        tmp_path / "native",
        {**native, **build_non_mtp_weights()},
        shards=2,
        index_name=QUANT_INDEX,
    )
    output = tmp_path / "converted"

    MTPConverter(policy=GlmMtpPolicy()).convert(
        native_dir, output, base_model=str(native_dir), validate=True
    )

    manifest = json.loads((output / "glm_native_manifest.json").read_text())
    assert manifest["index_name"] == QUANT_INDEX
    # ``validate=True`` exercised from_pretrained, which reads the verifier
    # embed/lm_head/norm through the quant index (load_model_layers).
    assert MTPDraftModel.from_pretrained(str(output)) is not None


def test_load_model_layers_supports_quant_index(tmp_path, native):
    from speculators.utils.loading import load_model_layers

    verifier = write_checkpoint(
        tmp_path / "verifier",
        {**native, **build_non_mtp_weights()},
        shards=2,
        index_name=QUANT_INDEX,
    )

    weights = load_model_layers(
        ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
        str(verifier),
    )

    assert set(weights) == {
        "embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    }
    assert weights["lm_head.weight"].shape == (
        TINY_CONFIG["vocab_size"],
        TINY_CONFIG["hidden_size"],
    )


def test_shape_mismatch_reports_native_key(tmp_path, native):
    broken = dict(native)
    broken[MTP_PREFIX + "eh_proj.weight"] = torch.zeros(
        HIDDEN, HIDDEN, dtype=torch.float32
    )
    native_dir = write_checkpoint(tmp_path / "native", broken)

    with pytest.raises(ValueError, match="Shape mismatch for native key"):
        MTPConverter(policy=GlmMtpPolicy()).convert(
            native_dir,
            tmp_path / "converted",
            base_model=str(native_dir),
            validate=False,
        )


def test_missing_expert_projection_raises(tmp_path, native):
    broken = dict(native)
    del broken[f"{MTP_PREFIX}mlp.experts.{EXPERTS - 1}.down_proj.weight"]
    native_dir = write_checkpoint(tmp_path / "native", broken)

    with pytest.raises(
        ValueError,
        match=r"model\.layers\.2\.mlp\.experts\.3\.down_proj\.weight",
    ):
        MTPConverter(policy=GlmMtpPolicy()).convert_to_state_dict(native_dir)


def test_explicit_index_name_does_not_fall_back(tmp_path, native):
    write_checkpoint(tmp_path / "native", native, shards=2)

    with pytest.raises(FileNotFoundError, match="quant_model_weights"):
        MTPConverter(index_name=QUANT_INDEX).convert_to_state_dict(tmp_path / "native")


def test_duplicate_target_key_is_rejected(tmp_path):
    layout = NativeLayout(
        name="test",
        native_prefixes=("mtp.",),
        exact_remap={
            "mtp.a.weight": "x.weight",
            "mtp.b.weight": "x.weight",
        },
    )
    write_checkpoint(
        tmp_path / "native",
        {"mtp.a.weight": torch.ones(1), "mtp.b.weight": torch.ones(1)},
    )

    with pytest.raises(ValueError, match="must not be consumed twice"):
        MTPConverter(layout=layout).convert_to_state_dict(tmp_path / "native")


def test_mixed_dtypes_keep_fp32_master(tmp_path):
    native = build_native_mtp(dtype=torch.bfloat16)
    write_checkpoint(tmp_path / "native", native)

    converter = MTPConverter(policy=GlmMtpPolicy())
    converter.convert_to_state_dict(tmp_path / "native")

    assert converter.last_manifest is not None
    assert converter.last_manifest["source_dtype"] == "mixed"


def test_float8_source_is_rejected():
    tensor = torch.zeros(2, dtype=torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError, match="quantized"):
        MTPConverter()._check_tensor("model.layers.2.x.weight", tensor)


def test_non_mtp_keys_are_not_consumed(tmp_path, native):
    write_checkpoint(tmp_path / "native", {**native, **build_non_mtp_weights()})

    converter = MTPConverter(policy=GlmMtpPolicy())
    state = converter.convert_to_state_dict(tmp_path / "native")

    assert not any(key.startswith("model.") for key in state)


def test_convert_model_accepts_glm_kwargs(tmp_path, native):
    native_dir = write_checkpoint(
        tmp_path / "native", {**native, **build_non_mtp_weights()}
    )
    output = tmp_path / "converted"
    key_map_out = tmp_path / "key_map.json"

    convert_model(
        model=str(native_dir),
        verifier=str(native_dir),
        algorithm="mtp",
        output_path=str(output),
        num_speculative_steps=1,
        native_mtp_key_prefix=MTP_PREFIX,
        own_mtp_head=False,
        own_mtp_embed=False,
        has_rot_weight=False,
        is_rot_used=False,
        quant_metadata_policy="copy-rebuild",
        key_map_out=str(key_map_out),
    )

    assert (output / "glm_native_manifest.json").exists()
    manifest = json.loads((output / "glm_native_manifest.json").read_text())
    assert manifest["is_rot_used"] is False
    assert manifest["rot_gate_consistent"] is True
    assert json.loads(key_map_out.read_text())["native_mtp_key_prefix"] == MTP_PREFIX


def test_convert_model_rejects_inplace_quant_policy():
    with pytest.raises(ValueError, match="quant_metadata_policy"):
        convert_model(
            model="unused",
            verifier="unused",
            algorithm="mtp",
            output_path="unused",
            quant_metadata_policy="inplace",
        )


def test_convert_model_coerces_string_bools(tmp_path, native):
    native_dir = write_checkpoint(
        tmp_path / "native", {**native, **build_non_mtp_weights()}
    )
    output = tmp_path / "converted"

    # A wrongly coerced "false" would flip own_mtp_head to True and make the
    # conversion fail on the shared-head policy check.
    convert_model(
        model=str(native_dir),
        verifier=str(native_dir),
        algorithm="mtp",
        output_path=str(output),
        own_mtp_head="false",
        own_mtp_embed="false",
        has_rot_weight="false",
    )

    assert (output / "glm_native_manifest.json").exists()


def test_convert_model_rejects_non_boolean_flags():
    with pytest.raises(ValueError, match="own_mtp_head must be a boolean"):
        convert_model(
            model="unused",
            verifier="unused",
            algorithm="mtp",
            output_path="unused",
            own_mtp_head="maybe",
        )


def test_untied_glm_rejects_lm_head_fallback(tmp_path, native):
    # No top-level lm_head.weight in the verifier: from_pretrained must not
    # silently reuse the embedding matrix as the draft head for untied GLM.
    native_dir = write_checkpoint(tmp_path / "native", native)
    output = tmp_path / "converted"

    MTPConverter(policy=GlmMtpPolicy()).convert(
        native_dir, output, base_model=str(native_dir), validate=False
    )

    with pytest.raises(ValueError, match="refusing to silently reuse"):
        MTPDraftModel.from_pretrained(str(output))
