"""Unit tests for stitching GLM MTP weights back into a native checkpoint."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from speculators.cli.stitch import stitch
from speculators.convert.mtp.converter import MTPConverter
from speculators.models.mtp import MTPDraftModel
from tests.unit.convert.glm_mtp_fixtures import (
    EXPERTS,
    INTER,
    MTP_PREFIX,
    build_native_mtp,
    build_non_mtp_weights,
    write_checkpoint,
)

QUANT_INDEX = "quant_model_weights.safetensors.index.json"


def _make_finetuned(
    tmp_path, native: dict[str, torch.Tensor], *, mutate: bool = True
) -> tuple[Path, MTPDraftModel]:
    native_dir = write_checkpoint(
        tmp_path / "native", {**native, **build_non_mtp_weights()}
    )
    converted = tmp_path / "converted"
    MTPConverter().convert(
        native_dir, converted, base_model=str(native_dir), validate=False
    )
    model = MTPDraftModel.from_pretrained(str(converted))
    if mutate:
        with torch.no_grad():
            model.mtp_layers[0].input_proj.weight.add_(0.5)
            model.mtp_layers[0].token_layernorm.weight.add_(0.25)
            model.mtp_layers[0].mlp.experts.gate_up_proj.add_(0.125)
            model.mtp_layers[0].self_attn.indexer.k_norm.bias.add_(0.75)
    finetuned = tmp_path / "finetuned"
    model.save_pretrained(str(finetuned))
    return finetuned, model


def _load_sharded(directory) -> tuple[dict[str, torch.Tensor], str]:
    index_name = next(
        name
        for name in ("model.safetensors.index.json", QUANT_INDEX)
        if (directory / name).exists()
    )
    weight_map = json.loads((directory / index_name).read_text())["weight_map"]
    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        weights.update(load_file(str(directory / shard)))
    return weights, index_name


def test_stitch_round_trip(tmp_path):
    native = build_native_mtp()
    extras = build_non_mtp_weights()
    finetuned, model = _make_finetuned(tmp_path, native)
    verifier = write_checkpoint(tmp_path / "verifier", {**native, **extras})
    output = tmp_path / "stitched"

    stitch(finetuned, verifier, output)

    stitched = load_file(str(output / "model.safetensors"))

    for key, tensor in extras.items():
        assert torch.equal(stitched[key], tensor), key
    for name in ("shared_head.head.weight", "embed_tokens.weight"):
        assert torch.equal(stitched[MTP_PREFIX + name], native[MTP_PREFIX + name])

    layer = model.mtp_layers[0]
    torch.testing.assert_close(
        stitched[MTP_PREFIX + "eh_proj.weight"], layer.input_proj.weight.detach()
    )
    torch.testing.assert_close(
        stitched[MTP_PREFIX + "enorm.weight"],
        layer.token_layernorm.weight.detach(),
    )
    torch.testing.assert_close(
        stitched[MTP_PREFIX + "self_attn.indexer.k_norm.bias"],
        layer.self_attn.indexer.k_norm.bias.detach(),
    )

    gate_up = layer.mlp.experts.gate_up_proj.detach()
    down = layer.mlp.experts.down_proj.detach()
    for expert in range(EXPERTS):
        prefix = f"{MTP_PREFIX}mlp.experts.{expert}."
        torch.testing.assert_close(
            stitched[prefix + "gate_proj.weight"], gate_up[expert, :INTER]
        )
        torch.testing.assert_close(
            stitched[prefix + "up_proj.weight"], gate_up[expert, INTER:]
        )
        torch.testing.assert_close(stitched[prefix + "down_proj.weight"], down[expert])

    assert not any(key.startswith("mtp_layers.") for key in stitched)


def test_stitch_preserves_quant_index_name(tmp_path):
    native = build_native_mtp()
    extras = build_non_mtp_weights()
    finetuned, model = _make_finetuned(tmp_path, native)
    verifier = write_checkpoint(
        tmp_path / "verifier",
        {**native, **extras},
        shards=3,
        index_name=QUANT_INDEX,
    )
    output = tmp_path / "stitched"

    stitch(finetuned, verifier, output)

    stitched, index_name = _load_sharded(output)
    assert index_name == QUANT_INDEX
    assert not (output / "model.safetensors.index.json").exists()
    torch.testing.assert_close(
        stitched[MTP_PREFIX + "eh_proj.weight"],
        model.mtp_layers[0].input_proj.weight.detach(),
    )
    assert torch.equal(stitched["model.norm.weight"], extras["model.norm.weight"])


def test_stitch_rejects_quant_siblings(tmp_path):
    clean_native = build_native_mtp()
    finetuned, _ = _make_finetuned(tmp_path, clean_native)
    quant_native = build_native_mtp(include_quant_scale=True)
    verifier = write_checkpoint(
        tmp_path / "verifier", {**quant_native, **build_non_mtp_weights()}
    )

    with pytest.raises(NotImplementedError, match="quantized metadata"):
        stitch(finetuned, verifier, tmp_path / "stitched")


def test_stitch_bf16_round_trip_is_bitwise(tmp_path):
    native = build_native_mtp(dtype=torch.bfloat16)
    finetuned, _ = _make_finetuned(tmp_path, native, mutate=False)
    verifier = write_checkpoint(
        tmp_path / "verifier", {**native, **build_non_mtp_weights()}
    )
    output = tmp_path / "stitched"

    stitch(finetuned, verifier, output)

    stitched = load_file(str(output / "model.safetensors"))
    for name in (
        "enorm.weight",
        "eh_proj.weight",
        "self_attn.q_a_proj.weight",
        "mlp.experts.0.gate_proj.weight",
        f"mlp.experts.{EXPERTS - 1}.down_proj.weight",
    ):
        key = MTP_PREFIX + name
        assert stitched[key].dtype == torch.bfloat16
        assert torch.equal(stitched[key], native[key]), key

    bias_key = MTP_PREFIX + "mlp.gate.e_score_correction_bias"
    assert stitched[bias_key].dtype == torch.float32
    assert torch.equal(stitched[bias_key], native[bias_key])


def test_stitch_does_not_add_missing_head_or_embed(tmp_path):
    native = build_native_mtp(include_head_embed=False)
    finetuned, _ = _make_finetuned(tmp_path, native)
    verifier = write_checkpoint(
        tmp_path / "verifier", {**native, **build_non_mtp_weights()}
    )
    output = tmp_path / "stitched"

    stitch(finetuned, verifier, output)

    stitched = load_file(str(output / "model.safetensors"))
    assert MTP_PREFIX + "shared_head.head.weight" not in stitched
    assert MTP_PREFIX + "embed_tokens.weight" not in stitched


def test_stitch_rejects_layer_index_mismatch(tmp_path):
    # A finetuned config claiming a different verifier depth than the native
    # MTP head means the directories do not belong together.
    native = build_native_mtp()
    finetuned, _ = _make_finetuned(tmp_path, native)
    config_path = finetuned / "config.json"
    config = json.loads(config_path.read_text())
    config["transformer_layer_config"]["num_hidden_layers"] = 3
    config_path.write_text(json.dumps(config))
    verifier = write_checkpoint(
        tmp_path / "verifier", {**native, **build_non_mtp_weights()}
    )

    with pytest.raises(ValueError, match="num_hidden_layers"):
        stitch(finetuned, verifier, tmp_path / "stitched")
