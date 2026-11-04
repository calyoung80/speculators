"""
Provides the entry points for converting non-speculators model checkpoints to
Speculators model format with the `convert_model` function.

It supports the following algorithms and conversion from their associated
research repositories:
- EAGLE3
- MTP
- DFlash

Functions:
    convert_model: Converts a model checkpoint to the Speculators format.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from transformers import PretrainedConfig

from speculators.convert.dflash.converter import DFlashConverter
from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.mtp.converter import MTPConverter
from speculators.convert.mtp.glm_layout import GlmMtpPolicy, build_glm_layout

__all__ = ["convert_model", "maybe_convert_external_checkpoint"]


def _coerce_bool(value: Any, flag: str) -> bool | None:
    """Accept booleans and the common string spellings from JSON kwargs."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    raise ValueError(f"{flag} must be a boolean, got {value!r}")


def _mtp_converter_from_kwargs(kwargs: dict[str, Any]) -> MTPConverter:
    """Build an MTP converter from the MTP-only optional keyword arguments."""
    prefix = kwargs.pop("native_mtp_key_prefix", None)
    own_head = _coerce_bool(kwargs.pop("own_mtp_head", None), "own_mtp_head")
    own_embed = _coerce_bool(kwargs.pop("own_mtp_embed", None), "own_mtp_embed")
    has_rot = _coerce_bool(kwargs.pop("has_rot_weight", None), "has_rot_weight")
    index_name = kwargs.pop("index_name", None)
    quant_policy = kwargs.pop("quant_metadata_policy", None)

    if quant_policy is not None and quant_policy != "copy-rebuild":
        raise ValueError(
            f"Unsupported quant_metadata_policy={quant_policy!r}; only "
            "'copy-rebuild' is supported (in-place quantized shards are rejected)."
        )

    policy = None
    if any(value is not None for value in (prefix, own_head, own_embed, has_rot)):
        policy = GlmMtpPolicy(
            own_mtp_head=bool(own_head),
            own_mtp_embed=bool(own_embed),
            has_rot_weight=bool(has_rot),
        )
    layout = (
        build_glm_layout(prefix, policy or GlmMtpPolicy())
        if prefix is not None
        else None
    )
    return MTPConverter(layout=layout, policy=policy, index_name=index_name)


def convert_model(
    model: str,
    verifier: str,
    algorithm: Literal["eagle3", "mtp", "dflash"],
    output_path: str = "converted",
    validate_device: str | None = None,
    **kwargs,
):
    """
    Convert a non speculator's model checkpoint to a speculator's model checkpoint
    for use within the Speculators library, Hugging Face Hub, or vLLM.

    algorithm=="eagle3":
        Eagle v3: https://github.com/SafeAILab/EAGLE
        ::
        # general
        convert_model(
            model="./eagle/checkpoint",
            verifier="meta-llama/Llama-3.1-8B-Instruct",
            algorithm="eagle3",
        )
        # with normalization before the residual
        convert_model(
            model="./eagle/checkpoint",
            verifier="meta-llama/Llama-3.1-8B-Instruct",
            algorithm="eagle3",
            norm_before_residual=True,
        )

    algorithm=="mtp":
        MTP (Multi-Token Prediction): models with native MTP layers
        (e.g. Qwen3-Next, Qwen3.5, Qwen3.5-MoE)
        ::
        convert_model(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            verifier="Qwen/Qwen3-Next-80B-A3B-Instruct",
            algorithm="mtp",
            num_speculative_steps=3,
        )

    algorithm=="dflash":
        DFlash: https://z-lab.ai/projects/dflash/
        ::
        convert_model(
            model="z-lab/Qwen3-8B-DFlash-b16",
            verifier="Qwen/Qwen3-8B",
            algorithm="dflash",
        )

    :param model: Path to the input model checkpoint or Hugging Face model ID.
    :param verifier: Verifier model checkpoint or Hugging Face model ID
        to attach as the verification/base model for speculative decoding
    :param algorithm: The conversion algorithm to use:
        "eagle3", "mtp", or "dflash".
    :param output_path: Directory path where the converted model will be saved.
    :param kwargs: Additional keyword arguments for the conversion algorithm.
        Options for Eagle3: {"norm_before_residual": true,
        "eagle_aux_hidden_state_layer_ids": [1,23,44]}.
        Options for MTP: {"num_speculative_steps": 3}; GLM-MoE-DSA also accepts
        {"native_mtp_key_prefix": "model.layers.78.", "index_name":
        "quant_model_weights.safetensors.index.json", "own_mtp_head": false,
        "own_mtp_embed": false, "has_rot_weight": false,
        "quant_metadata_policy": "copy-rebuild", "key_map_out": "glm_mtp_key_map.json"}.
        Options for DFlash: {"aux_hidden_state_layer_ids": [2,10,18,26,34]}.
    """

    if algorithm == "eagle3":
        Eagle3Converter().convert(
            model,
            output_path,
            verifier,
            validate=validate_device is not None,
            **kwargs,
        )
    elif algorithm == "mtp":
        converter = _mtp_converter_from_kwargs(kwargs)
        key_map_out = kwargs.pop("key_map_out", None)
        converter.convert(
            model,
            output_path,
            verifier,
            validate=validate_device is not None,
            **kwargs,
        )
        if key_map_out is not None and converter.last_key_map is not None:
            key_map_path = Path(key_map_out)
            key_map_path.parent.mkdir(parents=True, exist_ok=True)
            key_map_path.write_text(
                json.dumps(converter.last_key_map, indent=2, sort_keys=True)
            )
    elif algorithm == "dflash":
        DFlashConverter().convert(
            model,
            output_path,
            verifier,
            validate=validate_device is not None,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def maybe_convert_external_checkpoint(
    model: str | os.PathLike,
    verifier: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    output_path: str | None = None,
    config_dict: dict | None = None,
) -> str:
    """Convert an external (non-speculators) checkpoint to speculators format.

    A speculators checkpoint (config has ``speculators_model_type``) is returned
    unchanged; otherwise the external format is detected and converted (which
    requires ``verifier``) to ``output_path``, defaulting to a temp dir. Powers
    the unified ``from_pretrained`` finetuning pathway. Pass ``config_dict`` to
    reuse an already-loaded config and skip re-reading it.
    """
    if config_dict is None:
        config_dict, _ = PretrainedConfig.get_config_dict(model, cache_dir=cache_dir)
    if "speculators_model_type" in config_dict:
        return str(model)

    architectures = config_dict.get("architectures") or []
    algorithm: Literal["dflash"] = "dflash"
    if not (
        "dflash_config" in config_dict or any("DFlash" in a for a in architectures)
    ):
        raise NotImplementedError(
            f"Cannot auto-convert checkpoint '{model}': unrecognized external "
            "format. Supported auto-conversion: DFlash."
        )

    if verifier is None:
        raise ValueError(
            f"Converting an external {algorithm} checkpoint requires a verifier. "
            "Pass `verifier=<model id or path>`."
        )

    output_path = output_path or tempfile.mkdtemp(prefix="speculators_converted_")
    logger.info(f"Auto-converting external {algorithm} checkpoint to {output_path}")
    convert_model(
        model=str(model),
        verifier=verifier,
        algorithm=algorithm,
        output_path=output_path,
        cache_dir=cache_dir,
    )
    return output_path
