"""Per-model capabilities for the generic MTP draft model.

``ModelComponents`` is a frozen four-field ``NamedTuple``; model-specific
loader/freeze/checkpoint policy must therefore travel through a separate
capability object instead of new tuple fields.  ``core.py`` only dispatches on
these values and never contains GLM key/shape rules.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from speculators.models.mtp.core import MTPDraftModel

from loguru import logger

__all__ = [
    "MTP_CAPABILITIES",
    "MTPCapability",
    "capability_ignore_keys",
    "get_mtp_capability",
]

HiddenStateStage = Literal["pre_norm", "post_norm"]
SharedWeightLoader = Callable[["MTPDraftModel", str], None]


def _read_native_tensor(checkpoint: Path, key: str):
    """Load one tensor from a sharded or single-file safetensors package."""
    from safetensors import safe_open

    for name in (
        "quant_model_weights.safetensors.index.json",
        "model.safetensors.index.json",
    ):
        index_path = checkpoint / name
        if index_path.exists():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))[
                "weight_map"
            ]
            shard = weight_map.get(key)
            if shard is None:
                return None
            with safe_open(str(checkpoint / shard), framework="pt") as handle:
                return handle.get_tensor(key)
    single = checkpoint / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as handle:
            if key in handle:
                return handle.get_tensor(key)
    return None


def glm_shared_weight_loader(model: MTPDraftModel, weights_path: str) -> None:
    """Load own head/embed/rot weights from a native GLM package.

    Own-embedding/head assets (customer w8a8c8) carry FLOAT ``shared_head.
    head.weight`` / ``embed_tokens.weight`` under the native MTP prefix and a
    FLOAT ``rot.weight`` sidecar.  They are copied over the model parameters
    after the verifier vocab has been materialized (plan section 8.2).
    """
    from safetensors import safe_open  # noqa: F401

    checkpoint = Path(weights_path)
    layer_config = model.config.transformer_layer_config
    prefix = f"model.layers.{layer_config.num_hidden_layers}."
    candidates: dict[str, str] = {
        "lm_head.weight": f"{prefix}shared_head.head.weight",
        "embed_tokens.weight": f"{prefix}embed_tokens.weight",
        "mtp_layers.0.rot.weight": "rot.weight",
    }
    if "rot.weight" not in json_names(checkpoint):
        candidates["mtp_layers.0.rot.weight"] = f"{prefix}rot.weight"

    for parameter_name, native_key in candidates.items():
        tensor = _read_native_tensor(checkpoint, native_key)
        if tensor is None:
            continue
        parameter = model.get_parameter(parameter_name)
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Native '{native_key}' shape {tuple(tensor.shape)} does not "
                f"match '{parameter_name}' {tuple(parameter.shape)}."
            )
        parameter.data.copy_(tensor.to(parameter.dtype))
        logger.info(
            f"Loaded own MTP weight '{parameter_name}' from native '{native_key}'"
        )


def json_names(checkpoint: Path) -> set[str]:
    """Return the key set of the checkpoint (index-aware, header only)."""
    for name in (
        "quant_model_weights.safetensors.index.json",
        "model.safetensors.index.json",
    ):
        index_path = checkpoint / name
        if index_path.exists():
            return set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    from safetensors import safe_open

    single = checkpoint / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as handle:
            return set(handle.keys())
    return set()


def apply_glm_asset_facts(config: Any, verifier_path: str | Path) -> None:
    """Populate ``mtp_own_head`` / ``mtp_own_embed`` / ``mtp_has_rot``.

    Reads the native GLM package at ``verifier_path`` (index headers only for
    key presence, quant metadata for the rot gate) so the training model's
    capability reflects the actual asset.  No-op for non-GLM model types.
    """
    layer_config = config.transformer_layer_config
    if getattr(layer_config, "model_type", None) != "glm_moe_dsa":
        return
    checkpoint = Path(verifier_path)
    keys = json_names(checkpoint)
    if not keys:
        return
    from speculators.convert.mtp.glm_layout import (
        detect_glm_native_prefix,
        read_quant_is_rot_used,
    )

    prefix = detect_glm_native_prefix(keys)
    config.mtp_own_head = f"{prefix}shared_head.head.weight" in keys
    config.mtp_own_embed = f"{prefix}embed_tokens.weight" in keys
    config.mtp_has_rot = "rot.weight" in keys or f"{prefix}rot.weight" in keys
    if config.mtp_has_rot:
        quant_is_rot_used = read_quant_is_rot_used(checkpoint)
        if quant_is_rot_used is False:
            raise ValueError(
                "rot_gate_consistent=false: quant metadata is_rot_used=False "
                "but the package carries a rot.weight tensor."
            )


@dataclass(frozen=True)
class MTPCapability:
    """Non-architectural policy for one MTP model type.

    The values are structural defaults.  Native-package facts (``own_mtp_*``,
    ``has_rot_weight``) come from the probe/manifest and may be replaced in
    :data:`MTP_CAPABILITIES` once the customer checkpoint is available.
    """

    model_type: str
    hidden_state_stage: HiddenStateStage = "pre_norm"
    own_mtp_head: bool = False
    own_mtp_embed: bool = False
    own_mtp_final_norm: bool = True
    has_rot_weight: bool = False
    native_mtp_key_prefix: str | None = None
    index_name: str = "model.safetensors.index.json"
    shared_mtp_weights_path: str | None = None
    indexer_chunk_size: int | None = None
    freeze_indexer: bool = False
    shared_weight_loader: SharedWeightLoader | None = field(
        default=None, compare=False, repr=False
    )


MTP_CAPABILITIES: dict[str, MTPCapability] = {
    "glm_moe_dsa": MTPCapability(
        model_type="glm_moe_dsa",
        # K=1 main path trains on post-norm teacher captures; ``verifier_norm``
        # is only for pre-norm captures (Phase 5 / Phase 8).
        hidden_state_stage="post_norm",
        # Structural defaults from the community probe.  Per-asset facts
        # (own head/embed, rot) are carried by the speculator config
        # (``mtp_own_head`` / ``mtp_own_embed`` / ``mtp_has_rot``), which
        # ``MTPDraftModel`` applies as overrides over these defaults.
        own_mtp_head=False,
        own_mtp_embed=False,
        own_mtp_final_norm=True,
        has_rot_weight=False,
        native_mtp_key_prefix="model.layers.{num_hidden_layers}.",
        index_name="quant_model_weights.safetensors.index.json",
        indexer_chunk_size=None,
        freeze_indexer=True,
        shared_weight_loader=glm_shared_weight_loader,
    ),
}


def get_mtp_capability(model_type: str) -> MTPCapability | None:
    """Return the capability for ``model_type`` or ``None`` for generic types."""
    return MTP_CAPABILITIES.get(model_type)


def capability_ignore_keys(
    ignore_keys: Any, capability: MTPCapability | None
) -> set[str]:
    """Return checkpoint keys to ignore, honouring own head/embed weights.

    :param ignore_keys: Existing ignore collection (set/list/None).
    :param capability: Capability policy or ``None`` (generic path).
    """
    keys = set(ignore_keys or [])
    if capability is None:
        return keys
    if capability.own_mtp_head:
        keys.discard("lm_head.weight")
    if capability.own_mtp_embed:
        keys.discard("embed_tokens.weight")
    return keys
