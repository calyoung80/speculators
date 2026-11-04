"""Per-model capabilities for the generic MTP draft model.

``ModelComponents`` is a frozen four-field ``NamedTuple``; model-specific
loader/freeze/checkpoint policy must therefore travel through a separate
capability object instead of new tuple fields.  ``core.py`` only dispatches on
these values and never contains GLM key/shape rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from speculators.models.mtp.core import MTPDraftModel

__all__ = [
    "MTP_CAPABILITIES",
    "MTPCapability",
    "capability_ignore_keys",
    "get_mtp_capability",
]

HiddenStateStage = Literal["pre_norm", "post_norm"]
SharedWeightLoader = Callable[["MTPDraftModel", str], None]


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
        # Community probe (glm_asset_probe_community.json); override from the
        # customer probe when it becomes available.
        own_mtp_head=False,
        own_mtp_embed=False,
        own_mtp_final_norm=True,
        has_rot_weight=False,
        native_mtp_key_prefix="model.layers.{num_hidden_layers}.",
        index_name="quant_model_weights.safetensors.index.json",
        indexer_chunk_size=None,
        freeze_indexer=True,
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
