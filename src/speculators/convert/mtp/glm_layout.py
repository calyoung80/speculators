"""GLM-MoE-DSA native MTP key layout, policy and manifest helpers.

The native GLM MTP head lives *after* the decoder stack at
``model.layers.{num_hidden_layers}.*``.  Its module names differ from the
training-state names (``enorm``/``hnorm``/``eh_proj``/``shared_head.norm``),
and its routed experts are stored per-expert, so the converter remaps names
first and then fuses experts with ``MTPConverter._fuse_moe_experts``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from speculators.convert.mtp.layout import NativeLayout

if TYPE_CHECKING:
    from collections.abc import Iterable

GLM_MODEL_TYPE = "glm_moe_dsa"

_NATIVE_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.eh_proj\.weight$")

# Quantized checkpoints attach auxiliary tensors next to a float weight.  The
# community fp8 package uses ``*.weight_scale_inv``; customer w4a8 builds may
# use other names.  These keys are reported as unused, never loaded.
_QUANT_SUFFIXES = (
    ".weight_scale_inv",
    ".weight_scale",
    ".weight_zero_point",
    ".weight_offset",
    ".weight_quant_bias",
)

_FUSED_EXPERT_TARGETS = {
    "gate_up_proj": "mtp_layers.0.mlp.experts.gate_up_proj",
    "down_proj": "mtp_layers.0.mlp.experts.down_proj",
}


@dataclass(frozen=True)
class GlmMtpPolicy:
    """Native-package facts that drive head/embed/rot handling.

    Defaults come from the community probe (``own_mtp_head=false``,
    ``own_mtp_embed=false``, ``own_mtp_final_norm=true``,
    ``has_rot_weight=false``).  The customer w4a8 probe must override these
    when it becomes available.
    """

    own_mtp_head: bool = False
    own_mtp_embed: bool = False
    own_mtp_final_norm: bool = True
    has_rot_weight: bool = False


def detect_glm_native_prefix(keys: Iterable[str]) -> str:
    """Return ``model.layers.{N}.`` for the GLM MTP head in ``keys``."""
    for key in keys:
        match = _NATIVE_LAYER_PATTERN.match(key)
        if match:
            return f"model.layers.{match.group(1)}."
    raise ValueError(
        "No GLM MTP head found. Expected a key like 'model.layers.<N>.eh_proj.weight'."
    )


def build_glm_layout(prefix: str, policy: GlmMtpPolicy | None = None) -> NativeLayout:
    """Build the native -> training-state layout for one GLM MTP head."""
    policy = policy or GlmMtpPolicy()
    exact_remap = {f"{prefix}eh_proj.weight": "mtp_layers.0.input_proj.weight"}
    ignored_keys: list[str] = []

    if policy.own_mtp_head:
        exact_remap[f"{prefix}shared_head.head.weight"] = "lm_head.weight"
    else:
        # Ignored even when present, matching the ``has_own_lm_head=False``
        # runtime behaviour.
        ignored_keys.append(f"{prefix}shared_head.head.weight")

    if policy.own_mtp_embed:
        exact_remap[f"{prefix}embed_tokens.weight"] = "embed_tokens.weight"
    else:
        ignored_keys.append(f"{prefix}embed_tokens.weight")

    prefix_remap = (
        (f"{prefix}enorm.", "mtp_layers.0.token_layernorm."),
        (f"{prefix}hnorm.", "mtp_layers.0.hidden_layernorm."),
        (f"{prefix}shared_head.norm.", "mtp_layers.0.final_layernorm."),
        (prefix, "mtp_layers.0."),
    )
    return NativeLayout(
        name=GLM_MODEL_TYPE,
        native_prefixes=(prefix,),
        exact_remap=exact_remap,
        prefix_remap=prefix_remap,
        ignored_keys=tuple(ignored_keys),
        ignored_suffixes=_QUANT_SUFFIXES,
    )


def verify_glm_policy(
    policy: GlmMtpPolicy,
    keys: Iterable[str],
    prefix: str,
) -> None:
    """Fail fast when the policy and the checkpoint disagree.

    ``rot.weight`` presence must match the probe flag in both directions.
    Independent head/embed storage is not implemented end-to-end yet (the
    glue has no loader body), so it is rejected before training can silently
    drop those tensors.
    """
    key_set = set(keys)

    has_rot = f"{prefix}rot.weight" in key_set
    if has_rot != policy.has_rot_weight:
        raise ValueError(
            f"rot.weight presence ({has_rot}) does not match "
            f"has_rot_weight={policy.has_rot_weight} for prefix '{prefix}'."
        )
    if has_rot:
        raise NotImplementedError(
            "GLM checkpoints with model.layers.<N>.rot.weight are not "
            "supported yet: GlmMTPLayer has no rot projection. Implement the "
            "Phase 1 rot fallback before converting this checkpoint."
        )

    has_head = f"{prefix}shared_head.head.weight" in key_set
    if policy.own_mtp_head:
        if not has_head:
            raise ValueError(
                f"own_mtp_head=True but '{prefix}shared_head.head.weight' "
                "is missing from the checkpoint."
            )
        raise NotImplementedError(
            "Independent GLM MTP head loading is not implemented yet; "
            "train/export with own_mtp_head=false (shared verifier head)."
        )

    has_embed = f"{prefix}embed_tokens.weight" in key_set
    if policy.own_mtp_embed:
        if not has_embed:
            raise ValueError(
                f"own_mtp_embed=True but '{prefix}embed_tokens.weight' "
                "is missing from the checkpoint."
            )
        raise NotImplementedError(
            "Independent GLM MTP embedding loading is not implemented yet; "
            "train/export with own_mtp_embed=false (shared verifier embed)."
        )


def build_glm_key_map(
    layout: NativeLayout,
    keys: Iterable[str],
) -> dict:
    """Build the ``glm_mtp_key_map.json`` report (plan section 21.11)."""
    prefix = layout.native_prefixes[0]
    source_used: list[str] = []
    source_unused: list[str] = []
    for key in keys:
        if not layout.is_native_key(key):
            continue
        if layout.is_ignored(key):
            source_unused.append(key)
        else:
            source_used.append(key)

    return {
        "native_mtp_key_prefix": prefix,
        "exact": dict(layout.exact_remap),
        "prefix": [[source, target] for source, target in layout.prefix_remap],
        "fused_experts": dict(_FUSED_EXPERT_TARGETS),
        "source_keys_used": sorted(source_used),
        "source_keys_unused": sorted(source_unused),
    }


def build_glm_native_manifest(
    source_config: dict,
    layout: NativeLayout,
    policy: GlmMtpPolicy,
    index_name: str,
    source_keys_used: list[str],
    source_keys_unused: list[str],
    dtypes: Iterable[object] = (),
    target_key_count: int | None = None,
) -> dict:
    """Build the ``glm_native_manifest.json`` record (plan section 21.4)."""
    prefix = layout.native_prefixes[0]
    layer_index = int(prefix.rstrip(".").rsplit(".", 1)[-1])
    dtype_names = sorted({str(dtype) for dtype in dtypes})
    if len(dtype_names) > 1:
        source_dtype: str | None = "mixed"
    elif dtype_names:
        source_dtype = dtype_names[0]
    else:
        source_dtype = None

    return {
        "schema_version": "glm-mtp/v1",
        "model_type": GLM_MODEL_TYPE,
        "verifier_source_layer_ordinal": layer_index,
        "verifier_source_layer_index_zero_based": layer_index - 1,
        "verifier_source_tensor_semantics": "decoder_output_before_model_norm",
        "capture_tensor_semantics": "model_norm_output",
        "vllm_target_layer_id": layer_index,
        "native_mtp_layer_index": layer_index,
        "native_mtp_key_prefix": prefix,
        "native_mtp_tensor_semantics": "native_mtp_head",
        "hidden_state_stage": "post_norm",
        "source_norm": "model.norm",
        "hidden_size": source_config.get("hidden_size"),
        "vocab_size": source_config.get("vocab_size"),
        "num_experts": source_config.get("n_routed_experts"),
        "moe_intermediate_size": source_config.get("moe_intermediate_size"),
        "source_layout": "per_expert",
        "target_layout": "stacked_3d",
        "index_name": index_name,
        "own_mtp_head": policy.own_mtp_head,
        "own_mtp_embed": policy.own_mtp_embed,
        "own_mtp_final_norm": policy.own_mtp_final_norm,
        "has_rot_weight": policy.has_rot_weight,
        "is_rot_used": None,
        "index_share_for_mtp_iteration": source_config.get(
            "index_share_for_mtp_iteration", False
        ),
        "rms_norm_eps": source_config.get("rms_norm_eps"),
        "q_a_layernorm_eps": source_config.get("rms_norm_eps"),
        "kv_a_layernorm_eps": source_config.get("rms_norm_eps"),
        "source_dtype": source_dtype,
        "training_dtype": "bfloat16",
        "export_dtype": "bfloat16",
        "native_key_count": len(source_keys_used) + len(source_keys_unused),
        "source_keys_used_count": len(source_keys_used),
        "source_keys_unused_count": len(source_keys_unused),
        "target_key_count": target_key_count,
        "source_keys_used": sorted(source_keys_used),
        "source_keys_unused": sorted(source_keys_unused),
        "file_sha256": {},
    }
