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

# Own-embed MTP variants (customer w8a8c8) ship ``embed_tokens`` /
# ``shared_head.norm`` instead of the ``enorm``/``hnorm``/``eh_proj`` glue.
_NATIVE_LAYER_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.(?:eh_proj|shared_head\.norm|embed_tokens)\.weight$"
)

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
    # From the quant metadata.  ``None`` = not probed; a concrete value must
    # agree with ``has_rot_weight`` or conversion/export is blocked (plan
    # section 21.4, ``rot_gate_consistent``).
    is_rot_used: bool | None = None


# The three GLM MTP RMSNorm modules that may carry a quant-pack zero bias.
# ``self_attn.indexer.k_norm`` is a LayerNorm with a real bias and must never
# be dropped here.
_GLM_MTP_NORM_BIAS_TARGETS = (
    "mtp_layers.0.token_layernorm.bias",
    "mtp_layers.0.hidden_layernorm.bias",
    "mtp_layers.0.final_layernorm.bias",
)


def detect_glm_native_prefix(keys: Iterable[str]) -> str:
    """Return ``model.layers.{N}.`` for the GLM MTP head in ``keys``.

    Both MTP variants are recognised: the glue form (``eh_proj``) and the
    own-embed form (``shared_head.norm`` / ``embed_tokens``).
    """
    for key in keys:
        match = _NATIVE_LAYER_PATTERN.match(key)
        if match:
            return f"model.layers.{match.group(1)}."
    raise ValueError(
        "No GLM MTP head found. Expected a key like 'model.layers.<N>."
        "eh_proj.weight' or 'model.layers.<N>.shared_head.norm.weight'."
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

    if policy.has_rot_weight:
        # Native GLM packages store the MTP rot projection either under the
        # training-state prefix or as a top-level ``rot.weight`` sidecar (a
        # dedicated ``rot.safetensors`` shard; vllm-ascend remaps ``rot.*`` to
        # ``model.layers.<N>.rot.*`` at load).  Training state keeps it inside
        # the MTP layer (plan section 8.4 / section 21 key map).
        exact_remap["rot.weight"] = "mtp_layers.0.rot.weight"
        exact_remap[f"{prefix}rot.weight"] = "mtp_layers.0.rot.weight"

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


def verify_glm_layer_index(prefix: str, source_config: dict) -> None:
    """Check that the native MTP head follows the verifier stack.

    ``model.layers.<N>.*`` is the MTP head; ``N`` must equal the verifier's
    ``num_hidden_layers`` (plan sections 17.4/21.6).
    """
    match = _NATIVE_LAYER_PATTERN.match(f"{prefix}eh_proj.weight")
    if match is None:
        raise ValueError(f"Invalid GLM MTP prefix {prefix!r}.")
    layer_index = int(match.group(1))
    num_hidden_layers = source_config.get("num_hidden_layers")
    if num_hidden_layers is not None and layer_index != num_hidden_layers:
        raise ValueError(
            f"Native MTP layer index {layer_index} does not match "
            f"num_hidden_layers={num_hidden_layers} for prefix '{prefix}'."
        )


def verify_glm_policy(
    policy: GlmMtpPolicy,
    keys: Iterable[str],
    prefix: str,
) -> None:
    """Fail fast when the policy and the checkpoint disagree.

    ``rot.weight`` presence must match the probe flag in both directions
    (top-level sidecar and training-state prefix forms both count).
    """
    key_set = set(keys)

    has_rot = "rot.weight" in key_set or f"{prefix}rot.weight" in key_set
    if has_rot != policy.has_rot_weight:
        raise ValueError(
            f"rot.weight presence ({has_rot}) does not match "
            f"has_rot_weight={policy.has_rot_weight} for prefix '{prefix}'."
        )
    if policy.is_rot_used is not None and policy.is_rot_used != has_rot:
        raise ValueError(
            f"rot_gate_consistent=false: quant metadata is_rot_used="
            f"{policy.is_rot_used} but rot.weight presence={has_rot} for "
            f"prefix '{prefix}'."
        )
    # rot is supported: GlmMTPLayer owns the projection and applies it at the
    # start of the MTP forward (plan section 8.4); the layout remaps the
    # native sidecar onto ``mtp_layers.0.rot.weight``.

    if not policy.own_mtp_final_norm:
        raise NotImplementedError(
            "own_mtp_final_norm=false is not implemented yet: the training "
            "model always owns mtp_layers.0.final_layernorm."
        )
    if f"{prefix}shared_head.norm.weight" not in key_set:
        raise ValueError(
            f"own_mtp_final_norm=True but '{prefix}shared_head.norm.weight' "
            "is missing from the checkpoint."
        )

    if policy.own_mtp_head and (f"{prefix}shared_head.head.weight" not in key_set):
        raise ValueError(
            f"own_mtp_head=True but '{prefix}shared_head.head.weight' "
            "is missing from the checkpoint."
        )

    if policy.own_mtp_embed and (f"{prefix}embed_tokens.weight" not in key_set):
        raise ValueError(
            f"own_mtp_embed=True but '{prefix}embed_tokens.weight' "
            "is missing from the checkpoint."
        )


def drop_glm_optional_norm_bias(
    weights: dict,
    layout: NativeLayout,
) -> list[str]:
    """Drop zero-valued optional RMSNorm biases from extracted GLM weights.

    Returns the native keys that were dropped.  A non-zero bias means the
    checkpoint does not follow the ``GlmMoeDsaRMSNorm`` contract and must be
    rejected instead of silently ignored (plan section 9.2).  The DSA
    indexer's LayerNorm ``k_norm.bias`` is a real parameter and is never in
    the target list.
    """
    dropped: list[str] = []
    for target in _GLM_MTP_NORM_BIAS_TARGETS:
        tensor = weights.get(target)
        if tensor is None:
            continue
        if tensor.count_nonzero().item() != 0:
            native_key = layout.inverse_remap_key(target)
            raise ValueError(
                f"Non-zero optional RMSNorm bias at native key "
                f"'{native_key}' ({target}); GlmMoeDsaRMSNorm has no bias. "
                "Only an all-zero quant-pack placeholder may be dropped."
            )
        del weights[target]
        dropped.append(layout.inverse_remap_key(target))
    return dropped


def read_quant_is_rot_used(checkpoint_dir) -> bool | None:
    """Read ``is_rot_used`` from the checkpoint's quant metadata.

    Resolution order: ``quantization_config.json`` (``quantization_config``
    subtree or top level), ``config.json`` (same), then the MindIE-style
    ``quant_model_description.json`` (top-level key).  ``None`` = absent.
    """
    import json
    from pathlib import Path

    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    for name in (
        "quantization_config.json",
        "config.json",
        "quant_model_description.json",
    ):
        path = directory / name if directory is not None else None
        if path is None or not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        quant = raw.get("quantization_config") if isinstance(raw, dict) else None
        if not isinstance(quant, dict):
            quant = raw if isinstance(raw, dict) else {}
        if "is_rot_used" in quant:
            return bool(quant["is_rot_used"])
    return None


def derive_glm_policy(
    keys: Iterable[str], prefix: str, checkpoint_dir=None
) -> GlmMtpPolicy:
    """Derive the GLM policy from the native package itself.

    Key presence decides ``own_mtp_head`` / ``own_mtp_embed`` /
    ``has_rot_weight`` (top-level ``rot.weight`` sidecar and training-state
    prefix forms both count); ``is_rot_used`` comes from the quant metadata.
    The verifier asset is the single source of truth -- converter, stitch and
    ``MTPDraftModel.from_pretrained`` all derive the policy this way so a
    mismatched hand-written policy cannot diverge from the checkpoint.
    """
    key_set = set(keys)
    return GlmMtpPolicy(
        own_mtp_head=f"{prefix}shared_head.head.weight" in key_set,
        own_mtp_embed=f"{prefix}embed_tokens.weight" in key_set,
        own_mtp_final_norm=True,
        has_rot_weight="rot.weight" in key_set or f"{prefix}rot.weight" in key_set,
        is_rot_used=read_quant_is_rot_used(checkpoint_dir),
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
        "is_rot_used": policy.is_rot_used,
        "rot_gate_consistent": (
            None
            if policy.is_rot_used is None
            else policy.is_rot_used == policy.has_rot_weight
        ),
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
