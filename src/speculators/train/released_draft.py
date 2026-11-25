"""Build a DFlash-family draft model from a *released* deploy-format checkpoint.

The released draft directories on the cluster's shared cephfs
(``GLM-5.3-DSpark``, ``GLM-5.3-DFlash2``) and public packed releases such as
UCloud's ``GLM-5.2-FP8-DFlash`` (ModelScope/HuggingFace, ``UCloud-AILab``) are
vLLM **deploy** checkpoints: a flat ``config.json`` (``model_type=qwen3`` plus
a ``dflash_config``/``dspark_config`` section) and ``model.safetensors``, with
no ``speculators_config``.  ``SpeculatorModel.from_pretrained`` therefore
cannot load them.

Two release styles exist and both are supported:

* **A (GLM-5.3-style)**: every vocab-side tensor is omitted; they must be
  rebuilt from the verifier via ``load_verifier_weights``.
* **B (packed, UCloud-style)**: the target's frozen ``embed_tokens`` /
  ``lm_head`` are packed into the release so the draft can run standalone;
  ``verifier_lm_head`` / ``verifier_norm`` are still absent.  The packed
  tensors load by name+shape like any other released tensor, and -- when a
  verifier is given -- the verifier rebuild runs **after** and wins, because
  training targets are produced by the verifier we train against (the values
  are bit-identical for a matched release/verifier pair; a difference logs a
  warning instead of failing, e.g. an FP8 target vs a bf16 verifier).

This module turns the flat description into the full speculator config,
loads the released tensors by exact name **and** shape, and then rebuilds the
verifier-owned tensors (``embed_tokens`` / ``lm_head`` / ``verifier_lm_head`` /
``verifier_norm``) from the verifier checkpoint.

Fail-closed policy
------------------
Everything that cannot be satisfied raises ``ValueError`` naming the offending
tensors: silently training from partially-random weights is worse than
refusing to start.  The only escape hatch is ``init_unloaded="random"``, which
exists for memory-bounded smoke runs (e.g. a compressed draft vocabulary that
cannot hold the released full-vocabulary ``markov_w2``); it fills the affected
tensors with ``normal_(std=0.02)`` and logs exactly which ones.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import transformers
from packaging import version
from safetensors import safe_open
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

__all__ = [
    "RELEASED_DRAFT_SPECULATORS",
    "ReleasedDraftStats",
    "build_draft_from_released",
    "read_released_draft_config",
]

logger = logging.getLogger(__name__)

#: Speculators whose released checkpoints carry the flat deploy format this
#: module understands (Qwen3-style decoder + ``dflash_config`` section):
#: DSpark and DFlash2 ship as GLM-5.3 releases on cephfs; plain DFlash (v1)
#: ships as public packed releases (e.g. UCloud GLM-5.2-FP8-DFlash).
RELEASED_DRAFT_SPECULATORS = frozenset({"dspark", "dflash", "dflash2"})

#: Vocab-side tensors that belong to the verifier.  GLM-5.3-style (A) releases
#: omit all of them -- they are rebuilt from the verifier.  Packed (B) releases
#: ship the target's frozen ``embed_tokens``/``lm_head`` for standalone
#: inference; ``verifier_lm_head``/``verifier_norm`` are absent in both styles.
_VERIFIER_OWNED_KEYS = frozenset(
    {
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    }
)

_TRANSFORMERS_AT_LEAST_5 = version.parse(transformers.__version__) >= version.parse(
    "5.0.0"
)

# Layer-index prefix of the draft decoder ("layers.0.", "layers.12.", ...).
_LAYER_KEY_RE = re.compile(r"^layers\.(\d+)\.")


@dataclass(frozen=True)
class ReleasedDraftStats:
    """What ``build_draft_from_released`` actually loaded.

    The counts make the "trained from released weights" claim checkable:
    ``loaded_tensors + len(layer_reduction_skipped) + len(shape_skipped)``
    must equal ``released_tensors`` for a faithful full-profile build.
    ``release_vocab_side`` lists which verifier-owned tensors the release
    itself packed (style B); the rest were rebuilt from the verifier or, in a
    random smoke profile, filled.
    """

    released_tensors: int = 0
    loaded_tensors: int = 0
    layer_reduction_skipped: tuple[str, ...] = field(default_factory=tuple)
    shape_skipped: tuple[str, ...] = field(default_factory=tuple)
    random_filled: tuple[str, ...] = field(default_factory=tuple)
    release_vocab_side: tuple[str, ...] = field(default_factory=tuple)


def read_released_draft_config(released_dir: str | Path) -> dict[str, Any]:
    """Parse and normalise a released deploy-format ``config.json``.

    The released files duplicate some keys across the top level, the
    ``dflash_config`` section and (DSpark) the ``dspark_config`` section; the
    top level wins, then ``dflash_config``, then ``dspark_config``.  The
    returned dict is flat and architecture-neutral so the builder (and tests)
    never have to re-implement the merge order.
    """
    path = Path(released_dir)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(
            f"--from-released: '{path}' has no config.json; a released "
            "deploy-format checkpoint directory is required (flat qwen3-style "
            "config.json + model.safetensors)"
        )
    top = json.loads(config_path.read_text())
    if not isinstance(top, dict):
        raise ValueError(
            f"--from-released: '{config_path}' does not contain a JSON object"
        )
    dflash_cfg = top.get("dflash_config") or {}
    dspark_cfg = top.get("dspark_config") or {}
    if not isinstance(dflash_cfg, dict) or not isinstance(dspark_cfg, dict):
        raise ValueError(
            f"--from-released: 'dflash_config'/'dspark_config' in "
            f"'{config_path}' must be JSON objects"
        )
    merged: dict[str, Any] = {**dspark_cfg, **dflash_cfg, **top}

    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "target_layer_ids",
    )
    missing = [key for key in required if key not in merged]
    if missing:
        raise ValueError(
            f"--from-released: '{config_path}' is missing required draft "
            f"dimensions: {missing}"
        )

    target_layer_ids = merged["target_layer_ids"]
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ValueError(
            "--from-released: released 'target_layer_ids' must be a non-empty "
            "list of auxiliary hidden-state layer ids"
        )

    return {
        # Decoder dimensions.
        "vocab_size": int(merged["vocab_size"]),
        "hidden_size": int(merged["hidden_size"]),
        "intermediate_size": int(merged["intermediate_size"]),
        "num_hidden_layers": int(merged["num_hidden_layers"]),
        "num_attention_heads": int(merged["num_attention_heads"]),
        "num_key_value_heads": int(merged["num_key_value_heads"]),
        "head_dim": merged.get("head_dim"),
        "hidden_act": merged.get("hidden_act", "silu"),
        "max_position_embeddings": merged.get("max_position_embeddings", 4096),
        "initializer_range": merged.get("initializer_range", 0.02),
        "rms_norm_eps": merged.get("rms_norm_eps", 1e-5),
        "sliding_window": merged.get("sliding_window", 2048),
        "layer_types": list(merged.get("layer_types") or []),
        "rope_theta": merged.get("rope_theta", 10000.0),
        "rope_parameters": merged.get("rope_parameters"),
        "is_causal": merged.get("is_causal", True),
        # Speculator-level fields (dflash_config / dspark_config / top level).
        "draft_vocab_size": int(merged.get("draft_vocab_size", merged["vocab_size"])),
        "block_size": int(merged.get("block_size", 8)),
        "mask_token_id": merged.get("mask_token_id"),
        "target_layer_ids": [int(i) for i in target_layer_ids],
        # DSpark extras.
        "markov_rank": int(merged.get("markov_rank", 256)),
        "markov_head_type": merged.get("markov_head_type", "vanilla"),
        "enable_confidence_head": bool(merged.get("enable_confidence_head", True)),
        "confidence_head_with_markov": bool(
            merged.get("confidence_head_with_markov", True)
        ),
        # DFlash2 extras.
        "conv_kernel_size": int(merged.get("conv_kernel_size", 2)),
        "conv_group_size": int(merged.get("conv_group_size", 16)),
        "selector_rank": int(merged.get("selector_rank", 256)),
        "selector_top_k": int(merged.get("selector_top_k", 16)),
    }


def _released_shard_paths(released_dir: Path) -> list[Path]:
    """Weight files of a released checkpoint (sharded via index, or single)."""
    index_path = released_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return sorted(
            {released_dir / name for name in set(weight_map.values())}
        )
    single = released_dir / "model.safetensors"
    if single.is_file():
        return [single]
    raise ValueError(
        f"--from-released: no model.safetensors (or shard index) under "
        f"'{released_dir}'"
    )


def _shard_key_shape(handle: Any, key: str) -> tuple[int, ...]:
    """Shape of ``key`` inside an open safetensors shard, without loading it."""
    try:
        return tuple(handle.get_slice(key).get_shape())
    except Exception:  # noqa: BLE001 - older safetensors without slice access
        return tuple(handle.get_tensor(key).shape)


def _build_decoder_config(
    desc: dict[str, Any],
    num_layers: int,
    draft_attn_impl: str | None,
) -> Qwen3Config:
    """Build the draft decoder config from the normalised release description.

    Mirrors ``create_transformer_layer_config`` (train/cli.py) for the fields a
    release actually carries: released ``layer_types`` are preserved so full-
    vs-sliding attention layers match the trained model, and rope follows the
    same transformers-version gate as the rest of the CLI.
    """
    layer_types = list(desc["layer_types"] or [])
    if layer_types and len(layer_types) < num_layers:
        raise ValueError(
            f"--from-released: released layer_types has {len(layer_types)} "
            f"entries but {num_layers} draft layers were requested"
        )
    layer_types = layer_types[:num_layers]

    head_dim = desc.get("head_dim") or desc["hidden_size"] // desc["num_attention_heads"]
    config = Qwen3Config(
        vocab_size=desc["vocab_size"],
        hidden_size=desc["hidden_size"],
        intermediate_size=desc["intermediate_size"],
        num_hidden_layers=num_layers,
        num_attention_heads=desc["num_attention_heads"],
        num_key_value_heads=desc["num_key_value_heads"],
        head_dim=head_dim,
        hidden_act=desc.get("hidden_act", "silu"),
        max_position_embeddings=desc.get("max_position_embeddings", 4096),
        initializer_range=desc.get("initializer_range", 0.02),
        rms_norm_eps=desc.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=False,
    )
    if layer_types:
        config.layer_types = layer_types
        config.sliding_window = desc.get("sliding_window", 2048)
        config.use_sliding_window = "sliding_attention" in layer_types

    if _TRANSFORMERS_AT_LEAST_5:
        rope = dict(desc["rope_parameters"]) if desc.get("rope_parameters") else {}
        rope.pop("type", None)
        rope.pop("mrope_interleaved", None)
        rope.setdefault("rope_theta", desc.get("rope_theta", 10000.0))
        rope.setdefault("rope_type", "default")
        config.rope_parameters = rope
    else:
        config.rope_theta = desc.get("rope_theta", 10000.0)

    if draft_attn_impl is not None:
        # Same private-attribute route as the rest of the CLI: HF configs never
        # serialize the attention implementation, so it is re-applied here.
        config._attn_implementation = draft_attn_impl  # noqa: SLF001
    return config


def _load_released_tensors(
    model: torch.nn.Module,
    shards: list[Path],
    built_layers: int,
) -> tuple[dict[str, torch.Tensor], list[str], list[str], list[str], int]:
    """Load released tensors by exact name and shape.

    Returns ``(usable, unknown, layer_reduction, shape_skipped, total)`` where
    ``unknown`` are released keys the model does not define at all (always an
    error), ``layer_reduction`` are keys of draft layers beyond ``built_layers``
    (legitimate for a reduced-layer smoke profile), and ``shape_skipped`` are
    keys that exist on both sides with different shapes (e.g. a full-vocab
    ``markov_w2`` against a compressed draft vocabulary).
    """
    own = model.state_dict()
    usable: dict[str, torch.Tensor] = {}
    unknown: list[str] = []
    layer_reduction: list[str] = []
    shape_skipped: list[str] = []
    total = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                total += 1
                own_tensor = own.get(key)
                if own_tensor is None:
                    match = _LAYER_KEY_RE.match(key)
                    if match and int(match.group(1)) >= built_layers:
                        layer_reduction.append(key)
                    else:
                        unknown.append(key)
                    continue
                if tuple(own_tensor.shape) != _shard_key_shape(handle, key):
                    shape_skipped.append(key)
                    continue
                usable[key] = handle.get_tensor(key)

    result = model.load_state_dict(usable, strict=False)
    if result.unexpected_keys:  # pragma: no cover - filtered by construction
        raise RuntimeError(
            "internal: shape-filtered released load produced unexpected keys "
            f"{result.unexpected_keys[:5]}"
        )
    return usable, unknown, layer_reduction, shape_skipped, total


def _nonfinite_tensors(
    model: torch.nn.Module,
) -> list[tuple[str, torch.Tensor]]:
    """Floating-point parameters/buffers that still hold non-finite values.

    The model classes initialise the verifier-owned tensors (embed/lm_head/
    verifier_lm_head) to NaN precisely so that a missing load fails loudly.
    """
    nonfinite: list[tuple[str, torch.Tensor]] = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            nonfinite.append((name, tensor))
    return nonfinite


def build_draft_from_released(  # noqa: C901
    *,
    speculator_type: str,
    released_dir: str | Path,
    verifier_name_or_path: str | None = None,
    t2d: torch.Tensor | None = None,
    d2t: torch.Tensor | None = None,
    draft_vocab_size: int | None = None,
    num_layers: int | None = None,
    target_layer_ids: list[int] | None = None,
    draft_attn_impl: str | None = None,
    trust_remote_code: bool = False,
    init_unloaded: Literal["error", "random"] = "error",
    init_dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, ReleasedDraftStats]:
    """Build a DSpark/DFlash/DFlash2 draft from a released deploy checkpoint.

    :param speculator_type: ``"dspark"``, ``"dflash"`` or ``"dflash2"``.
    :param released_dir: directory with the flat released ``config.json`` and
        ``model.safetensors`` (or shards plus ``model.safetensors.index.json``).
    :param verifier_name_or_path: verifier used to rebuild the tensors the
        release omits (verifier_lm_head/verifier_norm always; embed/lm_head
        only for style-A releases).  Without it the build fails closed unless
        the release packed every vocab-side tensor or
        ``init_unloaded="random"``.
    :param t2d / d2t: vocabulary mappings, only needed when ``draft_vocab_size``
        is smaller than the verifier vocabulary.
    :param draft_vocab_size: defaults to the release's ``draft_vocab_size``
        (the full verifier vocabulary for both GLM-5.3 releases).
    :param num_layers: defaults to the released layer count; a smaller value
        builds a reduced-layer profile (extra released layers are skipped and
        reported); a larger value is rejected.
    :param target_layer_ids: defaults to the release's auxiliary hidden-state
        layer ids; pass explicit ids only when the training data was rendered
        for different verifier layers.
    :param draft_attn_impl: attention implementation for the draft decoder
        (``"eager"`` on NPU; mirrors ``--draft-attn-impl``).
    :param init_unloaded: ``"error"`` (default) refuses to start when anything
        could not be loaded; ``"random"`` fills the affected tensors with
        ``normal_(std=0.02)`` for memory-bounded smoke runs.
    :param init_dtype: build the model under this default dtype (e.g.
        ``torch.bfloat16``) to halve peak host memory for multi-billion drafts;
        the trainer casts parameters to ``hidden_states_dtype`` regardless.
    :returns: ``(model, stats)``.
    """
    if speculator_type not in RELEASED_DRAFT_SPECULATORS:
        raise ValueError(
            f"--from-released supports speculator types "
            f"{sorted(RELEASED_DRAFT_SPECULATORS)}, got {speculator_type!r}"
        )
    if init_unloaded not in ("error", "random"):
        raise ValueError(
            f"init_unloaded must be 'error' or 'random', got {init_unloaded!r}"
        )

    desc = read_released_draft_config(released_dir)
    released_path = Path(released_dir)
    shards = _released_shard_paths(released_path)

    released_layers = desc["num_hidden_layers"]
    if num_layers is None:
        built_layers = released_layers
    elif num_layers > released_layers:
        raise ValueError(
            f"--from-released: --num-layers={num_layers} exceeds the released "
            f"draft's {released_layers} layers; there are no released tensors "
            "for the extra layers"
        )
    else:
        if num_layers < released_layers:
            logger.warning(
                "--from-released: building %d of the released %d draft layers; "
                "released layers %d..%d are skipped (reduced profile)",
                num_layers,
                released_layers,
                num_layers,
                released_layers - 1,
            )
        built_layers = num_layers

    draft_vocab = (
        desc["draft_vocab_size"] if draft_vocab_size is None else int(draft_vocab_size)
    )
    if speculator_type == "dflash2" and draft_vocab != desc["vocab_size"]:
        raise ValueError(
            "DFlash2 requires the full verifier vocabulary: "
            f"draft_vocab_size={draft_vocab}, released vocab_size="
            f"{desc['vocab_size']}; the candidate selector cannot be pruned"
        )
    if desc["mask_token_id"] is None:
        raise ValueError(
            "--from-released: the released config carries no 'mask_token_id', "
            "which DFlash-family training requires"
        )
    aux_layer_ids = (
        list(desc["target_layer_ids"])
        if target_layer_ids is None
        else [int(i) for i in target_layer_ids]
    )
    if not aux_layer_ids:
        raise ValueError("--from-released: target_layer_ids must not be empty")

    if (
        verifier_name_or_path
        and draft_vocab != desc["vocab_size"]
        and (t2d is None or d2t is None)
    ):
        raise ValueError(
            f"--from-released: draft_vocab_size={draft_vocab} is smaller than "
            f"the verifier vocabulary ({desc['vocab_size']}), so t2d/d2t "
            "mappings are required to rebuild the verifier-side heads"
        )

    decoder_config = _build_decoder_config(desc, built_layers, draft_attn_impl)
    block_size = desc["block_size"]
    # Mirror the config-class defaults the released checkpoints rely on: DSpark
    # samples from the anchor position, DFlash/DFlash2 do not; DFlash2 uses
    # bidirectional masking inside sliding-window draft blocks.
    sample_from_anchor = speculator_type == "dspark"
    sliding_window_non_causal = (
        speculator_type == "dflash2" and not desc["is_causal"]
    )
    speculative_tokens = block_size if sample_from_anchor else block_size - 1

    if verifier_name_or_path:
        verifier_config = VerifierConfig.from_pretrained(
            verifier_name_or_path, trust_remote_code=trust_remote_code
        )
    else:
        verifier_config = VerifierConfig(
            name_or_path=None, architectures=["GlmForConditionalGeneration"]
        )

    common = {
        "transformer_layer_config": decoder_config,
        "draft_vocab_size": draft_vocab,
        "block_size": block_size,
        "aux_hidden_state_layer_ids": aux_layer_ids,
        "mask_token_id": desc["mask_token_id"],
        "sample_from_anchor": sample_from_anchor,
        "sliding_window_non_causal": sliding_window_non_causal,
        "speculators_config": SpeculatorsConfig(
            algorithm=speculator_type,
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
            ],
            default_proposal_method="greedy",
            verifier=verifier_config,
        ),
    }

    previous_dtype = torch.get_default_dtype()
    if init_dtype is not None:
        torch.set_default_dtype(init_dtype)
    try:
        if speculator_type == "dflash2":
            from speculators.models.dflash2.config import DFlash2SpeculatorConfig
            from speculators.models.dflash2.core import DFlash2DraftModel

            model = DFlash2DraftModel(
                DFlash2SpeculatorConfig(
                    **common,
                    conv_kernel_size=desc["conv_kernel_size"],
                    conv_group_size=desc["conv_group_size"],
                    selector_rank=desc["selector_rank"],
                    selector_top_k=desc["selector_top_k"],
                )
            )
        elif speculator_type == "dflash":
            from speculators.models.dflash.config import DFlashSpeculatorConfig
            from speculators.models.dflash.core import DFlashDraftModel

            # Plain DFlash (v1): the common fields already describe it fully
            # (no markov head, no conv/selector knobs; vocabulary mapping via
            # DraftVocabMixin supports both full and pruned draft vocabularies).
            model = DFlashDraftModel(DFlashSpeculatorConfig(**common))
        else:
            from speculators.models.dspark.config import DSparkSpeculatorConfig
            from speculators.models.dspark.core import DSparkDraftModel

            model = DSparkDraftModel(
                DSparkSpeculatorConfig(
                    **common,
                    markov_rank=desc["markov_rank"],
                    markov_head_type=desc["markov_head_type"],
                    enable_confidence_head=desc["enable_confidence_head"],
                    confidence_head_with_markov=desc["confidence_head_with_markov"],
                )
            )
    finally:
        if init_dtype is not None:
            torch.set_default_dtype(previous_dtype)

    # ---- released draft tensors -------------------------------------------
    model.load_vocab_mappings(t2d, d2t)
    usable, unknown, layer_reduction, shape_skipped, total = _load_released_tensors(
        model, shards, built_layers
    )
    if unknown:
        raise ValueError(
            f"--from-released: the release contains tensors our "
            f"{speculator_type} model does not define: {unknown[:8]} "
            f"({len(unknown)} total); refusing to drop released weights "
            "silently"
        )
    if shape_skipped and init_unloaded == "error":
        raise ValueError(
            f"--from-released: released tensors do not fit this configuration "
            f"(name matches, shape differs): {shape_skipped[:8]} "
            f"({len(shape_skipped)} total). Common cause: --draft-vocab-size "
            "smaller than the release's full vocabulary drops full-vocabulary "
            "heads such as markov_w2. Train with the full vocabulary, or pass "
            "init_unloaded='random' for a smoke profile."
        )
    if shape_skipped:
        logger.warning(
            "--from-released: released tensors skipped for shape mismatch and "
            "left at their fresh random init: %s",
            shape_skipped,
        )

    # ---- verifier-owned tensors -------------------------------------------
    # Style-A releases (GLM-5.3 DSpark/DFlash2) omit every vocab-side tensor;
    # packed style-B releases (e.g. UCloud GLM-5.2-FP8-DFlash) carry the
    # target's frozen embed/lm_head, which just loaded like any other released
    # tensor.  The verifier is the source of truth for training, so its
    # rebuild runs last and overwrites whatever the release packed; the
    # signatures are compared so a genuine release/verifier mismatch (e.g. an
    # FP8 target vs a bf16 verifier) is *reported* rather than silently kept.
    state = model.state_dict()
    packed = sorted(key for key in usable if key in _VERIFIER_OWNED_KEYS)
    packed_sigs = {
        key: state[key].sum(dtype=torch.float64).item() for key in packed
    }
    if verifier_name_or_path:
        # Uniform training precedence: even a packed release's embed/lm_head
        # (frozen copies of the release's own target) are replaced by *this*
        # run's verifier, which produces the training targets and hidden
        # states.  DFlash's own load_verifier_weights override already does
        # exactly this; the overwrite flags are the base class's documented
        # internal-caller escape hatch, applied here for every released kind
        # so dspark/dflash2 cannot silently keep a stale packed copy (the
        # values are bit-identical for a matched release/verifier pair).
        model._load_verifier_weights(  # noqa: SLF001 - internal caller
            overwrite_embed_tokens=True,
            overwrite_lm_head=True,
        )
    else:
        model.load_verifier_weights()  # no-op without a verifier
    if packed and verifier_name_or_path:
        state = model.state_dict()
        changed = [
            key
            for key, signature in packed_sigs.items()
            if state[key].sum(dtype=torch.float64).item() != signature
        ]
        if changed:
            logger.warning(
                "--from-released: the release packed vocab-side tensors that "
                "differ from the verifier's: %s. The verifier rebuild wins "
                "(training targets are produced by the verifier); expected "
                "when the release was built against a different target "
                "variant (e.g. FP8 vs bf16).",
                changed,
            )
    if packed:
        logger.info(
            "[from-released] release packed vocab-side tensors %s; verifier "
            "rebuild: %s",
            packed,
            "overwrites them"
            if verifier_name_or_path
            else "not run (no verifier) -- kept from the release",
        )
    random_filled: list[str] = []
    nonfinite = _nonfinite_tensors(model)
    if nonfinite:
        names = [name for name, _ in nonfinite]
        if init_unloaded == "error":
            provided = (
                ", ".join(packed) if packed else "none of them"
            )
            raise ValueError(
                f"--from-released: {len(names)} verifier-owned tensors are "
                f"still uninitialized (NaN): {names[:8]}. The release "
                f"provides {provided}; the remaining ones are rebuilt from "
                "the verifier, so pass --verifier-name-or-path (e.g. "
                "/mnt/cephfs/weight/GLM-5.3)."
            )
        with torch.no_grad():
            for name, tensor in nonfinite:
                torch.nn.init.normal_(tensor, std=0.02)
                random_filled.append(name)
        logger.warning(
            "--from-released: filled %d uninitialized tensors with "
            "normal_(std=0.02): %s",
            len(random_filled),
            random_filled,
        )

    stats = ReleasedDraftStats(
        released_tensors=total,
        loaded_tensors=len(usable),
        layer_reduction_skipped=tuple(sorted(layer_reduction)),
        shape_skipped=tuple(sorted(shape_skipped)),
        random_filled=tuple(sorted(random_filled)),
        release_vocab_side=tuple(packed),
    )
    logger.info(
        "[from-released] %s: loaded %d/%d released tensors from %s "
        "(layers %d/%d, draft_vocab %d, aux layers %s); verifier rebuild: %s",
        speculator_type,
        stats.loaded_tensors,
        stats.released_tensors,
        released_path,
        built_layers,
        released_layers,
        draft_vocab,
        aux_layer_ids,
        verifier_name_or_path or "<not provided>",
    )
    return model, stats
