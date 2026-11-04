"""MTP checkpoint converter.

Extracts only the MTP layer weights from a checkpoint with native MTP
layers and saves a Speculators checkpoint that loads with
``MTPDraftModel.from_pretrained(path)``.

Only the ``mtp.*`` subtree is extracted from the (potentially sharded)
safetensors file; the rest of the model is never loaded.  The embed_tokens
and lm_head are loaded from the verifier at runtime via
``load_verifier_weights()``.
"""

import json
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.mtp.glm_layout import (
    GLM_MODEL_TYPE,
    GlmMtpPolicy,
    build_glm_key_map,
    build_glm_layout,
    build_glm_native_manifest,
    derive_glm_policy,
    detect_glm_native_prefix,
    drop_glm_optional_norm_bias,
    verify_glm_layer_index,
    verify_glm_policy,
)
from speculators.convert.mtp.glm_quant import fold_w8a8_siblings
from speculators.convert.mtp.layout import NativeLayout
from speculators.convert.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
)
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import SAFETENSORS_INDEX_NAMES, list_checkpoint_keys

__all__ = ["GlmMtpPolicy", "MTPConverter", "NativeLayout", "build_glm_layout"]

_MTP_PREFIX = "mtp."

MTP_EXACT_REMAP: dict[str, str] = {
    "mtp.fc.weight": "mtp_layers.0.input_proj.weight",
    "mtp.norm.weight": "mtp_layers.0.final_layernorm.weight",
}

MTP_PREFIX_REMAP: list[tuple[str, str]] = [
    ("mtp.pre_fc_norm_hidden.", "mtp_layers.0.hidden_layernorm."),
    ("mtp.pre_fc_norm_embedding.", "mtp_layers.0.token_layernorm."),
    ("mtp.layers.0.", "mtp_layers.0."),
]

_EXPERT_PATTERN = re.compile(
    r"^(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _build_qwen_layout() -> NativeLayout:
    return NativeLayout(
        name="mtp",
        native_prefixes=(_MTP_PREFIX,),
        exact_remap=dict(MTP_EXACT_REMAP),
        prefix_remap=tuple(MTP_PREFIX_REMAP),
    )


class MTPConverter:
    """Extract the MTP head from a checkpoint with native MTP layers.

    Reads only the MTP layer, embed_tokens, and lm_head from the source
    checkpoint.  Sharded safetensors files are handled transparently via
    the weight index -- the main transformer stack is never loaded.

    :param layout: Optional explicit :class:`NativeLayout`.  When omitted the
        layout is detected from the native keys (``mtp.`` or GLM's
        ``model.layers.<N>.`` MTP head).
    :param policy: Optional GLM head/embed/rot policy from the asset probe.
    :param index_name: Optional explicit safetensors index filename.
    """

    def __init__(
        self,
        layout: NativeLayout | None = None,
        policy: GlmMtpPolicy | None = None,
        index_name: str | None = None,
    ) -> None:
        self._layout = layout
        self._policy = policy
        self._index_name = index_name
        self._source_config: dict | None = None
        self._index_name_used: str | None = None
        self._source_keys_used: list[str] = []
        self._source_keys_unused: list[str] = []
        self._dtypes: set[torch.dtype] = set()
        self._last_glm_policy: GlmMtpPolicy | None = None
        self._resolved_policy: GlmMtpPolicy | None = None
        self.last_key_map: dict | None = None
        self.last_manifest: dict | None = None

    def convert_to_state_dict(
        self,
        input_path: str | Path,
        cache_dir: str | Path | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract native MTP weights and return them as a state dict.

        Performs the full pipeline (download/locate → verify → extract →
        remap → fuse MoE experts) without writing anything to disk.
        """
        logger.info(f"Extracting native MTP weights from {input_path}")

        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        all_keys = list_checkpoint_keys(local_path, index_name=self._index_name)
        layout = self._resolve_layout(all_keys, local_path)
        self._verify_mtp_format(all_keys, layout)

        glm_policy: GlmMtpPolicy | None = None
        if layout.name == GLM_MODEL_TYPE:
            self._source_config = load_checkpoint_config(local_path)
            glm_policy = self._effective_glm_policy(layout)
            self._last_glm_policy = glm_policy
            if glm_policy.is_rot_used is None:
                logger.warning(
                    "rot_gate_consistent is null: the quant metadata did not "
                    "provide is_rot_used. Provide it via GlmMtpPolicy / "
                    "--is-rot-used once the asset probe has run; Phase 3/7 "
                    "gates require a concrete value."
                )
            verify_glm_layer_index(layout.native_prefixes[0], self._source_config)
            verify_glm_policy(
                glm_policy,
                all_keys,
                layout.native_prefixes[0],
            )

        self._dtypes = set()
        weights = self._extract_weights(local_path, all_keys, layout)
        # W8A8_DYNAMIC packages (customer w8a8c8): materialize int8 payloads
        # that carry a weight_scale sibling; every remaining int8 tensor is a
        # format error.
        fold_w8a8_siblings(weights)
        leftover_int8 = sorted(
            key for key, tensor in weights.items() if tensor.dtype == torch.int8
        )
        if leftover_int8:
            raise ValueError(
                "Int8 weights without a weight_scale sibling cannot be "
                f"materialized: {leftover_int8[:5]}"
            )
        if layout.name == GLM_MODEL_TYPE:
            # Quant packs may attach an all-zero RMSNorm bias; it has no
            # training-side parameter, so drop it before fusion/save.
            dropped = drop_glm_optional_norm_bias(weights, layout)
            if dropped:
                dropped_set = set(dropped)
                self._source_keys_used = [
                    key for key in self._source_keys_used if key not in dropped_set
                ]
                self._source_keys_unused = sorted({*self._source_keys_unused, *dropped})
        weights = self._fuse_moe_experts(weights, layout)
        self._layout = layout

        if layout.name == GLM_MODEL_TYPE:
            self.last_key_map = build_glm_key_map(layout, all_keys)
            if self._source_config is not None and glm_policy is not None:
                self.last_manifest = build_glm_native_manifest(
                    self._source_config,
                    layout,
                    glm_policy,
                    self._index_name_used or "model.safetensors",
                    self._source_keys_used,
                    self._source_keys_unused,
                    self._dtypes,
                    target_key_count=len(weights),
                )

        logger.info(f"Extracted {len(weights)} MTP weight tensors")
        return weights

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        base_model: str,
        num_speculative_steps: int = 3,
        validate: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        logger.info(f"Converting MTP checkpoint: {input_path}")

        weights = self.convert_to_state_dict(input_path, cache_dir)

        local_path = ensure_checkpoint_is_local(input_path, cache_dir)
        source_config = self._source_config or load_checkpoint_config(local_path)

        if "text_config" in source_config:
            source_config = source_config["text_config"]

        config = self._build_config(source_config, base_model, num_speculative_steps)
        saved_path = self._save(config, weights, output_path)
        self._write_glm_reports(saved_path)
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate(saved_path)

    def _resolve_layout(
        self, all_keys: list[str], checkpoint_dir: Path | None = None
    ) -> NativeLayout:
        if self._layout is not None:
            return self._layout
        if any(key.startswith(_MTP_PREFIX) for key in all_keys):
            return _build_qwen_layout()
        try:
            prefix = detect_glm_native_prefix(all_keys)
        except ValueError as exc:
            raise ValueError(
                "No native MTP keys found. Expected keys starting with "
                "'mtp.' (Qwen) or 'model.layers.<N>.eh_proj.weight' / "
                "'model.layers.<N>.shared_head.norm.weight' (GLM)."
            ) from exc
        policy = self._policy
        if policy is None:
            # Derive the per-asset facts (own head/embed, rot sidecar, quant
            # is_rot_used) from the package itself -- single source of truth.
            policy = derive_glm_policy(all_keys, prefix, checkpoint_dir)
            self._resolved_policy = policy
        return build_glm_layout(prefix, policy)

    def _effective_glm_policy(self, layout: NativeLayout) -> GlmMtpPolicy:
        """Return the GLM policy, validated against the explicit layout.

        An explicit layout built with a different policy than the converter's
        would extract head/embed weights that ``MTPDraftModel`` then drops at
        save time (capability ignore list), so the two must agree.
        """
        prefix = layout.native_prefixes[0]
        policy = (
            self._policy
            or getattr(self, "_resolved_policy", None)
            or GlmMtpPolicy()
        )
        mapped_head = f"{prefix}shared_head.head.weight" in layout.exact_remap
        mapped_embed = f"{prefix}embed_tokens.weight" in layout.exact_remap
        if policy.own_mtp_head != mapped_head or policy.own_mtp_embed != mapped_embed:
            raise ValueError(
                "GlmMtpPolicy disagrees with the GLM layout's head/embed "
                f"mapping (policy: own_mtp_head={policy.own_mtp_head}, "
                f"own_mtp_embed={policy.own_mtp_embed}; layout maps head="
                f"{mapped_head}, embed={mapped_embed}). Build the layout "
                "with the same policy, or pass the matching policy to "
                "MTPConverter."
            )
        return policy

    def _verify_mtp_format(
        self, keys: list[str], layout: NativeLayout | None = None
    ) -> None:
        layout = layout or self._resolve_layout(keys)
        if not any(layout.is_native_key(key) for key in keys):
            raise ValueError(
                f"No keys with native prefixes {layout.native_prefixes} found. "
                "This converter requires checkpoints with native MTP layers "
                "(e.g. Qwen3-Next, Qwen3.5, GLM-MoE-DSA)."
            )

    def _find_weights_file(
        self, checkpoint_dir: Path
    ) -> tuple[Path | None, Path | None]:
        single = checkpoint_dir / "model.safetensors"
        if self._index_name is not None:
            # Explicit index name: never fall back to a different index.
            index_path = checkpoint_dir / self._index_name
            if index_path.exists():
                return index_path, None
        else:
            for name in SAFETENSORS_INDEX_NAMES:
                index_path = checkpoint_dir / name
                if index_path.exists():
                    return index_path, None

        if single.exists():
            return None, single

        expected = self._index_name or ", ".join(SAFETENSORS_INDEX_NAMES)
        raise FileNotFoundError(
            f"No safetensors checkpoint found at {checkpoint_dir}. "
            f"Expected {expected} or model.safetensors."
        )

    def _check_tensor(self, key: str, tensor: torch.Tensor) -> None:
        self._dtypes.add(tensor.dtype)
        if "float8" in str(tensor.dtype):
            raise NotImplementedError(
                f"Native key '{key}' is quantized ({tensor.dtype}). "
                "Dequantizing fp8/w4a8 MTP heads is not implemented yet "
                "(plan section 17.5); convert a BF16/FP16 MTP source or add "
                "a dequant step first."
            )

    def _write_glm_reports(self, output_path: Path) -> None:
        if self.last_key_map is None:
            return
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "glm_mtp_key_map.json").open("w") as f:
            json.dump(self.last_key_map, f, indent=2, sort_keys=True)
        if self.last_manifest is not None:
            with (output_path / "glm_native_manifest.json").open("w") as f:
                json.dump(self.last_manifest, f, indent=2, sort_keys=True)

    @staticmethod
    def _remap_key(key: str) -> str:
        return _build_qwen_layout().remap_key(key)

    @staticmethod
    def _fuse_moe_experts(
        weights: dict[str, torch.Tensor],
        layout: NativeLayout | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse individual expert weights into packed tensors.

        MoE checkpoints (e.g. Qwen3-Next) store per-expert weights as
        separate keys (``experts.{N}.gate_proj``), but the transformers
        model uses fused 3D tensors (``experts.gate_up_proj``).
        """

        def native_key(target: str) -> str:
            return layout.inverse_remap_key(target) if layout is not None else target

        groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        non_expert: dict[str, torch.Tensor] = {}

        for key, tensor in weights.items():
            m = _EXPERT_PATTERN.match(key)
            if m:
                prefix = m.group(1)
                idx = int(m.group(2))
                proj = m.group(3)
                groups.setdefault(prefix, {}).setdefault(idx, {})[proj] = tensor
            else:
                non_expert[key] = tensor

        if not groups:
            return weights

        for prefix, experts_by_idx in groups.items():
            num_experts = max(experts_by_idx.keys()) + 1
            expected = set(range(num_experts))
            if set(experts_by_idx.keys()) != expected:
                raise ValueError(
                    f"Non-contiguous expert indices at target prefix "
                    f"'{prefix}' (native example key "
                    f"'{native_key(f'{prefix}.0.gate_proj.weight')}'): "
                    f"found {sorted(experts_by_idx.keys())}, "
                    f"expected 0..{num_experts - 1}"
                )

            for i in range(num_experts):
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    if proj_name not in experts_by_idx[i]:
                        raise ValueError(
                            f"Expert {i} at target prefix '{prefix}' missing "
                            f"{proj_name} weight (native key "
                            f"'{native_key(f'{prefix}.{i}.{proj_name}.weight')}')"
                        )

            gate_list = [experts_by_idx[i]["gate_proj"] for i in range(num_experts)]
            up_list = [experts_by_idx[i]["up_proj"] for i in range(num_experts)]
            down_list = [experts_by_idx[i]["down_proj"] for i in range(num_experts)]

            gate_up = torch.stack(
                [
                    torch.cat([g, u], dim=0)
                    for g, u in zip(gate_list, up_list, strict=True)
                ]
            )
            down = torch.stack(down_list)

            non_expert[f"{prefix}.gate_up_proj"] = gate_up
            non_expert[f"{prefix}.down_proj"] = down
            logger.debug(
                f"Fused {num_experts} experts at {prefix}: "
                f"gate_up_proj={gate_up.shape}, "
                f"down_proj={down.shape}"
            )

        return non_expert

    def _extract_weights(
        self, checkpoint_dir: Path, all_keys: list[str], layout: NativeLayout
    ) -> dict[str, torch.Tensor]:
        native_keys = [key for key in all_keys if layout.is_native_key(key)]
        needed = {key for key in native_keys if not layout.is_ignored(key)}
        # W8A8 packages: pull the per-channel scale/offset siblings of every
        # needed int8 weight so ``fold_w8a8_siblings`` can materialize them.
        key_set = set(all_keys)
        for key in sorted(needed):
            if not key.endswith(".weight"):
                continue
            for suffix in ("_scale", "_offset"):
                sibling = f"{key}{suffix}"
                if sibling in key_set:
                    needed.add(sibling)
        self._source_keys_used = sorted(needed)
        self._source_keys_unused = sorted(set(native_keys) - needed)

        index_path, single = self._find_weights_file(checkpoint_dir)
        if index_path is not None:
            self._index_name_used = index_path.name
            return self._extract_from_shards(checkpoint_dir, index_path, needed, layout)

        self._index_name_used = "model.safetensors"
        if single is None:  # pragma: no cover - guarded by _find_weights_file
            raise FileNotFoundError(f"No safetensors checkpoint at {checkpoint_dir}")
        return self._extract_single(single, needed, layout)

    def _extract_from_shards(
        self,
        checkpoint_dir: Path,
        index_path: Path,
        needed_keys: set[str],
        layout: NativeLayout,
    ) -> dict[str, torch.Tensor]:
        with index_path.open() as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]

        missing = sorted(needed_keys - set(weight_map))
        if missing:
            raise ValueError(f"Missing native MTP keys in {index_path.name}: {missing}")

        shard_to_keys: dict[str, list[str]] = {}
        for key in needed_keys:
            shard = weight_map[key]
            shard_to_keys.setdefault(shard, []).append(key)

        weights: dict[str, torch.Tensor] = {}
        source_by_target: dict[str, str] = {}
        for shard_filename, keys in shard_to_keys.items():
            shard_path = checkpoint_dir / shard_filename
            logger.debug(f"Reading {len(keys)} key(s) from shard {shard_filename}")
            with safe_open(str(shard_path), framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    self._check_tensor(key, tensor)
                    target = layout.remap_key(key)
                    if target in weights:
                        raise ValueError(
                            f"Native keys '{source_by_target[target]}' and "
                            f"'{key}' both map to '{target}'; source keys "
                            "must not be consumed twice."
                        )
                    source_by_target[target] = key
                    weights[target] = tensor

        return weights

    def _extract_single(
        self,
        safetensors_path: Path,
        needed_keys: set[str],
        layout: NativeLayout,
    ) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = {}
        source_by_target: dict[str, str] = {}
        with safe_open(str(safetensors_path), framework="pt") as f:
            present = set(f.keys())
            missing = sorted(needed_keys - present)
            if missing:
                raise ValueError(
                    f"Missing native MTP keys in {safetensors_path.name}: {missing}"
                )
            for key in sorted(needed_keys):
                tensor = f.get_tensor(key)
                self._check_tensor(key, tensor)
                target = layout.remap_key(key)
                if target in weights:
                    raise ValueError(
                        f"Native keys '{source_by_target[target]}' and '{key}' "
                        f"both map to '{target}'; source keys must not be "
                        "consumed twice."
                    )
                source_by_target[target] = key
                weights[target] = tensor
        return weights

    def _build_config(
        self,
        source_config: dict,
        base_model: str,
        num_speculative_steps: int,
    ) -> MTPSpeculatorConfig:
        verifier_config_dict, _ = PretrainedConfig.get_config_dict(base_model)

        source_hidden = source_config.get("hidden_size")
        target_hidden = verifier_config_dict.get("hidden_size")
        if source_hidden and target_hidden and source_hidden != target_hidden:
            raise ValueError(
                f"Architecture mismatch: source MTP checkpoint has "
                f"hidden_size={source_hidden} but base_model '{base_model}' "
                f"has hidden_size={target_hidden}. Dimensions must match."
            )

        speculators_config = SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=num_speculative_steps)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=base_model,
                architectures=verifier_config_dict.get("architectures", []),
            ),
        )
        return MTPSpeculatorConfig(
            transformer_layer_config=source_config,  # type: ignore[arg-type]
            speculators_config=speculators_config,
            mtp_own_head=(
                self._last_glm_policy.own_mtp_head
                if self._last_glm_policy is not None
                else False
            ),
            mtp_own_embed=(
                self._last_glm_policy.own_mtp_embed
                if self._last_glm_policy is not None
                else False
            ),
            mtp_has_rot=(
                self._last_glm_policy.has_rot_weight
                if self._last_glm_policy is not None
                else False
            ),
        )

    def _save(
        self,
        config: MTPSpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        model = MTPDraftModel(config)
        expected_shapes = model.state_dict()
        for key, tensor in weights.items():
            if key in expected_shapes and expected_shapes[key].shape != tensor.shape:
                native_key = (
                    self._layout.inverse_remap_key(key)
                    if self._layout is not None
                    else key
                )
                raise ValueError(
                    f"Shape mismatch for native key '{native_key}' ({key}): "
                    f"expected {tuple(expected_shapes[key].shape)}, "
                    f"got {tuple(tensor.shape)}"
                )

        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(
                "Unexpected keys in extracted weights -- the "
                "checkpoint structure does not match the model "
                "architecture. This may indicate a wrong layer_idx "
                "(full_attention vs linear_attention). "
                f"Unexpected keys: {unexpected}"
            )
        if missing:
            critical_missing = [k for k in missing if k.startswith("mtp_layers.")]
            if critical_missing:
                raise ValueError(
                    "Critical MTP layer weights missing after extraction. "
                    "The checkpoint may be incompatible or use an "
                    "unsupported format. "
                    f"Missing keys: {critical_missing}"
                )
            logger.debug(
                f"Keys not in extracted weights (loaded from verifier at runtime): "
                f"{missing}"
            )

        float_dtypes = {t.dtype for t in weights.values() if t.is_floating_point()}
        if len(float_dtypes) == 1:
            model.to(dtype=next(iter(float_dtypes)))  # type: ignore[call-arg]
        elif len(float_dtypes) > 1:
            # Mixed per-tensor dtypes (e.g. fp8 unquantized norms + quantized
            # experts, or bf16 weights + fp32 router buffers).  A single model
            # dtype cannot represent that, so keep the fp32 training master
            # (plan section 17.5); stitch casts each tensor back to the
            # verifier's per-key dtype.
            logger.warning(
                f"Mixed float dtypes in checkpoint: {sorted(map(str, float_dtypes))}. "
                "Keeping the fp32 training master."
            )

        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        logger.info("Validating converted checkpoint...")
        try:
            MTPDraftModel.from_pretrained(str(output_path))
            logger.success("Validation succeeded")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error(f"Validation failed: {exc}")
            raise
