import logging
import os
from typing import ClassVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.dflash import DFlashSpeculatorConfig  # [DEEP_PROFILE]
from speculators.models.dflash import deep_profile as dp
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.utils import (
    conditional_torch_compile,
    flatten_rope_parameters,
    resolve_target_layer_ids,
    resolve_verifier_norm_class,
)


def _sp_state() -> tuple[int, object | None, int]:
    """Return ``(sp_size, sp_group, sp_rank)``; ``(1, None, 0)`` outside training.

    The training package is imported lazily: the models package must stay
    importable without it (inference-only installs).
    """
    try:
        from speculators.train.distributed import (  # noqa: PLC0415
            get_sp_group,
            get_sp_rank,
            get_sp_size,
        )
    except Exception:  # pragma: no cover - train extras not installed
        return 1, None, 0
    return get_sp_size(), get_sp_group(), get_sp_rank()


def _global_rank() -> int:
    """Global rank for error messages; 0 when distributed is not initialised."""
    try:
        import torch.distributed as dist  # noqa: PLC0415

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        from speculators.train.distributed import get_rank  # noqa: PLC0415

        return get_rank()
    except Exception:  # pragma: no cover - train extras not installed
        return 0

logger = logging.getLogger(__name__)

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
if os.environ.get("SPECULATORS_DISABLE_COMPILE", "0") == "1":
    _compiled_create_block_mask = create_block_mask
else:
    _compiled_create_block_mask = torch.compile(create_block_mask)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    supports_gradient_checkpointing = True  # noqa: D003  # Qwen3DFlashDecoderLayer inherits GradientCheckpointingLayer
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    # DFlash-family drafts shard the sequence *inside* the model: the trainer
    # keeps the full batch because anchor blocks may straddle SP boundaries and
    # the verifier target gather needs the full verifier hidden states.
    _sp_splits_batch: ClassVar[bool] = False
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        # verifier_lm_head is reloaded from the verifier (see load_verifier_weights)
        # and excluded on save, so it is expected to be absent from checkpoints.
        # lm_head is handled per-instance in __init__: omitted only for full-vocab
        # drafts, where it is the exact frozen verifier projection.
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> nn.Module:
        """Build one draft decoder layer.

        DFlash-family variants override this factory when they wrap the shared
        attention and MLP with additional modules.
        """
        return Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            _compiled_create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config
        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                self._make_decoder_layer(config, layer_idx)
                for layer_idx in range(num_draft_layers)
            ]
        )
        # Both are optional: declared only for alternating/sliding attention.
        self.sliding_window = getattr(tl_config, "sliding_window", None)
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(
                getattr(tl_config, "layer_types", None) or []
            )
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        rotary_config = flatten_rope_parameters(config.transformer_layer_config)
        self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = resolve_verifier_norm_class(config)(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size

        # Warn if using DFlash with sample_from_anchor=True (may not be supported)
        if type(self).__name__ == "DFlashDraftModel" and config.sample_from_anchor:
            logger.warning(
                "DFlash with sample_from_anchor=True may not be supported in "
                "all inference engines (e.g., vLLM). Verify compatibility with your "
                "deployment target."
            )

        # Sequence parallelism: swap every attention module onto the SDPA/Ulysses
        # kernel.  This must happen after the process groups exist (the kernel
        # reads the SP topology), which is why the CLI initialises distributed
        # *before* building the model; the forward-time guard in
        # ``_backbone_forward`` catches a wrong construction order instead of
        # silently producing context-free attention.
        self._anchor_override: tuple[torch.Tensor, torch.Tensor] | None = None
        sp_size, _sp_group, _sp_rank = _sp_state()
        if sp_size > 1:
            for layer in self.layers:
                layer.self_attn._attn_impl_override = "dflash_sdpa_sp_attention"

        self.post_init()
        # Verifier-owned weights are reconstructed on load. Keep a reduced-vocab
        # lm_head serialized because current runtimes cannot derive it from the
        # full verifier head.
        #
        # Shadow the ClassVar lists with per-instance copies so full- and
        # reduced-vocabulary siblings cannot mutate each other's save rules.
        keys_to_ignore_on_save = list(type(self)._keys_to_ignore_on_save)  # noqa: SLF001
        keys_to_ignore_on_load_missing = list(
            type(self)._keys_to_ignore_on_load_missing  # noqa: SLF001
        )
        keys_to_ignore_on_save.append("embed_tokens.weight")
        if not self.use_draft_vocab:
            keys_to_ignore_on_save.append("lm_head.weight")
            keys_to_ignore_on_load_missing.append("lm_head.weight")
        self.__dict__["_keys_to_ignore_on_save"] = keys_to_ignore_on_save
        self.__dict__["_keys_to_ignore_on_load_missing"] = (
            keys_to_ignore_on_load_missing
        )

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    # ------------------------------------------------------------------
    # Test-only anchor replay (L2.5 equivalence tests)
    # ------------------------------------------------------------------

    def set_anchor_override(
        self,
        anchor_positions: torch.Tensor,
        anchor_valid: torch.Tensor | None = None,
    ) -> None:
        """Pin ``select_anchors`` to a fixed result. Test-only.

        ``anchor_valid`` controls ``aligned_loss_mask``, so replaying positions
        alone would leave the loss denominator free to drift between the
        single-process reference and the SP replay.
        """
        if anchor_positions.dim() != 1:
            raise ValueError(
                f"anchor_positions must be 1-D, got shape "
                f"{tuple(anchor_positions.shape)}"
            )
        if anchor_positions.dtype != torch.long:
            raise ValueError(
                f"anchor_positions must be torch.long, got {anchor_positions.dtype}"
            )
        if anchor_positions.numel() == 0:
            raise ValueError("anchor_positions must not be empty")
        if anchor_valid is None:
            anchor_valid = torch.ones_like(anchor_positions, dtype=torch.bool)
        if anchor_valid.shape != anchor_positions.shape:
            raise ValueError(
                f"anchor_valid shape {tuple(anchor_valid.shape)} does not match "
                f"anchor_positions shape {tuple(anchor_positions.shape)}"
            )
        if anchor_valid.dtype != torch.bool:
            raise ValueError(f"anchor_valid must be torch.bool, got {anchor_valid.dtype}")
        # Only *valid* anchors are ordered/range-checked: ``select_anchors`` pads
        # the tail with zeros, so invalid slots may sit anywhere.
        valid_positions = anchor_positions[anchor_valid]
        if valid_positions.numel() > 0:
            if torch.any(valid_positions[1:] < valid_positions[:-1]):
                raise ValueError("valid anchor_positions must be sorted ascending")
            if int(valid_positions.min()) < 0:
                raise ValueError(
                    "valid anchor_positions must be non-negative, got "
                    f"{int(valid_positions.min())}"
                )
        self._anchor_override = (anchor_positions.clone(), anchor_valid.clone())

    def clear_anchor_override(self) -> None:
        """Drop a pinned anchor result and go back to random sampling."""
        self._anchor_override = None

    def load_verifier_weights(self):
        """Reconstruct weights intentionally omitted from DFlash checkpoints."""
        self._load_verifier_weights(
            overwrite_embed_tokens=True,
            overwrite_lm_head=not self.use_draft_vocab,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        config = DFlashSpeculatorConfig(
            **cls._build_base_config_kwargs("dflash", verifier_config, **kwargs)
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _build_base_config_kwargs(
        algorithm: str,
        verifier_config: "PretrainedConfig",
        **kwargs,
    ) -> dict:
        """Shared DFlash-family config kwargs for ``from_training_args``.

        DSpark reuses this and appends its Markov/confidence/loss fields.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"),
            kwargs["verifier_name_or_path"],
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        if kwargs.get("causal_attention") is not None:
            verifier_config.is_causal = kwargs["causal_attention"]
        block_size = kwargs.get("block_size", 8)

        default_sample_from_anchor = algorithm == "dspark"
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        sample_from_anchor = (
            default_sample_from_anchor
            if sample_from_anchor_arg is None
            else sample_from_anchor_arg
        )
        default_non_causal = algorithm == "dflash2"
        non_causal_arg = kwargs.get("sliding_window_non_causal")
        sliding_window_non_causal = (
            default_non_causal if non_causal_arg is None else non_causal_arg
        )

        # Calculate speculative tokens based on sample_from_anchor
        # False: anchor is bonus token (block_size - 1 tokens)
        # True: sample from anchor too (block_size tokens)
        speculative_tokens = block_size if sample_from_anchor else block_size - 1

        return {
            "transformer_layer_config": verifier_config,
            "draft_vocab_size": kwargs["draft_vocab_size"],
            "block_size": block_size,
            "aux_hidden_state_layer_ids": target_layer_ids,
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": sliding_window_non_causal,
            "sample_from_anchor": sample_from_anchor,
            "speculators_config": SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        }

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], kwargs.get("loss_implementation", "fused")
        )
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 512)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
    ):
        """Honor explicit causality, preserving legacy per-layer defaults otherwise."""
        is_causal = getattr(self.config.transformer_layer_config, "is_causal", None)
        if is_causal is None:
            is_causal = (
                sliding_window is not None and not self.config.sliding_window_non_causal
            )
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        total_seq_len = loss_mask.shape[1]
        sp_size, sp_group, sp_rank = _sp_state()

        # Anchor selection must be identical on every SP rank: a per-rank
        # ``select_anchors`` would sample different positions and silently train
        # on mismatched targets.  Rank 0 of the SP group samples and broadcasts
        # both tensors; the masks below are then built from the *global* anchors
        # on every rank (the SP attention needs the global grid).
        if self._anchor_override is not None:
            anchor_positions, anchor_valid = self._anchor_override
            if anchor_positions.numel() != max_anchors:
                raise ValueError(
                    f"anchor override has {anchor_positions.numel()} positions but "
                    f"max_anchors={max_anchors}"
                )
            anchor_positions = anchor_positions.to(device=device)
            anchor_valid = anchor_valid.to(device=device)
        elif sp_size > 1:
            if sp_rank == 0:
                anchor_positions, anchor_valid = select_anchors(
                    loss_mask, max_anchors, self.block_size
                )
            else:
                anchor_positions = torch.zeros(
                    max_anchors, dtype=torch.long, device=device
                )
                anchor_valid = torch.zeros(
                    max_anchors, dtype=torch.bool, device=device
                )
        else:
            anchor_positions, anchor_valid = select_anchors(
                loss_mask, max_anchors, self.block_size
            )

        if sp_size > 1:
            if max_anchors % sp_size != 0:
                raise ValueError(
                    f"max_anchors ({max_anchors}) must be divisible by sp_size "
                    f"({sp_size})"
                )
            src = dist.get_global_rank(sp_group, 0)  # type: ignore[arg-type]
            dist.broadcast(anchor_positions, src=src, group=sp_group)  # type: ignore[arg-type]
            dist.broadcast(anchor_valid, src=src, group=sp_group)  # type: ignore[arg-type]

        # Every entry (valid or padding) is later indexed into input_ids /
        # position_ids / verifier hidden states, so even the invalid padding
        # slots must lie in [0, total_seq_len).  The sampled path satisfies this
        # by construction; the check is defensive for ``set_anchor_override``.
        if anchor_positions.numel() > 0 and (
            int(anchor_positions.min()) < 0
            or int(anchor_positions.max()) >= total_seq_len
        ):
            raise ValueError(
                f"anchor_positions out of range [{int(anchor_positions.min())}, "
                f"{int(anchor_positions.max())}) for total_seq_len={total_seq_len}"
            )

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        **kwargs,
    ):
        """Run the anchored-block draft transformer up to the draft logits.

        Returns ``(hidden, logits, targets, aligned_loss_mask,
        anchored_block_indices)``. DSpark reuses this and adds its Markov and
        confidence heads before computing its own loss.
        """
        dp.begin("forward")  # [DEEP_PROFILE]
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 512)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        dp.begin("build_mask")  # [DEEP_PROFILE]
        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )
        dp.end("build_mask")

        sp_size, _sp_group, sp_rank = _sp_state()
        if sp_size > 1 and total_seq_len % sp_size != 0:
            # CLI validates the *configured* total_seq_len at startup, but the
            # actual packed/padded batch length can differ; without this check
            # the flex path fails on the mask-shape assert while the dense-mask
            # path would silently attend across misaligned segments.
            configured = getattr(self, "_configured_seq_len", None)
            configured_str = "unknown" if configured is None else str(configured)
            raise ValueError(
                f"actual_seq_len={total_seq_len} is not divisible by "
                f"sp_size={sp_size}; configured_seq_len={configured_str}, "
                f"rank={_global_rank()}, sp_rank={sp_rank}. Packed batches must "
                "be padded to a multiple of sp_size."
            )
        if sp_size > 1:
            missing = [
                idx
                for idx, layer in enumerate(self.layers)
                if getattr(layer.self_attn, "_attn_impl_override", None)
                != "dflash_sdpa_sp_attention"
            ]
            if missing:
                raise RuntimeError(
                    f"sp_size={sp_size} but draft layers {missing} are not on the "
                    "SP attention kernel: the model was built before "
                    "maybe_setup_distributed() created the SP process group"
                )
        if sp_size > 1:
            # Local view: this rank owns anchor rows [a_start, a_stop) and the
            # context segment [s_start, s_stop).  Anchor blocks are contiguous
            # because select_anchors returns sorted positions, so the local rows
            # are simply a slice of the global grid.
            local_num_anchors = num_anchors // sp_size
            a_start = sp_rank * local_num_anchors
            a_stop = a_start + local_num_anchors
            local_anchors = anchor_positions[a_start:a_stop]
            local_anchor_valid = anchor_valid[a_start:a_stop]

            local_seq_len = total_seq_len // sp_size
            s_start = sp_rank * local_seq_len
            s_stop = s_start + local_seq_len

            q_start = a_start * self.block_size
            q_stop = a_stop * self.block_size
            hidden_in = hidden_states[:, s_start:s_stop]
        else:
            local_num_anchors = num_anchors
            local_anchors = anchor_positions
            local_anchor_valid = anchor_valid
            s_start, s_stop = 0, total_seq_len
            q_start, q_stop = 0, num_anchors * self.block_size
            hidden_in = hidden_states

        mask_tokens_size = local_num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, local_num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, local_anchors]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, local_num_anchors*block_size, hidden_size]

        dp.begin("fc")  # [DEEP_PROFILE]
        fc_output = self.fc(hidden_in)
        fc_output = self.hidden_norm(fc_output)
        dp.end("fc")
        # shape: [1, local_seq_len, hidden_size]

        # Noise positions come from the *global* anchor positions; the context
        # half keeps its own absolute ids, so the local rotary table is
        # [ctx_local | noise_local] exactly like the [ctx | noise] KV layout.
        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, local_anchors], self.block_size
        )
        position_ids = torch.cat(
            [position_ids[:, s_start:s_stop], mask_position_ids.unsqueeze(0)], dim=1
        )
        # shape: [1, local_seq_len + local_num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_in, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size] (global: targets need global indices)

        dp.begin("verifier_lm_head")  # [DEEP_PROFILE]
        with torch.no_grad():
            if anchored_block_indices.numel() < total_seq_len:
                target_indices = (
                    anchored_block_indices
                    if self.config.sample_from_anchor
                    else (anchored_block_indices - 1) % total_seq_len
                )
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states[:, target_indices])
                )
            else:
                verifier_logits = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                if not self.config.sample_from_anchor:
                    verifier_logits = torch.roll(verifier_logits, 1, dims=1)
                targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        dp.end("verifier_lm_head")
        dp.begin("layers")  # [DEEP_PROFILE]
        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=(
                    sliding_window_attn_mask
                    if layer_idx in self.sliding_window_indices
                    else full_attn_mask
                ),
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        dp.end("layers")
        dp.begin("norm_lm_head")  # [DEEP_PROFILE]
        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)
        dp.end("norm_lm_head")
        dp.end("forward")
        # shape: [1, local_num_anchors*block_size, vocab_size]

        if sp_size > 1:
            # Targets are computed on the global grid (the verifier head is
            # frozen and cheap), then sliced to this rank's rows so that targets,
            # loss mask and logits stay row-aligned.  anchored_block_indices
            # keeps its *global* meaning because callers use it to index the
            # verifier hidden states; the local row slice is returned separately
            # through ``aligned_loss_mask`` order.
            targets = targets[:, q_start:q_stop]
            local_block_indices = anchored_block_indices[q_start:q_stop]
        else:
            local_block_indices = anchored_block_indices

        aligned_loss_mask = loss_mask.clone()[:, local_block_indices]
        # shape: [1, local_num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            local_anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, local_num_anchors*block_size]

        # For sample_from_anchor=False, mask slot 0 (anchor) since it's not trained
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, local_block_indices

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        _, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        return None, loss, metrics
