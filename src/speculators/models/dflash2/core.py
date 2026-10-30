from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig

from speculators.losses import LossConfig, kl_div_loss, resolve_loss_config, tv_loss
from speculators.model import SpeculatorModel
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import (
    compute_metrics,
    selector_training_candidates,
)
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "DFlash2DraftModel",
]

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


@SpeculatorModel.register("dflash2")
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash with local convolution and bilinear candidate reranking."""

    config_class: ClassVar[type[DFlash2SpeculatorConfig]] = DFlash2SpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    def __init__(self, config: DFlash2SpeculatorConfig) -> None:
        target_vocab_size = config.transformer_layer_config.vocab_size
        if config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "DFlash2 currently requires the full verifier vocabulary: "
                f"draft_vocab_size={config.draft_vocab_size}, "
                f"verifier_vocab_size={target_vocab_size}."
            )
        if config.selector_top_k > target_vocab_size:
            raise ValueError(
                f"selector_top_k ({config.selector_top_k}) cannot exceed "
                f"verifier_vocab_size ({target_vocab_size})."
            )
        super().__init__(config=config)

        for layer_ in self.layers:
            assert isinstance(layer_, Qwen3DFlash2DecoderLayer)  # noqa: S101
            layer_.reset_convolutions()

        initializer_range = getattr(
            config.transformer_layer_config, "initializer_range", 0.02
        )
        self.candidate_selector = CandidateSelector(
            vocab_size=target_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
            initializer_range=initializer_range,
        )

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> Qwen3DFlash2DecoderLayer:
        assert isinstance(config, DFlash2SpeculatorConfig)  # noqa: S101
        return Qwen3DFlash2DecoderLayer(
            config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlash2DraftModel":
        """Create a DFlash2 model from the shared training arguments."""
        config = DFlash2SpeculatorConfig(
            **cls._build_base_config_kwargs("dflash2", verifier_config, **kwargs),
            conv_kernel_size=kwargs.get("conv_kernel_size", 2),
            conv_group_size=kwargs.get("conv_group_size", 16),
            selector_rank=kwargs.get("selector_rank", 256),
            selector_top_k=kwargs.get("selector_top_k", 16),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve the unary and selector objectives used during training."""
        implementation = kwargs.get("loss_implementation", "fused")
        loss_config = resolve_loss_config(kwargs["loss_fn"], implementation)
        tv_loss_fn = resolve_loss_config("tv", implementation)["tv"][0]
        shared = {
            "loss_config": loss_config,
            "tv_loss_fn": tv_loss_fn,
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "objective_chunk_blocks": kwargs.get("objective_chunk_blocks", 0),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "selector_loss_alpha": kwargs.get("selector_loss_alpha", 1.0),
        }
        return dict(shared), dict(shared)

    def _predecessor_ids(
        self,
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        anchored_block_indices: torch.Tensor,  # shape: [num_anchors*block_size]
    ) -> torch.Tensor:
        block_tokens = input_ids[0, anchored_block_indices].view(
            -1, self.block_size
        )  # shape: [num_anchors, block_size]
        if self.config.sample_from_anchor:
            return block_tokens
        # For token sequence 0 1 2 3 return previous token ids 0 0 1 2
        return torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)

    def _compute_objective(
        self,
        hidden: torch.Tensor,
        unary_logits: torch.Tensor,
        targets: torch.Tensor,
        aligned_loss_mask: torch.Tensor,
        predecessor_ids: torch.Tensor,
        *,
        loss_config: LossConfig | None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        gamma: float,
        selector_loss_alpha: float,
        per_position_loss_weight: str,
        dpace_alpha: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Evaluate unary and selector losses for one objective block."""
        target_ids = targets.argmax(dim=-1)
        candidate_ids = unary_logits.topk(self.candidate_selector.top_k, dim=-1).indices
        training_candidate_ids, target_positions, contains_target = (
            selector_training_candidates(candidate_ids, target_ids)
        )
        candidate_logits = self.candidate_selector.score_candidates(
            unary_logits,
            hidden,
            predecessor_ids,
            training_candidate_ids,
        )
        return compute_metrics(
            unary_logits=unary_logits,
            targets=targets,
            training_candidate_ids=training_candidate_ids,
            candidate_logits=candidate_logits,
            target_positions=target_positions,
            contains_target=contains_target,
            loss_mask=aligned_loss_mask,
            block_size=self.block_size,
            top_k=self.candidate_selector.top_k,
            sample_from_anchor=self.config.sample_from_anchor,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            selector_loss_alpha=selector_loss_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
        )

    def _compute_objective_chunk(
        self,
        hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        aligned_loss_mask: torch.Tensor,
        predecessor_ids: torch.Tensor,
        *,
        loss_config: LossConfig | None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        gamma: float,
        selector_loss_alpha: float,
        per_position_loss_weight: str,
        dpace_alpha: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Project and score one anchor chunk.

        The teacher projection is deliberately inside this method so a caller can
        checkpoint the whole objective chunk and release its [Q, vocab] tensors
        before moving to the next chunk.
        """
        with torch.no_grad():
            targets = self.verifier_lm_head(target_hidden)
        unary_logits = self.lm_head(hidden)
        return self._compute_objective(
            hidden,
            unary_logits,
            targets,
            aligned_loss_mask,
            predecessor_ids,
            loss_config=loss_config,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            selector_loss_alpha=selector_loss_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
        )

    @staticmethod
    def _merge_chunk_metrics(
        total_metrics: dict[str, Any],
        chunk_metrics: dict[str, Any],
        valid_count: torch.Tensor,
        block_valid_count: torch.Tensor,
    ) -> None:
        """Merge metrics while preserving the numerator/denominator contract."""
        mean_names = ("loss", "unary_loss", "selector_loss")
        for name, value in chunk_metrics.items():
            metric_value = value
            if name.endswith("_sum"):
                metric_name = name[: -len("_sum")]
                if metric_name in mean_names:
                    metric_value = value * valid_count
            if (
                name.endswith("_total")
                and name[: -len("_total")] in mean_names
                or name == "accept_rate_total"
            ):
                metric_value = valid_count.to(value.dtype)
            elif name == "accept_len_total":
                metric_value = block_valid_count.to(value.dtype)
            if name in total_metrics:
                total_metrics[name] = total_metrics[name] + metric_value
            else:
                total_metrics[name] = (
                    metric_value.clone()
                    if torch.is_tensor(metric_value)
                    else metric_value
                )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 4.0,
        max_anchors: int = 512,
        objective_chunk_blocks: int = 0,
        selector_loss_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ) -> tuple[None, torch.Tensor, dict[str, Any]]:
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")

        if objective_chunk_blocks == 0:
            hidden, unary_logits, targets, aligned_loss_mask, block_indices = (
                self._backbone_forward(
                    hidden_states,
                    input_ids,
                    loss_mask,
                    verifier_last_hidden_states,
                    document_ids,
                    position_ids,
                    max_anchors=max_anchors,
                    **kwargs,
                )
            )
            predecessor_ids = self._predecessor_ids(input_ids, block_indices)
            loss, metrics = self._compute_objective(
                hidden,
                unary_logits,
                targets,
                aligned_loss_mask,
                predecessor_ids.reshape(1, -1),
                loss_config=loss_config,
                tv_loss_fn=tv_loss_fn,
                gamma=gamma,
                selector_loss_alpha=selector_loss_alpha,
                per_position_loss_weight=per_position_loss_weight,
                dpace_alpha=dpace_alpha,
            )
            return None, loss, metrics

        hidden, target_hidden, aligned_loss_mask, block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                return_objective_inputs=True,
                **kwargs,
            )
        )
        predecessor_ids = self._predecessor_ids(input_ids, block_indices).reshape(1, -1)
        chunk_size = objective_chunk_blocks * self.block_size
        total_loss_numerator = torch.zeros(
            (), device=hidden.device, dtype=torch.float32
        )
        total_valid = torch.zeros((), device=hidden.device, dtype=torch.float32)
        metrics: dict[str, Any] = {}

        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            hidden_chunk = hidden[:, start:end]
            target_hidden_chunk = target_hidden[:, start:end]
            mask_chunk = aligned_loss_mask[:, start:end]
            predecessor_chunk = predecessor_ids[:, start:end]
            valid_count = mask_chunk.detach().float().sum()
            start_pos = 0 if self.config.sample_from_anchor else 1
            block_valid_count = (
                mask_chunk.detach()
                .reshape(-1, self.block_size)[:, start_pos:]
                .any(dim=-1)
                .float()
                .sum()
            )

            def objective_chunk(
                chunk_hidden: torch.Tensor,
                chunk_target_hidden: torch.Tensor,
                chunk_mask: torch.Tensor = mask_chunk,
                chunk_predecessor: torch.Tensor = predecessor_chunk,
            ) -> tuple[torch.Tensor, dict[str, Any]]:
                return self._compute_objective_chunk(
                    chunk_hidden,
                    chunk_target_hidden,
                    chunk_mask,
                    chunk_predecessor,
                    loss_config=loss_config,
                    tv_loss_fn=tv_loss_fn,
                    gamma=gamma,
                    selector_loss_alpha=selector_loss_alpha,
                    per_position_loss_weight=per_position_loss_weight,
                    dpace_alpha=dpace_alpha,
                )

            if torch.is_grad_enabled():
                chunk_loss, chunk_metrics = checkpoint(
                    objective_chunk,
                    hidden_chunk,
                    target_hidden_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_loss, chunk_metrics = objective_chunk(
                    hidden_chunk, target_hidden_chunk
                )
            total_loss_numerator = (
                total_loss_numerator + chunk_loss.float() * valid_count
            )
            self._merge_chunk_metrics(
                metrics, chunk_metrics, valid_count, block_valid_count
            )
            total_valid = total_valid + valid_count

        denominator = total_valid.clamp_min(1.0)
        loss = total_loss_numerator / denominator
        for name in ("accept_rate_total", "accept_len_total"):
            if name in metrics:
                metrics[name] = metrics[name].clamp_min(1.0)
        for name in ("loss_total", "unary_loss_total", "selector_loss_total"):
            if name in metrics:
                metrics[name] = metrics[name].clamp_min(1.0)
        metrics["loss_sum"] = total_loss_numerator.detach()
        metrics["loss_total"] = total_valid.detach().clamp_min(1.0)
        return None, loss, metrics
