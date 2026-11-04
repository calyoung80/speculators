"""Configuration for MTP speculator model."""

from typing import Literal

from pydantic import Field, field_validator
from transformers import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorModelConfig

__all__ = ["MTPSpeculatorConfig"]


@SpeculatorModelConfig.register("mtp")
class MTPSpeculatorConfig(SpeculatorModelConfig):
    """Configuration for MTP (Multi-Token Prediction) speculator.

    Architecture: a single MTP layer with attention and MLP, combining
    verifier hidden states with token embeddings via an explicit input
    projection. ``embed_tokens`` and ``lm_head`` share the verifier's
    full vocabulary.

    :param transformer_layer_config: Configuration for the underlying
        transformer architecture (e.g., ``Qwen2Config``). All architecture
        dimensions are derived from this config.
    :param num_nextn_predict_layers: Number of MTP prediction heads in
        the checkpoint. vLLM reads this field directly to instantiate the
        correct number of MTP head instances. Currently only ``1`` is
        supported.
    """

    speculators_model_type: Literal["mtp"] = "mtp"
    architectures: list[str] = Field(
        default_factory=lambda: ["MTPDraftModel"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen2Config,
        description=(
            "Underlying transformer architecture config "
            "(e.g., Qwen2Config, Qwen3NextConfig)"
        ),
    )

    num_nextn_predict_layers: int = Field(
        default=1,
        description=(
            "Number of MTP prediction heads in the checkpoint. vLLM "
            "reads this field to create the correct number of "
            "speculator head instances."
        ),
    )

    mtp_own_head: bool = Field(
        default=False,
        description=(
            "The native MTP layer owns an independent head "
            "(``shared_head.head.weight`` distinct from the verifier's "
            "``lm_head.weight``).  When true, the head is loaded from the "
            "native package, saved with checkpoints, and written back by "
            "stitch."
        ),
    )
    mtp_own_embed: bool = Field(
        default=False,
        description=(
            "The native MTP layer owns an independent token embedding "
            "(``embed_tokens.weight`` distinct from the verifier's).  "
            "Load/save/stitch semantics as for mtp_own_head."
        ),
    )
    mtp_has_rot: bool = Field(
        default=False,
        description=(
            "The native MTP layer carries a rot projection (top-level "
            "``rot.weight`` sidecar or ``model.layers.<N>.rot.weight``).  "
            "GlmMTPLayer applies it at the start of the MTP forward "
            "(plan section 8.4)."
        ),
    )

    @property
    def hidden_size(self) -> int:
        return self.transformer_layer_config.hidden_size  # type: ignore[return-value]

    @property
    def vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size  # type: ignore[return-value]

    @property
    def draft_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def num_speculative_steps(self) -> int:
        if (
            self.speculators_config is None
            or not self.speculators_config.proposal_methods
        ):
            raise ValueError(
                "Cannot determine num_speculative_steps: "
                "speculators_config is missing or has no proposal_methods. "
                "Provide a SpeculatorsConfig with at least one proposal method."
            )
        return self.speculators_config.proposal_methods[0].speculative_tokens  # type: ignore[union-attr,attr-defined]

    @field_validator("num_nextn_predict_layers")
    @classmethod
    def validate_num_nextn_predict_layers(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"MTP currently supports 1 layer, got {value}.")
        return value
