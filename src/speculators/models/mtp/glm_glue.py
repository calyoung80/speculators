"""Thin GLM-MoE-DSA adaptation for the MTP draft layer.

The MTP head reuses the upstream ``GlmMoeDsaDecoderLayer`` instead of
re-implementing a GLM decoder.  This module only contains the deltas that
cannot be expressed through ``ModelComponents``:

* force a single sparse MoE layer with a full DSA indexer;
* force the eager expert loop (the grouped/GEMM kernels are not required and
  are not exercised by the MTP training path);
* align the MLA ``q_a`` / ``kv_a`` LayerNorm eps with the checkpoint
  ``rms_norm_eps`` (upstream defaults to 1e-6);
* zero the token embedding at position 0 without touching the vocabulary row;
* force a cacheless, non-shared-indexer forward.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from speculators.models.mtp.model_definitions import MTPLayerMixin

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaRMSNorm,
)

__all__ = ["GlmMTPLayer"]


class GlmMTPLayer(MTPLayerMixin, GlmMoeDsaDecoderLayer):
    """Upstream GLM decoder layer plus the MTP input/final projections."""

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int = 0) -> None:
        modified = copy.copy(config)
        # The MTP head is a single sparse MoE block that must compute its own
        # DSA top-k mask (never reuse a previous layer's indices).
        n_layers = max(layer_idx + 1, 1)
        modified.indexer_types = ["full"] * n_layers
        modified.mlp_layer_types = ["sparse"] * n_layers
        modified._experts_implementation = "eager"  # noqa: SLF001

        super().__init__(modified, layer_idx)
        self._setup_mtp_modules(modified, GlmMoeDsaRMSNorm)

        # Upstream builds these two norms without an eps argument (1e-6), while
        # the checkpoint and the rest of the decoder use ``rms_norm_eps``.
        for norm_name in ("q_a_layernorm", "kv_a_layernorm"):
            norm = getattr(self.self_attn, norm_name, None)
            if norm is not None:
                norm.variance_epsilon = modified.rms_norm_eps

        indexer = getattr(self.self_attn, "indexer", None)
        if indexer is not None:
            for parameter in indexer.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if position_ids is not None and token_embeddings.shape[:-1] == position_ids.shape:
            token_embeddings = token_embeddings.masked_fill(
                (position_ids == 0).unsqueeze(-1), 0
            )

        kwargs["past_key_values"] = None
        kwargs["use_cache"] = False
        kwargs["prev_topk_indices"] = None

        return super().forward(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
