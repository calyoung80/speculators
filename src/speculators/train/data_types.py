"""Shared data structures for the training data pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DatasetRecord:
    """A preprocessed dataset row together with its storage identity."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    seq_len: int
    storage_index: int
    messages: list[dict[str, Any]] | None = None

    @classmethod
    def from_arrow_row(
        cls, row: Mapping[str, Any], *, storage_index: int
    ) -> "DatasetRecord":
        """Create a record from one formatted Arrow/Hugging Face row.

        HF ``datasets`` returns plain Python lists for List() columns (the
        producer's stack returned tensor-like rows), while downstream
        consumers (``build_client_item``, ``check_hidden_states``, packing)
        call ``.tolist()`` / ``torch.equal`` on these fields. Coerce to the
        dtypes the collate path uses (long / bool) so both list- and
        tensor-shaped rows work.
        """
        input_ids = row["input_ids"]
        loss_mask = row["loss_mask"]
        return cls(
            input_ids=input_ids
            if isinstance(input_ids, torch.Tensor)
            else torch.tensor(input_ids, dtype=torch.long),
            loss_mask=loss_mask
            if isinstance(loss_mask, torch.Tensor)
            else torch.tensor(loss_mask, dtype=torch.bool),
            seq_len=int(row["seq_len"]),
            storage_index=storage_index,
            messages=row.get("messages"),
        )


@dataclass(frozen=True)
class GenerationReceipt:
    """Handle and transfer parameters returned by one vLLM request."""

    handle: str
    transfer_params: dict[str, Any]
