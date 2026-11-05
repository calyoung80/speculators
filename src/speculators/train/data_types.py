"""Shared data structures for the training data pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


def _coerce_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Coerce one Arrow/HF row field to a tensor of the contract dtype.

    ``datasets`` rows yield plain Python lists for list-typed columns (and
    numpy arrays under other output formats), but every downstream consumer
    -- ``build_client_item``'s ``.tolist()``, the trainer's
    ``torch.equal`` token check, the collate -- assumes ``torch.Tensor``.
    Coerce once at the ``DatasetRecord`` boundary so all consumers see the
    documented contract regardless of how the row was materialized.
    """
    if isinstance(value, torch.Tensor):
        return value if value.dtype == dtype else value.to(dtype)
    return torch.tensor(value, dtype=dtype)


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
        """Create a record from one formatted Arrow/Hugging Face row."""
        return cls(
            input_ids=_coerce_tensor(row["input_ids"], torch.long),
            loss_mask=_coerce_tensor(row["loss_mask"], torch.bool),
            seq_len=int(row["seq_len"]),
            storage_index=storage_index,
            messages=row.get("messages"),
        )


@dataclass(frozen=True)
class GenerationReceipt:
    """Handle and transfer parameters returned by one vLLM request."""

    handle: str
    transfer_params: dict[str, Any]
