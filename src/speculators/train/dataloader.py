from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import os

import torch
from torch.utils.data import DataLoader

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.vllm_client import validate_vllm_model
from speculators.train.data import (
    ArrowDataset,
    SpeculatorsDataset,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.hidden_states_source import (
    HiddenStatesSource,
    OfflineHiddenStatesSource,
    VllmHiddenStatesSource,
)
from speculators.train.streaming_dataloader import StreamingDataLoader

# Use file_system sharing to avoid /dev/shm exhaustion (64GB limit in
# Docker). Must be set at module import time so both the main process
# and spawned DataLoader workers pick it up.
torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


def _limit_worker_threads() -> None:
    """Limit per-worker thread pools to avoid thread exhaustion.

    With ``multiprocessing_context='spawn'``, each worker is a full process
    that re-imports numpy (OpenBLAS) and torch, each creating thread pools
    sized to the core count.  DataLoader workers only do I/O and tensor
    slicing — they don't benefit from intra-op parallelism.

    The env vars must be set before numpy/torch are imported to take effect
    on OpenBLAS/OMP.  Call this at the top of the training entry point,
    before DataLoader construction.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _configure_sharing_strategy() -> None:
    """Use file_system sharing to avoid /dev/shm exhaustion."""
    torch.multiprocessing.set_sharing_strategy("file_system")


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    torch.set_num_threads(1)
    # Must be set inside each spawned worker process: spawn does not
    # inherit the parent's sharing strategy, and the default
    # file_descriptor strategy exhausts /dev/shm (64GB in Docker).
    torch.multiprocessing.set_sharing_strategy("file_system")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.accelerator.is_available():
        torch.accelerator.set_device_index(local_rank)


def identity_fn(x):
    return x


def _setup_dataloader(
    dataset: SpeculatorsDataset,
    total_seq_len: int,
    num_workers: int = 12,
    prefetch_factor: int | None = 4,
    max_batches: int | None = None,
    seed: int = 0,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
        seed=seed,
        max_batches=max_batches,
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=identity_fn,
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
        worker_init_fn=_worker_init_fn if use_workers else None,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    verifier_name_or_path: str,
    request_timeout: float | None,
    max_retries: int,
    generation_validation_retries: int,
    max_consecutive_generation_failures: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    train_data_ratio: float = 0.9,
    max_train_batches: int | None = None,
    seed: int = 0,
    max_transfer_workers: int = 1,
    dataloader: Literal["torch", "streaming"] = "torch",
    max_queued_batches: int = 4,
) -> tuple[DataLoader | StreamingDataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.

    ``seed`` sets the training packing order and must be the same on every
    rank. Validation keeps a fixed order, so runs with different seeds are
    validated on the same packed batches.
    """
    _limit_worker_threads()
    _configure_sharing_strategy()

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    train_records = ArrowDataset(
        datapath=data_path,
        train_ratio=train_data_ratio,
        split="train",
    )
    val_records = ArrowDataset(
        datapath=data_path,
        train_ratio=train_data_ratio,
        split="val",
    )

    # ``on_missing=generate`` is the backwards-compatible switch for the
    # vLLM source; streaming always needs the vLLM source (every sample is
    # generated at runtime), so it selects it regardless of the failure
    # policy ``on_missing`` then describes. The other values select offline
    # loading and define what happens when a cached sample is absent.
    source_transfer = transfer or FileTransfer(train_records.datapath / "hidden_states")
    hidden_states_source: HiddenStatesSource
    if on_missing == "generate" or dataloader == "streaming":
        validate_vllm_model(vllm_endpoint, verifier_name_or_path)
        hidden_states_source = VllmHiddenStatesSource(
            transfer=source_transfer,
            vllm_endpoint=vllm_endpoint,
            model=verifier_name_or_path,
            request_timeout=request_timeout,
            max_retries=max_retries,
            generation_validation_retries=generation_validation_retries,
            max_consecutive_generation_failures=max_consecutive_generation_failures,
            max_transfer_workers=max_transfer_workers,
        )
    else:
        hidden_states_source = OfflineHiddenStatesSource(
            transfer=source_transfer,
            on_missing=on_missing,
        )

    train_dataset = SpeculatorsDataset(
        records=train_records,
        max_len=total_seq_len,
        hidden_states=hidden_states_source,
        hidden_size=hidden_size,
        hidden_states_dtype=hidden_states_dtype,
        num_target_layers=num_target_layers,
        preprocess=preprocess,
    )
    val_dataset = SpeculatorsDataset(
        records=val_records,
        max_len=total_seq_len,
        hidden_states=hidden_states_source,
        hidden_size=hidden_size,
        hidden_states_dtype=hidden_states_dtype,
        num_target_layers=num_target_layers,
        preprocess=preprocess,
    )

    train_loader: DataLoader | StreamingDataLoader
    if dataloader == "streaming":
        if not isinstance(hidden_states_source, VllmHiddenStatesSource):
            raise ValueError(
                "--dataloader streaming requires the vLLM generation source "
                "(the online workflow)"
            )
        # For the streaming loader every sample is generated at runtime, so
        # on_missing selects the failure policy (generate = regenerate with
        # the baseline retry budget and circuit breaker, skip/warn = degrade,
        # raise = abort) instead of the torch path's data-acquisition mode.
        train_loader = StreamingDataLoader(
            train_dataset,
            hidden_states_source,
            batch_max_length=total_seq_len,
            rank=get_dp_rank(),
            world_size=get_dp_size(),
            max_queued_batches=max_queued_batches,
            fetch_workers=max(max_transfer_workers, 1),
            # The CLI prefetch factor feeds both streaming prefetch
            # surfaces: outstanding send batches (co-batch gate) and the
            # assembled-batch yield queue.
            in_flight_batches=prefetch_factor,
            yield_queue_depth=prefetch_factor,
            on_missing=on_missing,
            seed=seed,
            max_batches=max_train_batches,
        )
    else:
        train_loader = _setup_dataloader(
            train_dataset,
            total_seq_len,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            max_batches=max_train_batches,
            seed=seed,
        )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        # Val must iterate in-process: a DataLoader worker process would
        # re-init an NPU device context (hundreds of MB HBM) via
        # _worker_init_fn at epoch end, when training HBM is at its peak —
        # this crashed epoch-0 val on Ascend 910b (NPU 507033 / EL0019,
        # 2026-11-05 03:18). _setup_dataloader branches on use_workers,
        # so num_workers=0 disables spawn/worker_init_fn/prefetch safely.
        num_workers=0,
        prefetch_factor=None,
    )

    return train_loader, val_loader
