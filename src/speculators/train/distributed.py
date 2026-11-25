"""Single source of truth for distributed training topology.

Stores local_rank, SP/DP sizes and ranks, and process groups.
All other modules should import getters from here rather than
maintaining their own distributed state.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

logger = logging.getLogger("speculators")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_local_rank: int = 0
_rank: int = 0
_world_size: int = 1
_is_distributed: bool = False

_sp_size: int = 1
_sp_rank: int = 0
_dp_size: int = 1
_dp_rank: int = 0

_sp_group: ProcessGroup | None = None
_dp_group: ProcessGroup | None = None


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------


def get_local_rank() -> int:
    return _local_rank


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def is_distributed() -> bool:
    return _is_distributed


def get_sp_group() -> ProcessGroup | None:
    return _sp_group


def get_dp_group() -> ProcessGroup | None:
    return _dp_group


def get_sp_size() -> int:
    return _sp_size


def get_sp_rank() -> int:
    return _sp_rank


def get_dp_size() -> int:
    return _dp_size


def get_dp_rank() -> int:
    return _dp_rank


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _init_sp_process_groups(rank: int, world_size: int, sp_size: int) -> None:
    """Initialize sequence-parallel and data-parallel process groups.

    SP groups use contiguous ranks (e.g. sp_size=2, world_size=4: {0,1}, {2,3}).
    DP groups use strided ranks (e.g. sp_size=2, world_size=4: {0,2}, {1,3}).
    """
    global _sp_group, _dp_group, _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603

    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")

    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
        )

    dp_size = world_size // sp_size

    sp_group = None
    for i in range(dp_size):
        sp_ranks = list(range(i * sp_size, (i + 1) * sp_size))
        pg = dist.new_group(sp_ranks)
        if rank in sp_ranks:
            sp_group = pg

    dp_group = None
    for i in range(sp_size):
        dp_ranks = list(range(i, world_size, sp_size))
        pg = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            dp_group = pg

    if sp_group is None or dp_group is None:
        raise RuntimeError("Failed to initialize SP/DP process groups")

    _sp_group = sp_group
    _dp_group = dp_group
    _sp_size = sp_size
    _sp_rank = rank % sp_size
    _dp_size = dp_size
    _dp_rank = rank // sp_size


def maybe_setup_distributed(sp_size: int = 1) -> None:
    """Set up distributed training if launched with ``torchrun``.

    Always populates the module-level topology state so that callers
    can use the getter functions regardless of whether SP is enabled.
    Process groups are always created when distributed — with
    ``sp_size == 1`` the DP group spans all ranks and each SP group
    contains a single rank.
    """
    global _local_rank, _rank, _is_distributed, _world_size  # noqa: PLW0603

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "LOCAL_RANK" in os.environ

    _local_rank = local_rank
    _is_distributed = distributed

    if not distributed:
        return

    torch.accelerator.set_device_index(local_rank)
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        raise ValueError("No accelerator found")
    backend = torch.distributed.get_default_backend_for_device(acc)

    # Pre-initialize TE engine (ADXL) before HCCL to prevent HCCL conflict.
    # ADXL init on NPU 0 disrupts existing HCCL; creating HCCL after ADXL avoids this.
    if os.environ.get("TE_PRE_INIT", "0") == "1":
        try:
            from hs_connectors.mooncake_te_store import pre_init_te_engine
            pre_init_te_engine()
            logger.info("TE pre-init complete (ADXL before HCCL)",
                        extra={"override_rank0_filter": True})
        except Exception as e:
            logger.warning(f"TE pre-init failed (non-fatal): {e}",
                           extra={"override_rank0_filter": True})

    dist.init_process_group(backend, device_id=local_rank)

    _rank = dist.get_rank()
    _world_size = dist.get_world_size()

    _init_sp_process_groups(_rank, _world_size, sp_size)

    logger.info(
        f"Started distributed with local_rank={local_rank}, "
        f"dp_size={_dp_size}, sp_size={_sp_size}",
        extra={"override_rank0_filter": True},
    )


def maybe_destroy_distributed() -> None:
    """Destroy the distributed process group if using distributed training."""
    global _is_distributed, _local_rank, _rank, _world_size  # noqa: PLW0603
    global _sp_size, _sp_rank, _dp_size, _dp_rank  # noqa: PLW0603
    global _sp_group, _dp_group  # noqa: PLW0603

    if not _is_distributed:
        return

    dist.destroy_process_group()
    logger.info(
        "Destroyed distributed process group",
        extra={"override_rank0_filter": True},
    )

    _is_distributed = False
    _local_rank = 0
    _rank = 0
    _world_size = 1
    _sp_size = 1
    _sp_rank = 0
    _dp_size = 1
    _dp_rank = 0
    _sp_group = None
    _dp_group = None


def _frozen_params(model: torch.nn.Module) -> set[torch.nn.Parameter]:
    """Return the model's frozen (requires_grad=False) parameters."""
    return {p for p in model.parameters() if not p.requires_grad}


# ---------------------------------------------------------------------------
# Checkpoint weight distribution
# ---------------------------------------------------------------------------


def broadcast_module_state_from_rank0(model: torch.nn.Module) -> int:
    """Copy global rank 0's parameters **and buffers** to every rank.

    Only rank 0 reads a checkpoint from disk, so the loaded weights must be
    distributed explicitly.  ``DistributedDataParallel`` broadcasts from the
    root of *its* process group; under SP that group is the strided DP group
    (``{0, 2}``, ``{1, 3}``, ...), so only one DP group contains rank 0 and the
    others would keep freshly initialised weights.  Broadcasting over the world
    group first makes every rank identical, after which DDP's own per-group
    broadcast (params and buffers) is a semantic no-op.

    Returns the number of tensors broadcast (0 when world size is 1).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    tensors = [*model.parameters(), *model.buffers()]
    for tensor in tensors:
        # collectives require contiguous storage; buffers such as a sliced RoPE
        # cache may be non-contiguous views.
        if not tensor.is_contiguous():
            tensor.data = tensor.data.contiguous()
        dist.broadcast(tensor.data, src=0)
    return len(tensors)


# ---------------------------------------------------------------------------
# Sequence-parallel gradient reduction
# ---------------------------------------------------------------------------


def sp_grad_local_tensor(grad: torch.Tensor) -> torch.Tensor:
    """Return the locally-owned shard of *grad* (pass-through for plain tensors).

    Under FSDP2 the accumulated gradient of a sharded parameter is a
    ``DTensor``; the SP all-reduce must only touch the local shard.
    ``to_local()`` is the public API and is preferred; ``_local_tensor`` is kept
    as a fallback for older torch builds.
    """
    to_local = getattr(grad, "to_local", None)
    if callable(to_local):
        return to_local()
    local = getattr(grad, "_local_tensor", None)
    if local is not None:
        return local
    return grad


def reduce_sp_gradients(model: torch.nn.Module) -> int:
    """Sum the accumulated gradients across the SP group. Call once per step.

    Must run after the whole gradient-accumulation loop and *before*
    ``clip_grad_norm_`` / ``optimizer.step()`` so that clipping sees the global
    gradient norm. Doing it per micro-batch would re-reduce gradients that are
    already accumulated and scale the earlier micro-batches by ``sp_size``.

    The reduction is deliberately *not* implemented with per-parameter hooks: an
    explicit step-boundary call is idempotent per step and immune to FSDP2
    replacing parameter objects or to checkpoint restore.

    Returns the number of gradients that were reduced (0 when SP is off).
    """
    if _sp_size <= 1 or _sp_group is None or not dist.is_initialized():
        return 0

    reduced = 0
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        dist.all_reduce(sp_grad_local_tensor(grad), group=_sp_group)
        reduced += 1
    return reduced


def apply_fully_sharded(
    model: torch.nn.Module,
    param_dtype: torch.dtype = torch.bfloat16,
    replicate_frozen: bool = False,
):
    """Applies torch FSDP fully_shard to the model, wrapping layers in FSDPModule.

    Assumes the model has a `layers` attribute containing the decoder layers.
    Model should be validated with SpeculatorModel.verify_training_compatible()
    before calling this function. With replicate_frozen=True, frozen parameters
    stay replicated on each rank instead of being sharded and all-gathered.
    This trades per-rank memory for reduced communication.

    Raises:
        ValueError: when sequence parallelism is enabled. ``fully_shard`` uses
            the default world mesh, so SP ranks would hold *different* parameter
            shards and :func:`reduce_sp_gradients` would sum incompatible
            shards. A DP-only mesh (SP replicated) is required first; until
            then the combination fails closed here, not only in the CLI.
    """
    if _sp_size > 1:
        raise ValueError(
            f"FSDP2 (apply_fully_sharded) cannot be combined with sp_size="
            f"{_sp_size}: fully_shard() shards over the full world mesh, so SP "
            "ranks hold different parameter shards and the SP gradient "
            "reduction would be incorrect. A DP-only FSDP mesh with the SP "
            "dimension replicated is required (see docs/01 §M3.6)."
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
    )

    ignored_params = None
    if replicate_frozen:
        ignored_params = _frozen_params(model)
        logger.info(
            "FSDP: replicating %d frozen params (excluded from sharding)",
            len(ignored_params),
        )

    for layer in model.layers:  # type: ignore[union-attr]
        fully_shard(layer, mp_policy=mp_policy, ignored_params=ignored_params)

    fully_shard(model, mp_policy=mp_policy, ignored_params=ignored_params)

    if ignored_params:
        # FSDP does not move ignored parameters. Keep the model on one device
        # for distributed checkpoint loading, and match the compute dtype.
        acc = torch.accelerator.current_accelerator()
        if acc is not None:
            device = torch.device(acc.type, get_local_rank())
            for param in ignored_params:
                if param.is_floating_point():
                    param.data = param.data.to(device=device, dtype=param_dtype)
                else:
                    param.data = param.data.to(device=device)

    return model
