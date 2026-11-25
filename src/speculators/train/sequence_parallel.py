"""Ulysses sequence parallelism for speculator training (NPU-friendly).

This module implements the primitives needed to shard the training sequence
across a subset of ranks (the *SP group*) for the DFlash/DSpark draft models:

* ``all_to_all_sp``      -- Ulysses all-to-all primitive (differentiable,
                            torch.compile friendly custom op);
* ``ulysses_scatter`` /
  ``ulysses_gather``     -- sequence-parallel <-> head-parallel layout changes;
* ``maybe_replicate_kv_heads`` -- GQA support when ``sp_size > num_kv_heads``;
* ``split_batch_for_sp`` -- batch dict splitting for models that consume a
                            sequence-split batch (eagle3 path);
* ``sp_ratio_reduce``    -- SP-group numerator/denominator reduction for the
                            masked-mean losses and metrics;
* ``dflash_sdpa_sp_attention`` -- DFlash/DSpark attention kernel that keeps the
                            ``[context | noise]`` KV layout correct across the
                            all-to-all and uses ``scaled_dot_product_attention``
                            instead of ``flex_attention``, so it runs on Ascend
                            NPUs (community issue #1115: flex attention is not
                            available there).

All collectives use plain ``torch.distributed`` (gloo on CPU tests, HCCL on
Ascend).  With ``sp_size == 1`` every helper is an identity, so the non-SP
training path is unchanged.

Design notes
------------
DFlash attention has ``Q = noise`` (local anchor blocks) and
``KV = [ctx | noise]`` concatenated along the sequence dimension.  A naive
all-to-all over KV interleaves context and noise from different ranks and
breaks the anchor block mask, so the context and noise halves are exchanged
separately and concatenated back as ``[global_ctx | global_noise]`` -- the
exact layout ``create_anchor_block_mask_mod`` expects.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

__all__ = [
    "all_to_all_sp",
    "dflash_sdpa_sp_attention",
    "materialize_dense_mask",
    "maybe_replicate_kv_heads",
    "sp_ratio_reduce",
    "sp_sum_with_identity_grad",
    "split_batch_for_sp",
    "ulysses_gather",
    "ulysses_scatter",
]

# Batch keys that hold the packed sequence and must be split along dim 1.
_SPLIT_KEYS = frozenset(
    {
        "hidden_states",
        "verifier_last_hidden_states",
        "input_ids",
        "loss_mask",
        "position_ids",
    }
)

# Keys that must stay full-length: the attention mask needs global document ids.
_KEEP_FULL_KEYS = frozenset({"document_ids"})

_MIN_SPLIT_NDIM = 2
_MIN_TENSOR_NDIM = 2

# Dense-mask materialization granularity (query rows per chunk).  Keeps the
# temporary ``[1, 1, chunk, KV]`` mask bounded for long-context training.
_MASK_CHUNK = int(os.environ.get("SPECULATORS_SP_MASK_CHUNK", "1024"))


# ---------------------------------------------------------------------------
# Differentiable all-to-all (torch.compile-friendly custom op)
# ---------------------------------------------------------------------------


@torch.library.custom_op("speculators::all_to_all_sp", mutates_args=())
def all_to_all_sp(
    input_tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Ulysses all-to-all: scatter ``scatter_dim``, gather ``gather_dim``.

    Rank ``i`` keeps the ``i``-th chunk along ``scatter_dim`` of every rank and
    concatenates the received chunks along ``gather_dim``.

    Args:
        input_tensor: Input tensor, e.g. ``(B, H, S_local, D)``.
        scatter_dim: Dimension to split into ``sp_size`` chunks (heads for
            :func:`ulysses_scatter`).
        gather_dim: Dimension to concatenate along (sequence for
            :func:`ulysses_scatter`).

    Returns:
        Tensor with ``scatter_dim`` divided by ``sp_size`` and ``gather_dim``
        multiplied by ``sp_size``.
    """
    from speculators.train.distributed import get_sp_group, get_sp_size

    sp_size = get_sp_size()
    if sp_size == 1:
        return input_tensor

    sp_group = get_sp_group()
    if sp_group is None or not dist.is_initialized():
        raise RuntimeError("all_to_all_sp requires an initialized SP process group")

    ndim = input_tensor.ndim
    if ndim < _MIN_TENSOR_NDIM:
        raise ValueError(
            f"all_to_all_sp expects a tensor with ndim >= 2, got shape "
            f"{tuple(input_tensor.shape)}"
        )
    scatter_dim %= ndim
    gather_dim %= ndim
    if scatter_dim == gather_dim:
        raise ValueError(
            f"all_to_all_sp: scatter_dim and gather_dim must differ, both are "
            f"{scatter_dim}"
        )
    if input_tensor.shape[scatter_dim] % sp_size != 0:
        raise ValueError(
            f"all_to_all_sp: dimension {scatter_dim} of size "
            f"{input_tensor.shape[scatter_dim]} is not divisible by "
            f"sp_size ({sp_size}); unequal chunks would deadlock c10d "
            "all_to_all"
        )

    input_chunks = [
        c.contiguous() for c in input_tensor.chunk(sp_size, dim=scatter_dim)
    ]

    if _supports_all_to_all(sp_group):
        output_chunks = [torch.empty_like(c) for c in input_chunks]
        dist.all_to_all(output_chunks, input_chunks, group=sp_group)
        return torch.cat(output_chunks, dim=gather_dim).contiguous()

    # gloo (CPU tests) has no all_to_all.  Gather every rank's full input once
    # and take, from each sender, the chunk that sender addressed to this rank:
    # output_chunks[i] = chunk(my_rank) of rank i's input.  Same result as
    # all_to_all, at the cost of sp x the input memory (test-only path).
    gathered = [torch.empty_like(input_tensor.contiguous()) for _ in range(sp_size)]
    dist.all_gather(gathered, input_tensor.contiguous(), group=sp_group)
    my_rank = dist.get_rank(sp_group)
    output_chunks = [
        sender.chunk(sp_size, dim=scatter_dim)[my_rank] for sender in gathered
    ]
    return torch.cat(output_chunks, dim=gather_dim).contiguous()


def _supports_all_to_all(group) -> bool:
    """True when the process-group backend implements c10d ``all_to_all``."""
    backend = str(dist.get_backend(group)).lower()
    return "nccl" in backend or "hccl" in backend


@all_to_all_sp.register_fake
def _all_to_all_sp_fake(
    input_tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    from speculators.train.distributed import get_sp_size

    sp_size = get_sp_size()
    ndim = input_tensor.ndim
    scatter_dim %= ndim
    gather_dim %= ndim
    shape = list(input_tensor.shape)
    shape[scatter_dim] //= sp_size
    shape[gather_dim] *= sp_size
    return input_tensor.new_empty(shape)


def _all_to_all_sp_setup(ctx, inputs, _output) -> None:
    input_tensor, scatter_dim, gather_dim = inputs
    ndim = input_tensor.ndim
    ctx.scatter_dim = scatter_dim % ndim
    ctx.gather_dim = gather_dim % ndim


def _all_to_all_sp_backward(ctx, grad_output: torch.Tensor) -> tuple:
    return (
        torch.ops.speculators.all_to_all_sp(
            grad_output, ctx.gather_dim, ctx.scatter_dim
        ),
        None,
        None,
    )


torch.library.register_autograd(
    "speculators::all_to_all_sp",
    _all_to_all_sp_backward,
    setup_context=_all_to_all_sp_setup,
)


# ---------------------------------------------------------------------------
# Ulysses layout changes
# ---------------------------------------------------------------------------


def ulysses_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sequence-parallel -> head-parallel layout.

    ``(B,H,S_local,D)`` -> ``(B,H/sp,S_full,D)``.

    The SP degree always comes from the current process group.
    """
    from speculators.train.distributed import get_sp_size

    if get_sp_size() <= 1:
        return x
    return torch.ops.speculators.all_to_all_sp(x, 1, 2)


def ulysses_gather(x: torch.Tensor) -> torch.Tensor:
    """Head-parallel -> sequence-parallel layout.

    ``(B,H_local,S_full,D)`` -> ``(B,H_local*sp,S_full/sp,D)``.

    The SP degree always comes from the current process group.
    """
    from speculators.train.distributed import get_sp_size

    if get_sp_size() <= 1:
        return x
    return torch.ops.speculators.all_to_all_sp(x, 2, 1)


# ---------------------------------------------------------------------------
# GQA head replication
# ---------------------------------------------------------------------------


def maybe_replicate_kv_heads(
    key: torch.Tensor,
    value: torch.Tensor,
    sp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make KV heads splittable by ``sp_size`` for GQA models.

    When ``sp_size <= num_kv_heads`` the head count must be divisible by
    ``sp_size`` and the tensors are returned unchanged.  When ``sp_size``
    exceeds the KV head count, each KV head is replicated ``sp_size /
    num_kv_heads`` times so the all-to-all can distribute them evenly.

    Args:
        key: ``(B, num_kv_heads, S, D)``.
        value: ``(B, num_kv_heads, S, D)``.
        sp_size: Sequence-parallel world size.

    Returns:
        Possibly replicated ``(key, value)``.
    """
    if sp_size <= 1:
        return key, value

    num_kv_heads = key.shape[1]
    if sp_size <= num_kv_heads:
        if num_kv_heads % sp_size != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by "
                f"sp_size ({sp_size})"
            )
        return key, value

    if sp_size % num_kv_heads != 0:
        raise ValueError(
            f"sp_size ({sp_size}) must be divisible by num_kv_heads "
            f"({num_kv_heads}) when sp_size > num_kv_heads"
        )

    replication_factor = sp_size // num_kv_heads
    key = key.repeat_interleave(replication_factor, dim=1)
    value = value.repeat_interleave(replication_factor, dim=1)
    return key, value


# ---------------------------------------------------------------------------
# Batch splitting (models that consume a sequence-split batch)
# ---------------------------------------------------------------------------


def split_batch_for_sp(
    batch: dict[str, torch.Tensor],
    sp_rank: int,
    sp_size: int,
) -> dict[str, torch.Tensor]:
    """Split sequence tensors of *batch* along dim 1 for SP rank *sp_rank*.

    ``document_ids`` stays full-length (needed to build the global attention
    mask).  ``full_seq_len`` is added to the result so the model can recover the
    original packed length.

    Args:
        batch: Batch dict with tensors shaped ``(B, S, ...)`` (or ``(B, S)``).
        sp_rank: Rank inside the SP group.
        sp_size: Sequence-parallel world size.

    Returns:
        The sequence-split batch (identity when ``sp_size <= 1``).
    """
    if sp_size <= 1:
        return batch

    out: dict[str, torch.Tensor | int] = {}
    full_seq_len: int | None = None

    for key, tensor in batch.items():
        if (
            key in _SPLIT_KEYS
            and isinstance(tensor, torch.Tensor)
            and tensor.dim() >= _MIN_SPLIT_NDIM
        ):
            seq_len = tensor.shape[1]
            if full_seq_len is None:
                full_seq_len = seq_len
            if seq_len % sp_size != 0:
                raise ValueError(
                    f"Sequence length ({seq_len}) of '{key}' must be "
                    f"divisible by sp_size ({sp_size})"
                )
            out[key] = tensor.chunk(sp_size, dim=1)[sp_rank].contiguous()
        elif key in _KEEP_FULL_KEYS:
            if (
                full_seq_len is None
                and isinstance(tensor, torch.Tensor)
                and (tensor.dim() >= _MIN_SPLIT_NDIM)
            ):
                full_seq_len = tensor.shape[1]
            out[key] = tensor
        else:
            out[key] = tensor

    if full_seq_len is not None:
        # Python int on purpose: a Tensor here would leak into model(**batch) as
        # an unknown kwarg / extra device transfer. See docs/02_api_design.md 1.4.
        out["full_seq_len"] = full_seq_len
    return out


# ---------------------------------------------------------------------------
# SP-group loss reduction
# ---------------------------------------------------------------------------


def sp_sum_with_identity_grad(t: torch.Tensor) -> torch.Tensor:
    """Sum ``t`` across the SP group, keeping autograd identity on the local part.

    Forward value = global sum; backward ``d(out)/d(t) = 1`` so each rank keeps
    the gradient of its own contribution -- exactly the mathematical semantics
    of a sum.  The reduction is done on a detached copy and re-attached through
    ``t`` so no in-place collective ever touches an autograd-tracked tensor.

    No-op when ``sp_size <= 1`` (including non-distributed runs).
    """
    from speculators.train.distributed import get_sp_group, get_sp_size

    if get_sp_size() <= 1:
        return t
    sp_group = get_sp_group()
    if sp_group is None or not dist.is_initialized():
        return t
    reduced = t.detach().clone()
    dist.all_reduce(reduced, group=sp_group)
    return t + (reduced - t.detach())


def sp_ratio_reduce(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum the numerator/denominator of a masked mean across the SP group.

    Each SP rank computes the statistic from its own sequence/anchor slice; the
    global masked mean is ``sum(numerators) / sum(denominators)``.  Uses
    :func:`sp_sum_with_identity_grad`, so the returned tensors stay connected to
    the local autograd graph with unit gradient (no in-place mutation).

    No-op when ``sp_size <= 1`` (including non-distributed runs).
    """
    return (
        sp_sum_with_identity_grad(numerator),
        sp_sum_with_identity_grad(denominator),
    )


# ---------------------------------------------------------------------------
# Dense mask materialization (flex BlockMask -> additive SDPA mask)
# ---------------------------------------------------------------------------


def materialize_dense_mask(
    block_mask,
    q_start: int,
    q_stop: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize an additive SDPA mask for query rows ``[q_start, q_stop)``.

    Evaluates the flex ``mask_mod`` on a broadcasted index grid -- verified to be
    exactly equal to the corresponding slice of the full dense mask (see
    ``tests/unit/train/test_sequence_parallel.py``).

    Args:
        block_mask: A flex ``BlockMask`` (duck-typed: anything with
            ``mask_mod``).
        q_start: First query row (inclusive).
        q_stop: Last query row (exclusive).
        kv_len: Number of key/value rows.
        device: Target device.
        dtype: Mask dtype; must match the query dtype for SDPA.

    Returns:
        ``[1, 1, q_stop - q_start, kv_len]`` additive mask (0 keep, -inf drop).
    """
    mask_mod = getattr(block_mask, "mask_mod", None)
    if mask_mod is None:
        raise TypeError("materialize_dense_mask expects a BlockMask-like object")

    with torch.no_grad():
        b_idx = torch.zeros(1, dtype=torch.long, device=device)
        h_idx = torch.zeros(1, dtype=torch.long, device=device)
        q_idx = torch.arange(q_start, q_stop, device=device).view(1, 1, -1, 1)
        kv_idx = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
        keep = mask_mod(b_idx, h_idx, q_idx, kv_idx)
        mask = torch.zeros(keep.shape, device=device, dtype=dtype)
        mask.masked_fill_(~keep, float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# DFlash/DSpark SP attention (SDPA kernel)
# ---------------------------------------------------------------------------


def _mask_chunks(
    attention_mask,
    q_total: int,
    kv_total: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Yield ``(start, stop, additive_mask_chunk)`` for every query chunk."""
    chunk = max(1, min(_MASK_CHUNK, q_total))
    if isinstance(attention_mask, torch.Tensor):
        for start in range(0, q_total, chunk):
            stop = min(start + chunk, q_total)
            part = attention_mask[:, :, start:stop]
            if part.dtype != dtype:
                part = part.to(dtype)
            yield start, stop, part
        return
    if attention_mask is None:
        yield 0, q_total, None
        return
    for start in range(0, q_total, chunk):
        stop = min(start + chunk, q_total)
        yield (
            start,
            stop,
            materialize_dense_mask(
                attention_mask, start, stop, kv_total, device, dtype
            ),
        )


def dflash_sdpa_sp_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, None]:
    """Ulysses SP attention for DFlash/DSpark on top of SDPA.

    Matches the ``flex_attention_forward`` interface so it can be registered in
    ``ALL_ATTENTION_FUNCTIONS`` and selected via ``_attn_impl_override``.

    The KV layout is ``[context_local | noise_local]``; each half is
    all-to-all'ed separately and concatenated back as
    ``[global_ctx | global_noise]`` to preserve the anchor block mask layout.
    """
    from speculators.train.distributed import (
        get_sp_group,
        get_sp_size,
    )

    sp_size = get_sp_size()
    if sp_size <= 1:
        from speculators.models.attention import flex_attention_forward

        return flex_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, **_kwargs
        )

    sp_group = get_sp_group()
    if sp_group is None or not dist.is_initialized():
        raise RuntimeError("dflash_sdpa_sp_attention requires an initialized SP group")

    key, value = maybe_replicate_kv_heads(key, value, sp_size)

    q_local = query.shape[2]
    kv_local = key.shape[2]
    ctx_len = kv_local - q_local
    if ctx_len < 0:
        raise ValueError(
            f"KV length ({kv_local}) smaller than query length ({q_local})"
        )
    k_ctx, k_noise = key.split([ctx_len, q_local], dim=2)
    v_ctx, v_noise = value.split([ctx_len, q_local], dim=2)

    query = ulysses_scatter(query)
    key = torch.cat(
        [ulysses_scatter(k_ctx), ulysses_scatter(k_noise)],
        dim=2,
    )
    value = torch.cat(
        [ulysses_scatter(v_ctx), ulysses_scatter(v_noise)],
        dim=2,
    )

    q_total = query.shape[2]
    kv_total = key.shape[2]

    if hasattr(attention_mask, "mask_mod"):
        mask_q, mask_kv = tuple(attention_mask.shape)[2:]
        if mask_q != q_total or mask_kv != kv_total:
            raise ValueError(
                f"Attention mask shape {(mask_q, mask_kv)} does not match the "
                f"SP attention grid {(q_total, kv_total)}"
            )

    if _SDPA_STREAM_KV > 0:
        out = _streaming_sdpa_apply(
            query, key, value, attention_mask, q_total, kv_total, scaling
        )
        out = ulysses_gather(out)
        out = out.transpose(1, 2).contiguous()
        return out, None
    outs = []
    for start, stop, mask_chunk in _mask_chunks(
        attention_mask, q_total, kv_total, query.dtype, query.device
    ):
        outs.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:stop],
                key,
                value,
                attn_mask=mask_chunk,
                dropout_p=0.0,
                scale=scaling,
                is_causal=False,
                enable_gqa=query.shape[1] != key.shape[1],
            )
        )
    out = torch.cat(outs, dim=2)

    out = ulysses_gather(out)
    out = out.transpose(1, 2).contiguous()
    return out, None


_SDPA_STREAM_KV = int(os.environ.get("SPECULATORS_SDPA_STREAM_KV", "0"))


def _streaming_sdpa(query, key, value, attention_mask, q_total, kv_total, scaling):
    """Flash-style streaming SDPA: Q-chunk x KV-chunk online softmax.

    Numerically equivalent to F.scaled_dot_product_attention with an additive
    mask (fp32 running max/sum), but never materializes the full
    [heads, q, kv] score matrix -- per-block scores stay in the tens of MB.
    """
    kv_chunk = _SDPA_STREAM_KV
    hq, hkv = query.shape[1], key.shape[1]
    if hq != hkv:
        rep = hq // hkv
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    outs = []
    for start, stop, mask_chunk in _mask_chunks(
        attention_mask, q_total, kv_total, query.dtype, query.device
    ):
        q = query[:, :, start:stop]
        m_run = None
        l_run = None
        acc = None
        for ks in range(0, kv_total, kv_chunk):
            ke = min(ks + kv_chunk, kv_total)
            k = key[:, :, ks:ke]
            v = value[:, :, ks:ke]
            s = torch.matmul(q, k.transpose(-1, -2)).float() * scaling
            mc = mask_chunk[:, :, :, ks:ke]
            if mc.dtype != s.dtype:
                mc = mc.to(s.dtype)
            s = s + mc
            m_blk = s.amax(dim=-1, keepdim=True)
            if m_run is None:
                m_run = m_blk.clamp(min=-1e30)
                p = torch.exp(s - m_run)
                l_run = p.sum(dim=-1, keepdim=True)
                acc = torch.matmul(p.to(v.dtype), v).float()
            else:
                m_new = torch.maximum(m_run, m_blk)
                alpha = torch.exp(m_run - m_new)
                p = torch.exp(s - m_new)
                l_run = l_run * alpha + p.sum(dim=-1, keepdim=True)
                acc = acc * alpha + torch.matmul(p.to(v.dtype), v).float()
                m_run = m_new
        outs.append((acc / l_run.clamp(min=1e-20)).to(query.dtype))
    return torch.cat(outs, dim=2)


def _streaming_sdpa_apply(
    query, key, value, attention_mask, q_total, kv_total, scaling
):
    """Route: custom flash-style Function in training (graph-free saved state),
    plain streaming forward under no_grad otherwise."""
    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        return _StreamingSDPAFn.apply(
            query,
            key,
            value,
            attention_mask,
            q_total,
            kv_total,
            scaling,
            _SDPA_STREAM_KV,
        )
    with torch.no_grad():
        return _streaming_sdpa(
            query, key, value, attention_mask, q_total, kv_total, scaling
        )


def _streaming_sdpa_core(
    query, key, value, attention_mask, q_total, kv_total, scaling, kv_chunk
):
    """Streaming fwd returning (out, lse). lse = m + log(l), fp32 [1,H,q,1]."""
    outs = []
    lses = []
    for start, stop, mask_chunk in _mask_chunks(
        attention_mask, q_total, kv_total, query.dtype, query.device
    ):
        q = query[:, :, start:stop]
        m_run = None
        l_run = None
        acc = None
        for ks in range(0, kv_total, kv_chunk):
            ke = min(ks + kv_chunk, kv_total)
            k = key[:, :, ks:ke]
            v = value[:, :, ks:ke]
            s = torch.matmul(q, k.transpose(-1, -2)).float() * scaling
            mc = mask_chunk[:, :, :, ks:ke]
            if mc.dtype != s.dtype:
                mc = mc.to(s.dtype)
            s = s + mc
            m_blk = s.amax(dim=-1, keepdim=True)
            if m_run is None:
                m_run = m_blk.clamp(min=-1e30)
                p = torch.exp(s - m_run)
                l_run = p.sum(dim=-1, keepdim=True)
                acc = torch.matmul(p.to(v.dtype), v).float()
            else:
                m_new = torch.maximum(m_run, m_blk)
                alpha = torch.exp(m_run - m_new)
                p = torch.exp(s - m_new)
                l_run = l_run * alpha + p.sum(dim=-1, keepdim=True)
                acc = acc * alpha + torch.matmul(p.to(v.dtype), v).float()
                m_run = m_new
        l_safe = l_run.clamp(min=1e-20)
        outs.append((acc / l_safe).to(query.dtype))
        lses.append((m_run + torch.log(l_safe)).float())
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


class _StreamingSDPAFn(torch.autograd.Function):
    """Flash-style streaming SDPA as an explicit autograd boundary.

    Saved state: q, k, v refs + out + lse (~tens of MB). The full [H,q,kv]
    score/prob tensors are NEVER retained: backward recomputes them block by
    block. Defeats the graph-retention floor that killed launches 13-19.
    """

    @staticmethod
    def forward(
        ctx, query, key, value, attention_mask, q_total, kv_total, scaling, kv_chunk
    ):
        hq, hkv = query.shape[1], key.shape[1]
        rep = hq // hkv if hq != hkv else 1
        if rep > 1:
            key = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)
        out, lse = _streaming_sdpa_core(
            query, key, value, attention_mask, q_total, kv_total, scaling, kv_chunk
        )
        ctx.save_for_backward(query, key, value, out, lse)
        ctx.attention_mask = attention_mask
        ctx.q_total = q_total
        ctx.kv_total = kv_total
        ctx.scaling = scaling
        ctx.kv_chunk = kv_chunk
        ctx.rep = rep
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, out, lse = ctx.saved_tensors
        # SAVED_ALREADY_EXPANDED: fwd saved the GQA-expanded key/value
        # (heads == query heads); do NOT expand again here.
        q_total, kv_total = ctx.q_total, ctx.kv_total
        scaling, kv_chunk, rep = ctx.scaling, ctx.kv_chunk, ctx.rep
        dq = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
        dkf = torch.zeros(key.shape, dtype=torch.float32, device=key.device)
        dvf = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
        for start, stop, mask_chunk in _mask_chunks(
            ctx.attention_mask, q_total, kv_total, query.dtype, query.device
        ):
            q = query[:, :, start:stop].float()
            do = dout[:, :, start:stop].float()
            o = out[:, :, start:stop].float()
            delta = (do * o).sum(dim=-1, keepdim=True)
            lse_c = lse[:, :, start:stop, :]
            for ks in range(0, kv_total, kv_chunk):
                ke = min(ks + kv_chunk, kv_total)
                k = key[:, :, ks:ke].float()
                v = value[:, :, ks:ke].float()
                s = torch.matmul(q, k.transpose(-1, -2)) * scaling
                mc = mask_chunk[:, :, :, ks:ke]
                if mc.dtype != s.dtype:
                    mc = mc.to(s.dtype)
                s = s + mc
                p = torch.exp(s - lse_c)
                dvf[:, :, ks:ke] += torch.matmul(p.transpose(-1, -2), do)
                dp = torch.matmul(do, v.transpose(-1, -2))
                ds = p * (dp - delta)
                dq[:, :, start:stop] += torch.matmul(ds, k) * scaling
                dkf[:, :, ks:ke] += torch.matmul(ds.transpose(-1, -2), q) * scaling
        if rep > 1:
            b, h, n, d = dkf.shape
            dk = dkf.view(b, h // rep, rep, n, d).sum(2)
            b, h, n, d = dvf.shape
            dv = dvf.view(b, h // rep, rep, n, d).sum(2)
        else:
            dk, dv = dkf, dvf
        return (
            dq.to(query.dtype),
            dk.to(key.dtype),
            dv.to(value.dtype),
            None,
            None,
            None,
            None,
            None,
        )
