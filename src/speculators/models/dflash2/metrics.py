"""Runtime-aligned selector loss and metrics for DFlash2.

``eal`` follows the realized greedy selector path. ``accept_len`` remains the
analytical TV-overlap estimate over the unary logits.
"""

import math
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from speculators.losses import (
    LossConfig,
    dflash_loss_decay,
    dpace_loss_decay,
    loss_function,
    tv_loss,
)
from speculators.models.dspark.metrics import compute_metrics as compute_unary_metrics
from speculators.models.metrics import compute_accepted_length_counts

__all__ = [
    "compute_metrics",
    "compute_selector_loss",
    "selector_training_candidates",
]


_VOCAB_CHUNK_SIZE = 2048


def _chunked_logsumexp(
    logits: torch.Tensor, chunk_size: int = _VOCAB_CHUNK_SIZE
) -> torch.Tensor:
    """Compute vocab logsumexp without materializing the full FP32 tensor."""
    normalizer: torch.Tensor | None = None
    for start in range(0, logits.shape[-1], chunk_size):
        stop = min(start + chunk_size, logits.shape[-1])
        chunk_normalizer = torch.logsumexp(logits[..., start:stop].float(), dim=-1)
        if normalizer is None:
            normalizer = chunk_normalizer
        else:
            normalizer = torch.logaddexp(normalizer, chunk_normalizer)
    if normalizer is None:
        raise ValueError("Cannot normalize logits with an empty vocabulary")
    return normalizer


_EPS = 1e-5


def _kl_div_chunk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    log_norm_p: torch.Tensor,
) -> torch.Tensor:
    target_logits = targets.float()
    draft_logits = logits.float()
    target_probs = torch.exp(target_logits - log_norm_p.unsqueeze(-1))
    return (target_probs * (target_logits - draft_logits)).sum(dim=-1)


def _chunked_kl_div(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Chunked KL(P||Q): sum_i P_i * (log P_i - log Q_i).

    Processes vocab in chunks to avoid FP32 OOM on large vocab.
    """
    log_norm_q = _chunked_logsumexp(logits)
    log_norm_p = _chunked_logsumexp(targets)
    result = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    vocab_size = logits.shape[-1]
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        end = min(start + _VOCAB_CHUNK_SIZE, vocab_size)
        args = (
            logits[..., start:end],
            targets[..., start:end],
            log_norm_p,
        )
        if torch.is_grad_enabled():
            result = result + checkpoint(_kl_div_chunk, *args, use_reentrant=False)
        else:
            result = result + _kl_div_chunk(*args)
    result += log_norm_q - log_norm_p
    return result


def _chunked_reverse_kl(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Chunked RKL(Q||P): sum_i Q_i * (log Q_i - log P_i)."""
    log_norm_q = _chunked_logsumexp(logits)
    log_norm_p = _chunked_logsumexp(targets)
    result = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    vocab_size = logits.shape[-1]
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        end = min(start + _VOCAB_CHUNK_SIZE, vocab_size)
        ct = targets[..., start:end].float()
        cl = logits[..., start:end].float()
        q = torch.exp(cl - log_norm_q.unsqueeze(-1))
        result += (q * (cl - ct)).sum(dim=-1)
    result += log_norm_p - log_norm_q
    return result


def _chunked_jsd(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Chunked JSD: 0.5*KL(P||M) + 0.5*KL(Q||M), M=(P+Q)/2."""
    log_norm_q = _chunked_logsumexp(logits)
    log_norm_p = _chunked_logsumexp(targets)
    kl_p_to_m = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    kl_q_to_m = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    vocab_size = logits.shape[-1]
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        end = min(start + _VOCAB_CHUNK_SIZE, vocab_size)
        ct = targets[..., start:end].float()
        cl = logits[..., start:end].float()
        log_p = ct - log_norm_p.unsqueeze(-1)
        log_q = cl - log_norm_q.unsqueeze(-1)
        p = torch.exp(log_p)
        q = torch.exp(log_q)
        log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
        kl_p_to_m += (p * (log_p - log_m)).sum(dim=-1)
        kl_q_to_m += (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_p_to_m + kl_q_to_m)


def _chunked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Chunked hard CE: logsumexp(logits) - logits[argmax(targets)].

    Matches original ce_loss which uses argmax(targets) as hard labels.
    No softmax needed, only gather + chunked logsumexp.
    """
    target_ids = torch.argmax(targets, dim=-1)  # [1, seq_len]
    log_norm_q = _chunked_logsumexp(logits)  # [1, seq_len]
    target_logits = (
        logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).float()
    )  # [1, seq_len]
    return log_norm_q - target_logits


def _chunked_tv(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Chunked TV: 1 - sum_i min(P_i, Q_i)."""
    log_norm_q = _chunked_logsumexp(logits)
    log_norm_p = _chunked_logsumexp(targets)
    overlap = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    vocab_size = logits.shape[-1]
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        end = min(start + _VOCAB_CHUNK_SIZE, vocab_size)
        ct = targets[..., start:end].float()
        cl = logits[..., start:end].float()
        p = torch.exp(ct - log_norm_p.unsqueeze(-1))
        q = torch.exp(cl - log_norm_q.unsqueeze(-1))
        overlap += torch.minimum(p, q).sum(dim=-1)
    return 1.0 - overlap


def _chunked_nla(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Chunked NLA: -log(alpha) where alpha = sum_i min(P_i, Q_i)."""
    log_norm_q = _chunked_logsumexp(logits)
    log_norm_p = _chunked_logsumexp(targets)
    overlap = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    vocab_size = logits.shape[-1]
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        end = min(start + _VOCAB_CHUNK_SIZE, vocab_size)
        ct = targets[..., start:end].float()
        cl = logits[..., start:end].float()
        p = torch.exp(ct - log_norm_p.unsqueeze(-1))
        q = torch.exp(cl - log_norm_q.unsqueeze(-1))
        overlap += torch.minimum(p, q).sum(dim=-1)
    return -torch.log(overlap.clamp_min(_EPS))


def _chunked_lk_hybrid(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eta: float = 3.0,
) -> torch.Tensor:
    """Chunked LK hybrid: lambda*KL + (1-lambda)*TV, lambda=exp(-eta*alpha)."""
    log_norm_q = _chunked_logsumexp(logits)
    log_norm_p = _chunked_logsumexp(targets)
    overlap = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32)
    vocab_size = logits.shape[-1]
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        end = min(start + _VOCAB_CHUNK_SIZE, vocab_size)
        ct = targets[..., start:end].float()
        cl = logits[..., start:end].float()
        p = torch.exp(ct - log_norm_p.unsqueeze(-1))
        q = torch.exp(cl - log_norm_q.unsqueeze(-1))
        overlap += torch.minimum(p, q).sum(dim=-1)
    tv = 1.0 - overlap
    kl = _chunked_kl_div(logits, targets)
    weight = torch.exp(-eta * overlap.detach())
    return weight * kl + (1.0 - weight) * tv


_CHUNKED_BY_NAME: dict[str, Callable[..., torch.Tensor]] = {
    "kl_div": _chunked_kl_div,
    "rkl": _chunked_reverse_kl,
    "jsd": _chunked_jsd,
    "ce": _chunked_ce,
    "tv": _chunked_tv,
    "nla": _chunked_nla,
    "lk_hybrid": _chunked_lk_hybrid,
}


def _wrap_chunked(loss_config: LossConfig) -> LossConfig:
    """Replace loss fns in loss_config with chunked versions (DFlash2 only)."""
    return {
        name: (_CHUNKED_BY_NAME.get(name, fn), weight)
        for name, (fn, weight) in loss_config.items()
    }


def _wrap_tv_chunked(tv_loss_fn):
    """Wrap tv_loss_fn with chunked version."""
    return _CHUNKED_BY_NAME.get("tv", tv_loss_fn)


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [*, top_k]
    target_ids: torch.Tensor,  # [*]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build top-k selector candidates, injecting a missing target at rank K.

    Unary top-k is the serving candidate set. Training replaces its weakest
    candidate only when the hard target is absent, so every position has a
    well-defined K-way cross-entropy label without expanding the selector to the
    full vocabulary.

    Returns (training_candidate_ids, target_positions, contains_target) where
    contains_target is a boolean mask indicating positions where the target was
    already present in the original unary top-k.
    """
    top_k = candidate_ids.shape[-1]
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)
    target_positions = torch.where(
        contains_target,
        target_positions,
        top_k - 1,
    )

    training_candidate_ids = candidate_ids.clone()
    training_candidate_ids[..., -1] = torch.where(
        contains_target,
        training_candidate_ids[..., -1],
        target_ids,
    )
    return training_candidate_ids, target_positions, contains_target


def _candidate_cross_entropy(
    logits: torch.Tensor,  # [*, top_k]
    target_positions: torch.Tensor,  # [*]
) -> torch.Tensor:
    return functional.cross_entropy(
        logits.flatten(0, -2),
        target_positions.flatten(),
        reduction="none",
    ).view_as(target_positions)


def compute_selector_loss(
    candidate_logits: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_positions: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    *,
    gamma: float,
    per_position_loss_weight: str,
    dpace_alpha: float,
    sample_from_anchor: bool = False,
) -> torch.Tensor:
    """Compute teacher-forced hard CE over the runtime-sized candidate set."""
    pos_idx = (
        torch.arange(candidate_logits.shape[1], device=candidate_logits.device)
        % block_size
    ).unsqueeze(0)
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay,
            gamma=gamma,
            sample_from_anchor=sample_from_anchor,
        )
    return loss_function(
        candidate_logits,
        target_positions,
        loss_mask,
        pos_idx,
        loss_fn=_candidate_cross_entropy,
        decay_fn=decay_fn,
    )


def compute_metrics(
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    training_candidate_ids: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    candidate_logits: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_positions: torch.Tensor,  # [1, num_anchors*block_size]
    contains_target: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    top_k: int,
    sample_from_anchor: bool = False,
    *,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    selector_loss_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    loss_chunk_size: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine the unary DFlash objective with a K-way selector objective."""
    chunked_loss_config = _wrap_chunked(loss_config)
    chunked_tv_loss_fn = _wrap_tv_chunked(tv_loss_fn)
    unary_loss, metrics = compute_unary_metrics(
        unary_logits,
        targets,
        None,
        loss_mask,
        block_size,
        loss_config=chunked_loss_config,
        tv_loss_fn=chunked_tv_loss_fn,
        gamma=gamma,
        confidence_head_alpha=0.0,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
        loss_chunk_size=loss_chunk_size,
    )
    selector_loss = compute_selector_loss(
        candidate_logits,
        target_positions,
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )
    loss = unary_loss + selector_loss_alpha * selector_loss

    one = torch.ones((), device=unary_logits.device)
    # Token-weighted denominator from the unary objective (see dspark
    # compute_metrics): makes the cross-rank reduced losses true global
    # token-weighted means instead of means of per-rank means.
    unary_token_count = metrics.pop("loss_total", None)
    metrics["unary_loss_sum"] = unary_loss.detach().clone()
    metrics["unary_loss_total"] = (
        unary_token_count if unary_token_count is not None else one
    )
    metrics["selector_loss_sum"] = selector_loss.detach().clone()
    metrics["selector_loss_total"] = one.clone()
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = (
        unary_token_count if unary_token_count is not None else one.clone()
    )
    if unary_token_count is not None:
        # Live token count for the trainer's gradient-normalization
        # all-reduce (popped before metric reduction; see trainer.py).
        metrics["__loss_token_count__"] = unary_token_count

    with torch.no_grad():
        target_ids = targets.argmax(dim=-1)
        valid = loss_mask.to(torch.bool)
        valid_float = valid.to(unary_logits.dtype)
        valid_total = valid_float.sum()

        metrics[f"unary_candidate_recall_at_{top_k}_sum"] = (
            contains_target.to(valid_float.dtype) * valid_float
        ).sum()
        metrics[f"unary_candidate_recall_at_{top_k}_total"] = valid_total

        target_log_normalizer = _chunked_logsumexp(targets)
        candidate_target_logits = targets.gather(-1, training_candidate_ids).float()
        candidate_mass = torch.exp(
            torch.logsumexp(candidate_target_logits, dim=-1) - target_log_normalizer
        )
        metrics[f"unary_candidate_target_mass_at_{top_k}_sum"] = (
            candidate_mass * valid_float
        ).sum()
        metrics[f"unary_candidate_target_mass_at_{top_k}_total"] = valid_total.clone()

        teacher_forced_ids = training_candidate_ids.gather(
            -1, candidate_logits.detach().argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        serving_valid = valid_float * contains_target.to(valid_float.dtype)
        serving_total = serving_valid.sum()
        metrics["teacher_forced_selector_acc_sum"] = (
            teacher_forced_ids.eq(target_ids).to(valid_float.dtype) * serving_valid
        ).sum()
        metrics["teacher_forced_selector_acc_total"] = serving_total

        num_blocks = unary_logits.shape[1] // block_size
        contains_target_blocks = contains_target.view(num_blocks, block_size)
        valid_blocks = valid.view(num_blocks, block_size)

        # Teacher-forced predecessor tokens are exact while the greedy path is
        # alive. Gate on the original unary candidate set because training may
        # inject a missing target that would not be available during serving.
        start_pos = 0 if sample_from_anchor else 1
        selector_correct = teacher_forced_ids.eq(target_ids) & contains_target
        eal_sum, eal_total = compute_accepted_length_counts(
            selector_correct.view(num_blocks, block_size)[:, start_pos:],
            valid_blocks[:, start_pos:],
        )
        metrics["eal_sum"] = eal_sum
        metrics["eal_total"] = eal_total

        oracle_alive = torch.ones(
            num_blocks, dtype=torch.bool, device=unary_logits.device
        )
        oracle_accepted_length = torch.ones(
            num_blocks, dtype=torch.float32, device=unary_logits.device
        )
        for position in range(1, block_size):
            oracle_alive = (
                oracle_alive
                & valid_blocks[:, position]
                & contains_target_blocks[:, position]
            )
            oracle_accepted_length += oracle_alive.to(oracle_accepted_length.dtype)
        block_valid = valid_blocks[:, 1:].any(dim=-1)
        block_total = block_valid.sum().to(torch.float32)
        metrics[f"unary_top_{top_k}_oracle_accepted_length_sum"] = (
            oracle_accepted_length * block_valid
        ).sum()
        metrics[f"unary_top_{top_k}_oracle_accepted_length_total"] = block_total

    return loss, metrics
