# DFlash2 Memory Optimization Reference

## Scope

本文对照以下三类显存优化思路与当前 Qwen3.8-27B DFlash2 正式训练实现：

1. 沿 anchor/Q 维切分的 Chunk-Loss。
2. Decoder 层梯度重计算。
3. 只切 anchor、不切 verifier 序列的 Anchor-CP。

参考材料中的 16K/1024-anchor 显存与耗时数据作为外部参考，尚未在当前
Ascend、Qwen3.8-27B、248,320 词表环境中复测。当前 20K-step 正式训练不做
热修改，以下结论用于后续独立 smoke 和下一轮训练设计。

## Current Formal Run

当前正式配置为 8K context、512 anchors、block size 8，因此全词表位置数为：

```text
Q = 512 * 8 = 4096
vocab = 248320
```

当前状态：

| Capability | Formal run status | Implementation |
| --- | --- | --- |
| Vocabulary-chunked KL | Enabled | 2048 vocabulary entries per chunk |
| KL chunk recomputation | Enabled | non-reentrant `torch.utils.checkpoint` |
| Decoder-layer checkpointing | Not enabled | no trainer flag or enable call |
| Anchor-CP | Not enabled | distributed initialization uses `sp_size=1` |
| Parameter/optimizer sharding | Enabled | FSDP across Consumer NPU 2,3 |

The active code paths are:

- `src/speculators/models/dflash2/metrics.py`: exact vocabulary-chunked KL,
  TV/accept-rate calculation, and KL chunk recomputation.
- `src/speculators/models/dflash/core.py`: builds all selected anchor blocks and
  full verifier-derived context on each rank.
- `src/speculators/train/trainer.py`: applies FSDP but does not enable model-layer
  gradient checkpointing.
- `src/speculators/train/cli.py`: calls `maybe_setup_distributed()` with the
  default `sp_size=1`.

## Scheme 1: Chunk-Loss

### Reference design

The reference implementation chunks the independent Q/anchor-token dimension:

```text
[1, Q, vocab] -> [1, anchor_chunk_size * block_size, vocab]
```

This is mathematically valid because softmax, minimum, vocabulary reduction,
loss masking, and position decay are all independent between Q rows. For the
current `loss_function()` reduction, multiplying each chunk mean by its valid
mask count divided by the global valid mask count reconstructs the same scalar
loss.

### Current design

The formal run chunks the vocabulary dimension instead:

```text
[1, Q, vocab] -> [1, Q, 2048]
```

It first computes exact full-vocabulary log-normalizers using chunked
`logsumexp` plus `logaddexp`, then accumulates the KL contribution from each
vocabulary chunk. This preserves full-vocabulary normalization and the exact
`KL(target || draft)` objective without top-k truncation.

### Memory comparison for the production vocabulary

For Q=4096 and vocab=248,320:

```text
full FP32 tensor:
4096 * 248320 * 4 bytes = 3.79 GiB

64-anchor Q chunk, block size 8, 512 Q rows:
512 * 248320 * 4 bytes = 485 MiB per FP32 probability tensor

2048-entry vocabulary chunk:
4096 * 2048 * 4 bytes = 32 MiB per FP32 chunk tensor
```

Therefore the reference value `0.06 GB` for a 512-row Q chunk does not apply to
the production 248K vocabulary; it corresponds approximately to a 32K
vocabulary. Two 248K probability tensors alone are about 970 MiB before the
minimum/reduction workspace. The current vocabulary chunk has a much smaller
per-tensor FP32 footprint, at the cost of more loop iterations and global
normalizer passes.

### Accuracy and implementation notes

- Both axes are mathematically exact. Floating-point accumulation order can
  still produce small numerical differences, so “zero impact” should mean
  objective equivalence rather than guaranteed bitwise identity.
- The formal KL path has production-vocabulary loss and gradient parity tests.
  Other chunked loss names in `dflash2/metrics.py` have not received the same
  production validation and are not used by the formal run.
- The reference `_chunked_compound_loss` must accumulate every named term over
  all chunks. Returning only the last chunk's term value would make multi-loss
  metrics incorrect even if the total loss is correct.
- Adding outer Q chunking around the current inner vocabulary chunking would
  create a two-dimensional nested loop. It may reduce temporary memory further
  but can substantially increase launch/recomputation overhead; it should not be
  enabled without an isolated benchmark.
- `@torch.compiler.disable` is not currently required for this Ascend training
  path because `conditional_torch_compile` only compiles when
  `torch.cuda.is_available()` is true. It should be reconsidered if the compile
  policy is later extended to NPU.

### Decision

Keep the validated 2048-entry vocabulary-chunked KL for the active run. Treat
Q-axis Chunk-Loss as an optional future implementation for generic compound
losses or accept-rate metrics, not as a required replacement during training.

## Scheme 2: Decoder Gradient Checkpointing

`Qwen3DFlashDecoderLayer` inherits Transformers'
`GradientCheckpointingLayer`, and DFlash2 inherits that decoder layer. However,
the capability is dormant in the current training stack:

- `DFlashDraftModel` does not declare `supports_gradient_checkpointing = True`.
- `TrainerConfig` and the CLI have no gradient-checkpointing option.
- `Trainer.setup_model()` does not call `gradient_checkpointing_enable()`.
- The active launcher contains no gradient-checkpointing argument.

The checkpoint calls currently seen in `dflash2/metrics.py` recompute KL loss
chunks only; they do not recompute attention, MLP, or dynamic-convolution
activations.

A future implementation should:

1. Declare support on `DFlashDraftModel` so DFlash2 and DSpark inherit it.
2. Add a resolved configuration and CLI flag with a default of disabled.
3. Call `gradient_checkpointing_enable()` before DDP/FSDP wrapping.
4. Verify non-reentrant backward through DFlash2 attention and both dynamic
   convolution wrappers.
5. Run loss/gradient parity, one-step optimizer, validation, checkpoint, and
   resume tests before using it in a formal run.

This is the lowest-complexity next optimization to test for 16K/1024 anchors,
but the external `41G / 35.1s` result is not yet a local measurement.

## Scheme 3: Anchor-CP

### Why it is plausible

DFlash2 Q rows are anchor-derived noise blocks, while the context K/V source is
the full verifier hidden-state sequence. It is therefore possible in principle
for CP ranks to share the same full verifier input and process disjoint anchor
subsets. This can reduce noise embeddings, draft activations, logits, targets,
and selector tensors approximately in proportion to CP size.

### Why it is not currently available

- `maybe_setup_distributed()` is called with `sp_size=1`; no `--cp-size` or
  equivalent training option is wired.
- The current multipack sampler gives different dataset samples to data-parallel
  ranks. It does not establish the same batch plus disjoint anchors required by
  Anchor-CP.
- Existing SP process-group scaffolding is not connected to DFlash anchor
  selection, loss scaling, validation aggregation, or checkpoint tests.
- If SP size were enabled today, ranks with the same DP rank would independently
  fetch the same hidden states; the documented scatter optimization is not
  implemented in the dataloader.
- FSDP currently uses its default world process group, so CP and DP gradient
  semantics must be designed together rather than adding only an anchor slice.

### Correct loss scaling with PyTorch gradient averaging

The reference derivation assumes DDP/FSDP synchronizes gradients with SUM.
PyTorch DDP and normal FSDP semantics average gradients across participating
ranks. To preserve a per-DP-sample anchor mean with CP size `C`, define on each
CP rank:

```text
local_loss = S_local / N_local
N_group = all_reduce_cp(N_local, SUM)
backward_loss = C * (N_local / N_group) * local_loss
```

After the default average over `DP * C` ranks, this yields the average across DP
samples of each sample's exact full-anchor mean. Omitting the factor `C` scales
the gradient down by `1/C`.

Do not use an in-place `dist.all_reduce(loss)` on the differentiable loss as a
substitute for autograd-aware synchronization. Reduce detached numerator/count
values for logging, and let DDP/FSDP synchronize the correctly scaled local
gradients. If a custom SUM reduction hook is used, the scaling formula must be
adjusted and revalidated.

### Required validation

Anchor-CP needs a separate feature branch and smoke run covering:

1. Identical full verifier inputs inside each CP group.
2. Disjoint, exhaustive anchor assignment with padding handled once.
3. Attention-mask and dynamic-convolution block boundaries for local Q rows.
4. Loss and parameter-gradient parity against CP=1.
5. FSDP gradient scaling for CP-only and combined DP+CP topologies.
6. Validation metric aggregation, checkpoint save, and exact resume behavior.
7. Producer load when hidden states are broadcast instead of fetched repeatedly.

### Decision

Anchor-CP is promising but is not a launcher-only option in the current code.
It is the highest-risk of the three changes and should be attempted only after
decoder checkpointing is independently validated.

## Recommended Order

1. Do not modify or restart the active 20K-step 8K/512/FSDP run.
2. Preserve the current vocabulary-chunked KL as the known-good baseline.
3. On separate resources, add decoder-layer checkpointing behind a disabled-by-
   default flag and validate at 8K/512 before trying 16K/1024.
4. Benchmark Q-axis Chunk-Loss only if generic compound losses or accept-rate
   memory remain limiting; compare it against the current vocabulary chunking
   rather than stacking both by default.
5. Implement Anchor-CP last, with explicit process groups, same-batch delivery,
   `cp_size`-aware gradient scaling, and CP=1 parity tests.
