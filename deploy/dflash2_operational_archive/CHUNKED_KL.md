# DFlash2 Chunked KL Technical Note

## Summary

DFlash2 的 chunked KL 与原始 eager KL 优化的是完全相同的
`KL(P || Q)`。它不是近似损失，也没有改变目标分布或梯度方向。

两者的区别是执行方式：原始实现一次构造完整词表上的 FP32 概率和
对数概率；chunked 实现先计算全局归一化常数，再按 2048 个 vocabulary
entries 分块累加。训练时每个 KL chunk 还使用 activation checkpoint，
反向阶段重新计算该 chunk，从而降低峰值显存。

## Original Eager KL

设目标模型 logits 为 `t_i`，草稿模型 logits 为 `l_i`：

```text
P_i = exp(t_i) / sum_j exp(t_j)
Q_i = exp(l_i) / sum_j exp(l_j)
```

标准 forward KL 为：

```text
KL(P || Q) = sum_i P_i * (log(P_i) - log(Q_i))
```

原始 eager 实现等价于：

```python
log_q = log_softmax(logits, dim=-1, dtype=float32)
p = softmax(targets, dim=-1, dtype=float32)
loss = kl_div(log_q, p, reduction="none").sum(dim=-1)
```

代码位于 `src/speculators/losses/eager.py` 的 `kl_div_loss()`。

对 Qwen3.8-27B 的 248K 词表，若 anchor token 数为 4096，完整张量元素数
约为：

```text
4096 * 248320 = 1,017,118,720 elements
```

一个完整 FP32 张量约需要 3.79 GiB。softmax、log-softmax、KL 中间量和
反向激活可能同时存在，因此 eager 方式很容易超过 64 GiB NPU 显存。

## Equivalent Chunked Formula

定义两个全词表 log-normalizer：

```text
Z_P = log(sum_j exp(t_j))
Z_Q = log(sum_j exp(l_j))
```

于是：

```text
log(P_i) = t_i - Z_P
log(Q_i) = l_i - Z_Q
```

代入标准 KL：

```text
KL(P || Q)
  = sum_i P_i * ((t_i - Z_P) - (l_i - Z_Q))
  = sum_i P_i * (t_i - l_i) + Z_Q - Z_P
```

因为 `sum_i P_i = 1`，最后两个归一化项可以移到求和之外。该变形是
严格代数等价，不是近似。

## Chunked Execution

当前实现位于 `src/speculators/models/dflash2/metrics.py`。

### 1. Compute Global Normalizers

`_chunked_logsumexp()` 将词表切成 2048-entry chunks：

```text
chunk_z_k = logsumexp(logits[..., chunk_k])
Z = logaddexp(chunk_z_0, chunk_z_1, ...)
```

`logaddexp` 合并各 chunk 的 log-sum-exp，结果等价于一次性对完整词表执行
`logsumexp`，同时避免构造完整 FP32 softmax。

### 2. Accumulate KL by Vocabulary Chunk

对每个词表 chunk 计算：

```text
P_chunk = exp(target_chunk - Z_P)
partial = sum(P_chunk * (target_chunk - draft_chunk))
```

所有 `partial` 相加后，再加入：

```text
Z_Q - Z_P
```

最终结果仍是逐 token 的 `KL(P || Q)`。

### 3. Activation Checkpointing

训练开启梯度时，每个 KL chunk 通过：

```python
checkpoint(_kl_div_chunk, ..., use_reentrant=False)
```

执行。前向后不长期保存该 chunk 的 FP32 `P_chunk`、差值和乘积；反向时
重新计算它们。这会增加计算量，但显著降低同时驻留的激活内存。

## What Does Not Change

- KL 方向仍是 `KL(target || draft)`，即 `KL(P || Q)`。
- target 和 draft 的 softmax 归一化范围仍是完整词表。
- 没有使用 top-k 截断或概率质量裁剪。
- loss mask、位置衰减和最终 reduction 方式保持不变。
- 对 logits 和 targets 的梯度仍对应同一数学目标。
- DFlash2 selector 的小规模 K-way cross-entropy 不使用该全词表 chunking。

## Practical Differences

| Property | Eager KL | Chunked KL |
| --- | --- | --- |
| Mathematical objective | `KL(P || Q)` | Same |
| Vocabulary normalization | Full vocabulary | Full vocabulary |
| Processing | One full-vocab operation | 2048-entry chunks |
| FP32 temporary memory | Proportional to full vocabulary | Proportional to chunk size |
| Saved training activations | Full eager graph | Chunk activations recomputed |
| Numerical difference | Reference | Floating-point accumulation order only |
| Runtime | Lower compute overhead | More loops and backward recomputation |
| Peak memory | High | Lower |

## Validation Results

使用生产词表大小 248,320，对 `_chunked_kl_div()` 和 eager `kl_div_loss()`
进行了 loss 与 gradient 对照：

```text
maximum loss difference:             8.643e-06
maximum draft-logit gradient error:  6.257e-10
maximum target-logit gradient error: 2.910e-09
all outputs and gradients finite:    true
```

回归测试：

```text
tests/unit/models/test_dflash2_model_definitions.py
  test_chunked_kl_matches_eager_values_and_gradients
```

真实 8K/512-anchor/FSDP smoke 也已完成前向、反向、优化器更新、验证、
checkpoint 保存和 distributed teardown。

## Why FSDP Is Still Required

Chunking 解决的是 KL 概率计算和中间激活的峰值显存，但不能消除完整
logits 本身的梯度。

8K context、`max_anchors=512`、`block_size=8` 时有 4096 个 anchor positions。
完整 BF16 logits gradient 约为：

```text
4096 * 248320 * 2 bytes = 1.895 GiB
```

纯 DDP 在 backward 阶段申请该约 1.90 GiB 张量时仍会 OOM。因此正式配置
同时需要：

- checkpointed chunked KL，降低 loss 中间激活；
- FSDP，分片 draft 参数和 optimizer state；
- Producer `MAX_NUM_SEQS=1`，避免两个 8K hidden-state payload 同时超过
  512 MiB TE send buffer。

这三项解决的是不同的内存问题，不能互相替代。
