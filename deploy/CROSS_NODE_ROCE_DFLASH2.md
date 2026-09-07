# Qwen3.8-27B Cross-Node DFlash2 Runbook

## Topology

- Producer: `71.10.29.118`, NPU `1,2`, vLLM TP=2, port `18000`.
- Consumer: `71.10.29.119`, NPU `2,3`, FSDP=2 for the formal run.
- Hidden-state transfer: Mooncake TransferEngine over RoCE.
- Consumer-local DDP remains on its default HCCS path. Do not set
  `HCCL_INTRA_ROCE_ENABLE` or `HCCL_INTRA_PCIE_ENABLE`.
- Shared TE metadata: `/mnt/hcs/y00917737/dflash2_te_meta_118_119`.

## Weight Replication

The W8A8 deployment model replicated from node 119 is located at:

- Source: `119:/mnt/hcs/weights/Qwen3.8-27B-w8a8`.
- Destination: `118:/mnt/share/weight/Qwen3.8-27B-w8a8`.
- Expected source size: approximately 30 GiB.
- Completed verification: 26 regular files; `config.json` SHA-256 is
  `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` on
  both nodes.
- A no-NPU Transformers configuration load on node 118 succeeds
  (`model_type=qwen3_5`, `text_vocab_size=248320`).

Run the fixed replication script from the shared repository:

```bash
bash deploy/sync_qwen38_w8a8_119_to_118_share.sh
```

The script refuses to overwrite an existing destination, checks available
space, transfers the full directory, and validates the copied `config.json`
SHA-256 plus regular-file count.

The BF16 Qwen3.8-27B model was downloaded with:

```bash
modelscope download --model Qwen/Qwen3.8-27B --local_dir /mnt/share/weight
```

ModelScope writes the Qwen files under `/mnt/share/weight/Qwen/Qwen3.8-27B`.
The directory contains the 18 expected `model-*.safetensors` shards, a weight
index, configuration, and tokenizer files. Both node 118 and node 119 can read
it.

`deploy/create_container_dflash2.sh` mounts `/mnt/share/weight` read-only at
the same path in newly created DFlash2 containers. Existing containers are not
changed; recreate the Producer only when it is safe to replace its running
service. `deploy/start_vllm_te.sh` uses the copied W8A8 directory as its
default `MODEL_PATH`; set `MODEL_PATH` explicitly to use another Producer
model.

If a previous transfer stopped before it wrote data, remove only its empty
destination or staging directory before retrying:

```bash
bash deploy/sync_qwen38_w8a8_119_to_118_share.sh cleanup
```

If transfer data completed but publication was interrupted, validate and
publish the existing staging directory without transferring it again:

```bash
bash deploy/sync_qwen38_w8a8_119_to_118_share.sh publish-staged
```

## Training Readiness

The W8A8 model remains unsuitable as a DFlash2 verifier. The downloaded BF16
model is the verifier source. The running 119 Consumer container cannot be
recreated because it also hosts the NPU0,1 file-backend vLLM service. Instead,
`deploy/sync_minimal_verifier_118_to_119.sh` copies only the embedding,
normalization, and LM-head source shards plus configuration/tokenizer files to
`/mnt/hcs/y00917737/dflash2_verifier_minimal`, which the existing container
already mounts.

For DFlash2, `src/speculators/models/dflash2/metrics.py` automatically wraps
`kl_div` in the chunked implementation. The smoke script uses `--loss-fn
kl_div`, so its full-vocabulary KL is processed in vocabulary chunks rather
than materializing the full FP32 vocabulary tensor. The current checkpointed
implementation uses 2048-entry chunks.

During distributed training, step metrics are reduced as one stable,
alphabetically keyed FP32 vector. This prevents rank-specific metric insertion
order from issuing HCCL collectives in different sequences at epoch completion.

Replication does not stop or reconfigure the node 119 NPU0,1 file-backend
vLLM service.

## Smoke Result And Diagnosis (2026-09-06)

The run `dflash2_cross_node_20260906_021438` completed its one training step,
all 541 validation batches, checkpoint save, and distributed-process teardown.
The checkpoint is at
`/mnt/hcs/y00917737/dflash2_cross_node_smoke_ckpt/20260906_021438/0` and
contains the model, optimizer state, metrics, configuration, and training state.

The periodic `400 BadRequest` responses were not a RoCE, Mooncake TE, or HCCL
failure. The Producer has `--max-model-len 4096`, while 473 of the 50,000 source
samples exceed 4095 tokens. Hidden-state extraction requests `max_tokens=1`, so
an untrimmed 16,950-token prompt, for example, was rejected as 16,951 tokens.
The dataset has no `messages` column, so all affected requests use text token IDs
and are covered by the runtime prefix limit.

`ArrowDataset._get_raw_data()` now applies the sampler's runtime truncation
before calling vLLM and reserves one context token for the extraction request:
the request input is limited to `max_len - 1`, and the paired loss mask uses the
same prefix. This prevents repeat retries of requests that cannot fit the
Producer context window.

## 8K Producer Validation

The active node 118 Producer uses TP=2 on NPU `1,2`, port `18000`, and
`--max-model-len 8192`. Its BF16 model path is
`/mnt/share/weight/Qwen/Qwen3.8-27B`; the 18-shard, 104 GiB model's
`config.json` SHA-256 matches the Consumer minimal verifier. It has 26,943 KV
cache tokens, sufficient for 3.29 concurrent 8,192-token requests.

`deploy/test_cross_node_te_8k.py` performed a full cross-node request using
8,191 prompt tokens plus one generated token. Mooncake TE transferred prompt
hidden states of shape `(8191, 6, 5120)` in BF16, totaling 503,255,040 bytes,
from node 118 to node 119 successfully. The consumer verified matching token
and hidden-state lengths and finite values, then deleted the transfer metadata.
This is within the existing 512 MiB TE receive buffer. A 32K context would
exceed that buffer and requires a buffer redesign before serving or training.
For 8K DDP=2 training, set the Producer `MAX_NUM_SEQS=1`: two simultaneous 8K
requests would otherwise exceed the single 512 MiB Producer send buffer.

## Chunked KL Validation

The smoke configuration uses `--loss-fn kl_div`. Although
`--loss-implementation eager` resolves the initial loss configuration, DFlash2
always replaces its unary loss functions in `compute_metrics()` with the
2048-entry vocabulary-chunked implementation. The selector's small K-way CE is
unchanged.

Numerical validation compared `_chunked_kl_div` against the eager KL reference
on FP32 tensors with the production Qwen vocabulary size of 248,320. The maximum
per-position loss difference was `5.782e-06`; maximum draft-logit and target-logit
gradient differences were `6.403e-10` and `3.143e-09`, respectively, and all
outputs and gradients were finite. The regression test
`test_chunked_kl_matches_eager_values_and_gradients` covers multi-chunk loss and
gradient parity. The post-fix cross-node smoke completed one training step and
all 541 validation batches using this loss with finite train loss `6.921` and
validation loss `12.247`.

The full formula derivation and eager/chunked implementation comparison are in
`deploy/dflash2_operational_archive/CHUNKED_KL.md`.

## 8K 512-Anchor Training Validation

At 8K context, DDP replication alone cannot train with `max_anchors=512` on
the two 64 GiB Consumer NPUs. The full-vocabulary KL backward requires a 1.90
GiB logits-gradient allocation after the forward has already consumed 57.79
GiB per rank. `--fsdp-shard` shards the draft model and optimizer across NPU
`2,3`, resolving that allocation pressure.

The validated 8K training configuration is:

- Producer: `MAX_MODEL_LEN=8192`, `MAX_NUM_BATCHED_TOKENS=8192`, and
  `MAX_NUM_SEQS=1` on node 118 NPU `1,2`.
- Consumer: `TOTAL_SEQ_LEN=8192`, `MAX_ANCHORS=512`, `FSDP_SHARD=1` on node
  119 NPU `2,3`.
- Loss: DFlash2 checkpointed chunked KL with 2048-vocabulary chunks. It matches
  eager KL at the 248,320-token production vocabulary with maximum loss error
  `8.643e-06`, draft-logit gradient error `6.257e-10`, and target-logit gradient
  error `2.910e-09`.

Run `dflash2_cross_node_20260906_182105` completed one train step
(`train/loss=7.056`), validation (`val/loss_epoch=13.426`), full checkpoint
save, and distributed teardown. It recorded zero HTTP 400 responses and zero
dropped validation samples. Peak Consumer HBM was approximately 58.9 GiB per
NPU, leaving about 2 GiB headroom.

## Formal Run

The prepared formal dataset is
`/mnt/hcs/y00917737/dflash2_data_27b/formal_500k_qwen38_8k/training_data`.
It contains 553,830 valid 8K-bounded records derived from the existing
Qwen3.8-27B on-policy outputs. The formal launcher derives
`train_data_ratio=0.9028050123684163`, yielding exactly 500,000 training and
53,830 validation records.

Launch with `bash deploy/start_formal_qwen38_500k_train.sh`. It uses 20,000
optimizer steps, FSDP, 8K context, 512 anchors, chunked KL, and a rolling
checkpoint every 1,000 steps. Each periodic save overwrites the current epoch's
checkpoint with the latest `training_state.json`, so restart from the same run
directory resumes from the latest completed step without storing 20 full
checkpoints.

The active formal run is
`dflash2_qwen38_500k_8k_fsdp_20260907_045632`, with checkpoints under
`/mnt/hcs/y00917737/dflash2_formal_500k_8k_fsdp/20260907_045632`. Its
operational archive is `deploy/dflash2_operational_archive/`; use that directory
for container management, Producer management, data preparation, training,
transfer verification, checkpoint inspection, and post-training endpoint
evaluation entry points.

## Operational Archive Audit

The archive scripts were syntax-checked and their non-destructive behavior was
verified while the formal run was active. The 8K Producer readiness check
confirmed the API advertises `max_model_len=8192`. Archive scripts now refuse
to recreate the Producer, create containers, prepare data, run the 8K transfer
test, launch another formal run, or evaluate an endpoint while the formal
training process is active. The orchestration `status` command also uses the
actual BF16 Producer path `/mnt/share/weight/Qwen/Qwen3.8-27B` rather than the
obsolete `/mnt/hcs/weights` path.

The installed vLLM currently registers `dflash` and `dspark`, but not the
`dflash2` method or `DFlash2DraftModel` architecture. The archived evaluation
service entry point checks both registrations and fails closed until a real
DFlash2 serving adaptation is installed; using DFlash as a fallback would omit
the dynamic convolutions and predecessor-conditioned candidate selector.

## Latest Operational Check (2026-09-07)

At `2026-09-07T10:33:43+08:00`, the formal run remained active at
`global_step=843/20000`. The torchrun parent and both FSDP ranks were present.
The latest visible step reported `train/loss=1.672`,
`train/accept_rate=0.432`, `train/error_records=0`,
`profile/step_ms=2.12e+04`, and `profile/tokens_per_s=370.729`. Producer
requests were returning HTTP 200 and the Consumer continued to read
hidden-state payloads successfully over Mooncake TE. A focused scan found no
traceback, OOM, HTTP 400, or circuit-breaker.

The step-1000 rolling checkpoint had not yet been created, which is expected at
step 843. The archived checkpoint checker now distinguishes this pending state
from a corrupt checkpoint and exits with code 3. Shared storage had about
239 GiB available at the preceding storage check. All archived shell entry
points were syntax-checked, marked executable, and passed `git diff --check`;
the evaluation-service guard was also exercised and correctly refused startup
while formal training was active.

## Memory Optimization Reference

The proposed Q-axis Chunk-Loss, decoder gradient checkpointing, and Anchor-CP
designs were compared with the active implementation without modifying the
formal run. The full comparison is archived in
`deploy/dflash2_operational_archive/MEMORY_OPTIMIZATION_REFERENCE.md`.

The active run uses 2048-entry vocabulary-chunked KL with non-reentrant KL
chunk recomputation plus FSDP. It does not enable decoder-layer gradient
checkpointing or Anchor-CP. For the production 248,320 vocabulary, a 64-anchor,
512-Q-row FP32 probability chunk is approximately 485 MiB, not 0.06 GB; the
reference 0.06 GB estimate corresponds roughly to a 32K vocabulary. Anchor-CP
also cannot directly use a SUM-based loss derivation because default PyTorch
DDP/FSDP synchronization averages gradients; a CP implementation must include
the `cp_size` correction and pass CP=1 loss/gradient parity tests.
