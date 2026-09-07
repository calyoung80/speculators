# DFlash2 Operations Archive

This directory is the stable entry-point archive for the validated Qwen3.8-27B
DFlash2 topology. The scripts delegate to the canonical implementation one
level above so operational behavior has a single source of truth.

## Validated Topology

- Producer: `71.10.29.118`, NPU `1,2`, TP=2, port `18000`.
- Consumer: `71.10.29.119`, NPU `2,3`, FSDP=2.
- Hidden-state transport: Mooncake TransferEngine over RoCE.
- Producer model: `/mnt/share/weight/Qwen/Qwen3.8-27B`.
- Producer limits: 8192 context tokens, one sequence at a time.
- Consumer limits: 8192 sequence tokens, 512 anchors, checkpointed 2048-vocab
  chunked KL, FSDP required.
- NPU `0,1` on node 119 are reserved for the file-backend service.

## Entry Points

- `00_create_container.sh`: create a non-production DFlash2 container with
  requested NPU IDs. It refuses to replace `dflash2_train`.
- `01_producer_8k.sh`: inspect, start, or recreate the 8K Producer.
- `02_prepare_formal_data.sh`: turn existing on-policy conversations into the
  formal 500K training dataset. It does not generate new responses.
- `03_start_formal_training.sh`: start the 20K-step formal FSDP run.
- `04_verify_cross_node_te_8k.sh`: verify an 8191-token cross-node TE transfer.
- `05_check_formal_checkpoint.sh`: validate the latest rolling checkpoint.
- `06_evaluate_endpoint.sh`: run the existing endpoint performance evaluator
  after a separate evaluation vLLM service is online.
- `07_evaluation_service.sh`: start, inspect, or stop a post-training DFlash2
  evaluation service. It refuses to start unless the installed vLLM explicitly
  supports both the `dflash2` method and `DFlash2DraftModel` architecture.

Producer restart/recreation, 8K transfer verification, and data preparation
refuse to run while the formal training process is active. Container lifecycle
changes and endpoint evaluation are also deferred until the run is finished.

The currently installed vLLM exposes `dflash` and `dspark`, but not `dflash2`.
Therefore `07_evaluation_service.sh start` currently fails closed instead of
silently evaluating the checkpoint with an incorrect DFlash runtime. Source-level
DFlash2 serving adaptation is still required before post-training acceptance and
throughput evaluation.

## Checkpointing

Formal training saves a rolling recoverable checkpoint every 1000 optimizer
steps. The checkpoint contains model, optimizer, configuration, and
`training_state.json`. A restart from the same save directory resumes from that
state. Do not retain one full checkpoint per interval on this nearly full NFS.

## Current Run

- Run name: `dflash2_qwen38_500k_8k_fsdp_20260907_045632`.
- Save directory: `/mnt/hcs/y00917737/dflash2_formal_500k_8k_fsdp/20260907_045632`.
- Dataset: 500,000 training and 53,830 validation records.
- The current run must not be stopped or recreated while training is active.
- To resume after it exits, run
  `RUN_ID=20260907_045632 bash 03_start_formal_training.sh`.

## Latest Operational Check

At `2026-09-07T10:33:43+08:00`, the formal run was active at
`global_step=843/20000`. Both FSDP training ranks and the torchrun parent were
present. The latest step reported `train/loss=1.672`, `train/accept_rate=0.432`,
`train/error_records=0`, `profile/step_ms=2.12e+04`, and
`profile/tokens_per_s=370.729`; Producer requests returned HTTP 200 and
Mooncake TE reads succeeded. A focused scan found no traceback, OOM, HTTP 400,
or circuit-breaker. The active Producer advertised
`max_model_len=8192` and remained on node 118 NPU `1,2`.

The rolling checkpoint directory did not yet exist because the first save is
scheduled for step 1000. `05_check_formal_checkpoint.sh` reports this state as
`checkpoint_pending` with exit code 3; after a save it validates all four
required files. `/mnt/hcs` had approximately 239 GiB free at the preceding
storage check.

The evaluation startup guard was exercised while training was active and
correctly refused to launch. Post-training evaluation remains additionally
blocked by the missing vLLM DFlash2 serving registration described above.

See `../CROSS_NODE_ROCE_DFLASH2.md` for the full diagnostic history and
numerical validation results.

See `CHUNKED_KL.md` for the mathematical equivalence, implementation details,
memory behavior, and numerical validation of DFlash2 chunked KL.

See `MEMORY_OPTIMIZATION_REFERENCE.md` for the comparison between the current
vocabulary-chunked KL/FSDP baseline and the proposed Q-axis Chunk-Loss, decoder
gradient checkpointing, and Anchor-CP designs. The note records which features
are active, corrects production-vocabulary memory estimates, and defines the
required CP gradient scaling under PyTorch's averaged synchronization.
