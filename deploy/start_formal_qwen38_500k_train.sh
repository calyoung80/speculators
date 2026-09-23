#!/bin/bash
# Start the validated 8K / 512-anchor FSDP training configuration.
set -euxo pipefail

REPO=${REPO:-/mnt/hcs/y00917737/te_dspark_submission/speculators}
DATA_PATH=${DATA_PATH:-/mnt/hcs/y00917737/dflash2_data_27b/formal_500k_qwen38_8k/training_data}
SAVE_ROOT=${SAVE_ROOT:-/mnt/hcs/y00917737/dflash2_formal_500k_8k_fsdp}
TARGET_TRAIN_RECORDS=${TARGET_TRAIN_RECORDS:-500000}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
PASS=${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}
CONSUMER_HOST=${CONSUMER_HOST:-71.10.29.119}
CONSUMER_CONTAINER=${CONSUMER_CONTAINER:-dflash2_train}
DRAFT_CONFIG=${DRAFT_CONFIG:-/mnt/hcs/y00917737/dflash2_draft_config}

if sshpass -p "${PASS}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  "test_mtp@${CONSUMER_HOST}" \
  "docker exec ${CONSUMER_CONTAINER} pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"; then
  echo "A formal DFlash2 run is already active; refusing to start another one." >&2
  exit 1
fi

DATASET_ROWS=$(sshpass -p "${PASS}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  "test_mtp@${CONSUMER_HOST}" \
  "docker exec ${CONSUMER_CONTAINER} python3 -c 'from datasets import load_from_disk; print(len(load_from_disk(\"${DATA_PATH}\")))'")
if [ "${DATASET_ROWS}" -le "${TARGET_TRAIN_RECORDS}" ]; then
  echo "Need more than ${TARGET_TRAIN_RECORDS} prepared records, found ${DATASET_ROWS}." >&2
  exit 1
fi

TRAIN_DATA_RATIO=$(python3 -c "print((${TARGET_TRAIN_RECORDS} + 0.5) / ${DATASET_ROWS})")
export DATA_PATH
export TOTAL_SEQ_LEN=8192
export MAX_ANCHORS=512
export MAX_STEPS=20000
export EPOCHS=1
export TRAIN_DATA_RATIO
export DRAFT_CONFIG
export FSDP_SHARD=1
export CHECKPOINT_STEP_INTERVAL=1000
export SAVE_PATH="${SAVE_ROOT}/${RUN_ID}"
export RUN_NAME="dflash2_qwen38_500k_8k_fsdp_${RUN_ID}"
export RESUME_FROM_CHECKPOINT=1

mkdir -p "${SAVE_PATH}"
exec bash "${REPO}/deploy/start_cross_node_roce_smoke.sh" start-training
