#!/bin/bash
# Build the 500K-train/55,556-validation Qwen3.8-27B dataset from existing
# on-policy conversations. Rendering derives token boundaries; it never generates
# new answers.
set -euxo pipefail

REPO=/mnt/hcs/y00917737/te_dspark_submission/speculators
DATA_ROOT=/mnt/hcs/y00917737/dflash2_data_27b/formal_500k_qwen38_8k
SOURCE_PATH=${SOURCE_PATH:-${DATA_ROOT}/source.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-${DATA_ROOT}/training_data}
VERIFIER_PATH=${VERIFIER_PATH:-/mnt/hcs/y00917737/dflash2_verifier_minimal}
RENDER_ENDPOINT=${RENDER_ENDPOINT:-http://71.10.29.118:18000}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-555556}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
NUM_WORKERS=${NUM_WORKERS:-8}

test -f "${SOURCE_PATH}"
test -f "${VERIFIER_PATH}/config.json"
if [ -e "${OUTPUT_PATH}" ]; then
  echo "Refusing to overwrite existing training dataset: ${OUTPUT_PATH}" >&2
  exit 1
fi

cd "${REPO}"
exec python3 scripts/prepare_data.py \
  --model "${VERIFIER_PATH}" \
  --data "${SOURCE_PATH}" \
  --output "${OUTPUT_PATH}" \
  --seq-length "${SEQ_LENGTH}" \
  --max-samples "${TOTAL_SAMPLES}" \
  --render-endpoint "${RENDER_ENDPOINT}" \
  --num-preprocessing-workers "${NUM_WORKERS}" \
  --minimum-valid-tokens 1
