#!/bin/bash
# Verify the current recoverable checkpoint without touching running training.
set -euo pipefail

RUN_DIR=${1:-/mnt/hcs/y00917737/dflash2_formal_500k_8k_fsdp/20260907_045632}
CHECKPOINT="${RUN_DIR}/0"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "checkpoint_pending=${CHECKPOINT}" >&2
  echo "The first rolling checkpoint is expected after global_step=1000." >&2
  exit 3
fi

missing=0
for name in model.safetensors optimizer_state_dict.pt config.json training_state.json; do
  if [[ ! -s "${CHECKPOINT}/${name}" ]]; then
    echo "checkpoint_missing=${CHECKPOINT}/${name}" >&2
    missing=1
  fi
done
if (( missing )); then
  exit 1
fi

python3 -c "import json; print(json.load(open('${CHECKPOINT}/training_state.json')))"
du -sh "${CHECKPOINT}"
echo "checkpoint_ok=${CHECKPOINT}"
