#!/bin/bash
# Evaluate an already-running OpenAI-compatible DFlash2 serving endpoint.
set -euo pipefail

ARCHIVE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${ARCHIVE_DIR}/../.." && pwd)
TARGET=${TARGET:-http://127.0.0.1:8000/v1}
MODE=${1:-throughput}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO}/eval_results/$(date +%Y%m%d_%H%M%S)}

if sshpass -p "${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  test_mtp@71.10.29.119 \
  "docker exec dflash2_train pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"; then
  echo "Formal training is active; endpoint evaluation is deferred." >&2
  exit 1
fi

cd "${REPO}/scripts/evaluate"
exec python3 evaluate.py "${MODE}" \
  --target "${TARGET}" \
  --output-dir "${OUTPUT_DIR}" \
  "${@:2}"
