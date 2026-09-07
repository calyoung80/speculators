#!/bin/bash
# Manage the validated 8K Producer: status | start | recreate.
set -euo pipefail

ARCHIVE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ORCHESTRATOR="${ARCHIVE_DIR}/../start_cross_node_roce_smoke.sh"
ACTION=${1:-status}

formal_running() {
  sshpass -p "${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}" ssh \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o StrictHostKeyChecking=accept-new \
    test_mtp@71.10.29.119 \
    "docker exec dflash2_train pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"
}

case "${ACTION}" in
  status)
    exec bash "${ORCHESTRATOR}" status
    ;;
  start)
    if formal_running; then
      echo "Formal training is active; refusing to restart its Producer." >&2
      exit 1
    fi
    exec env MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash "${ORCHESTRATOR}" start-producer
    ;;
  recreate)
    if formal_running; then
      echo "Formal training is active; refusing to recreate its Producer." >&2
      exit 1
    fi
    env MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash "${ORCHESTRATOR}" prepare-producer
    env MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash "${ORCHESTRATOR}" start-producer
    exec env MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash "${ORCHESTRATOR}" wait-producer
    ;;
  *)
    echo "Usage: $0 {status|start|recreate}" >&2
    exit 2
    ;;
esac
