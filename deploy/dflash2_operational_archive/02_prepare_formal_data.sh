#!/bin/bash
# Build the formal Arrow dataset from existing Qwen3.8 on-policy outputs.
set -euo pipefail

PASS=${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}
REPO=/mnt/hcs/y00917737/te_dspark_submission/speculators
if sshpass -p "${PASS}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  test_mtp@71.10.29.119 \
  "docker exec dflash2_train pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"; then
  echo "Formal training is active; refusing to compete for Consumer resources." >&2
  exit 1
fi
exec sshpass -p "${PASS}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  test_mtp@71.10.29.119 \
  "docker exec dflash2_train bash -lc 'cd ${REPO} && PYTHONPATH=src:hs_connectors/src bash deploy/prepare_qwen38_formal_500k_data.sh'"
