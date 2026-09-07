#!/bin/bash
# Run the 8191-token Mooncake TE validation on Consumer NPU2.
set -euo pipefail

PASS=${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}
REPO=/mnt/hcs/y00917737/te_dspark_submission/speculators
if sshpass -p "${PASS}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  test_mtp@71.10.29.119 \
  "docker exec dflash2_train pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"; then
  echo "Formal training is active; refusing to occupy Consumer NPU2." >&2
  exit 1
fi
sshpass -p "${PASS}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  test_mtp@71.10.29.119 \
  "docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 -e TE_META_DIR=/mnt/hcs/y00917737/dflash2_te_meta_118_119 dflash2_train bash -lc 'cd ${REPO} && PYTHONPATH=src:hs_connectors/src python3 deploy/test_cross_node_te_8k.py'"
