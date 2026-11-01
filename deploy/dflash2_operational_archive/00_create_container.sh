#!/bin/bash
# Create a non-production DFlash2 container: ./00_create_container.sh NAME NPU_IDS
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 NAME NPU_IDS" >&2
  exit 2
fi
if [ "$1" = "dflash2_train" ]; then
  echo "Refusing to replace dflash2_train from the archive entry point." >&2
  exit 1
fi
if sshpass -p "${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}" ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new \
  test_mtp@71.10.29.119 \
  "docker exec dflash2_train pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"; then
  echo "Formal training is active; refusing container lifecycle changes." >&2
  exit 1
fi

ARCHIVE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${ARCHIVE_DIR}/../create_container_dflash2.sh" "$@"
