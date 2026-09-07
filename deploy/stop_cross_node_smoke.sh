#!/bin/bash
# Stop only DFlash2 cross-node smoke training processes in this container.
set -euo pipefail

PIDS=$(pgrep -f 'scripts/train.py.*dflash2_cross_node_' || true)
if [ -n "${PIDS}" ]; then
  kill -KILL ${PIDS}
  sleep 5
fi

if pgrep -f 'scripts/train.py.*dflash2_cross_node_' >/dev/null; then
  echo "Cross-node smoke training process did not exit." >&2
  exit 1
fi

rm -f /tmp/dflash2_cross_node_train.log
