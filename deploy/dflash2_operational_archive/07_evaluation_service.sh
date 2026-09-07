#!/bin/bash
# Manage a post-training DFlash2 evaluation service on node 119.
set -euo pipefail

PASS=${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}
HOST=${EVAL_HOST:-71.10.29.119}
CONTAINER=${EVAL_CONTAINER:-dflash2_eval}
TARGET_MODEL=${TARGET_MODEL:-/mnt/share/weight/Qwen/Qwen3.8-27B}
RUN_DIR=${RUN_DIR:-/mnt/hcs/y00917737/dflash2_formal_500k_8k_fsdp/20260907_045632}
CHECKPOINT=${CHECKPOINT:-${RUN_DIR}/0}
PORT=${PORT:-18080}
TP_SIZE=${TP_SIZE:-2}
NPU_IDS=${NPU_IDS:-2,3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
ACTION=${1:-status}

remote() {
  sshpass -p "${PASS}" ssh \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o StrictHostKeyChecking=accept-new \
    "test_mtp@${HOST}" "$1"
}

formal_running() {
  remote "docker exec dflash2_train pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"
}

case "${ACTION}" in
  status)
    remote "docker exec ${CONTAINER} sh -c 'pgrep -af \"vllm.*--port ${PORT}\" || true; curl -sf --max-time 5 http://127.0.0.1:${PORT}/v1/models || true; tail -80 /tmp/dflash2_eval.log 2>/dev/null || true'"
    ;;
  stop)
    remote "docker exec ${CONTAINER} sh -c 'pkill -f \"vllm.*--port ${PORT}\" || true'"
    ;;
  start)
    if formal_running; then
      echo "Formal training is active; evaluation service startup is deferred." >&2
      exit 1
    fi
    remote "docker inspect ${CONTAINER} >/dev/null"
    remote "docker exec ${CONTAINER} test -f ${TARGET_MODEL}/config.json"
    remote "docker exec ${CONTAINER} test -f ${CHECKPOINT}/config.json"
    if ! remote "docker exec ${CONTAINER} sh -c 'grep -q \"@register_speculator(\\\"dflash2\\\")\" /vllm-workspace/vllm/vllm/transformers_utils/configs/speculators/algos.py && grep -q \"DFlash2DraftModel\" /vllm-workspace/vllm/vllm/model_executor/models/registry.py'"; then
      echo "Installed vLLM lacks DFlash2 config/architecture registration; refusing to start an incorrect DFlash fallback." >&2
      exit 1
    fi
    remote "docker exec -d -e ASCEND_RT_VISIBLE_DEVICES=${NPU_IDS} ${CONTAINER} bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; VLLM_PLUGINS=ascend exec vllm serve ${TARGET_MODEL} --tensor-parallel-size ${TP_SIZE} --port ${PORT} --max-model-len ${MAX_MODEL_LEN} --trust-remote-code --speculative-config '\''{\"method\":\"dflash2\",\"model\":\"${CHECKPOINT}\",\"num_speculative_tokens\":7}'\'' > /tmp/dflash2_eval.log 2>&1'"
    for _ in $(seq 1 180); do
      if remote "docker exec ${CONTAINER} curl -sf --max-time 5 http://127.0.0.1:${PORT}/v1/models >/dev/null"; then
        exec "$0" status
      fi
      sleep 10
    done
    remote "docker exec ${CONTAINER} tail -120 /tmp/dflash2_eval.log"
    exit 1
    ;;
  *)
    echo "Usage: $0 {start|status|stop}" >&2
    exit 2
    ;;
esac
