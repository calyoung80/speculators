#!/bin/bash
# Run the fixed cross-node DFlash2 smoke topology.
# Producer: 71.10.29.118 NPU 1,2, TP=2. Consumer: 71.10.29.119 NPU 2,3, DDP=2.
set -euo pipefail

PASS=${D2_SSH_PASSWORD:?Set D2_SSH_PASSWORD before running this script}
USER=test_mtp
PRODUCER_HOST=71.10.29.118
CONSUMER_HOST=71.10.29.119
PRODUCER_PORT=18000
PRODUCER_CONTAINER=dflash2_roce_producer_118
CONSUMER_CONTAINER=dflash2_train
REPO=/mnt/hcs/y00917737/te_dspark_submission/speculators
PRODUCER_LOG=/tmp/dflash2_roce_producer.log
TRAIN_LOG=/tmp/dflash2_cross_node_train.log
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MODEL_PATH=${MODEL_PATH:-/mnt/share/weight/Qwen/Qwen3.8-27B}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
TOTAL_SEQ_LEN=${TOTAL_SEQ_LEN:-4096}
MAX_ANCHORS=${MAX_ANCHORS:-64}
MAX_STEPS=${MAX_STEPS:-1}
EPOCHS=${EPOCHS:-1}
TRAIN_DATA_RATIO=${TRAIN_DATA_RATIO:-0.9}
FSDP_SHARD=${FSDP_SHARD:-0}
CHECKPOINT_STEP_INTERVAL=${CHECKPOINT_STEP_INTERVAL:-0}
DATA_PATH=${DATA_PATH:-}
SAVE_PATH=${SAVE_PATH:-}
RUN_NAME=${RUN_NAME:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-0}

remote() {
  sshpass -p "${PASS}" ssh \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=15 \
    "${USER}@$1" "${2}" 2>/dev/null
}

check_producer_npu() {
  remote "${PRODUCER_HOST}" \
    "npu-smi info | grep 'No running processes found in NPU 1' && npu-smi info | grep 'No running processes found in NPU 2'"
}

check_producer_port() {
  if remote "${PRODUCER_HOST}" \
    "ss -ltn | grep -q ':${PRODUCER_PORT} '"; then
    echo "Producer port ${PRODUCER_PORT} is already in use on ${PRODUCER_HOST}." >&2
    return 1
  fi
}

remove_previous_producer() {
  remote "${PRODUCER_HOST}" \
    "docker stop ${PRODUCER_CONTAINER} >/dev/null 2>&1 || true; docker rm ${PRODUCER_CONTAINER} >/dev/null 2>&1 || true"
}

prepare_producer() {
  remove_previous_producer
  check_producer_npu
  check_producer_port
  remote "${PRODUCER_HOST}" \
    "cd ${REPO} && bash deploy/create_container_dflash2.sh ${PRODUCER_CONTAINER} 1,2"
}

start_producer() {
  remote "${PRODUCER_HOST}" \
    "docker exec ${PRODUCER_CONTAINER} sh -c 'rm -rf /mnt/hcs/y00917737/dflash2_te_meta_118_119 && mkdir -p /mnt/hcs/y00917737/dflash2_te_meta_118_119' && docker exec -d -e MODEL_PATH=${MODEL_PATH} -e MAX_MODEL_LEN=${MAX_MODEL_LEN} -e MAX_NUM_BATCHED_TOKENS=${MAX_MODEL_LEN} -e MAX_NUM_SEQS=${MAX_NUM_SEQS} ${PRODUCER_CONTAINER} bash ${REPO}/deploy/start_vllm_te.sh 1,2 2 ${PRODUCER_PORT} ${PRODUCER_HOST}"
}

wait_for_producer() {
  for _ in $(seq 1 180); do
    if remote "${PRODUCER_HOST}" \
      "docker exec ${PRODUCER_CONTAINER} sh -c \"pgrep -af 'vllm.entrypoints.cli.main' | grep -q -- '--port ${PRODUCER_PORT}'\"" \
      && remote "${PRODUCER_HOST}" \
        "docker exec ${PRODUCER_CONTAINER} curl -sf --max-time 5 http://127.0.0.1:${PRODUCER_PORT}/v1/models | grep -q Qwen3.8-27B" \
      && remote "${PRODUCER_HOST}" \
        "docker exec ${PRODUCER_CONTAINER} curl -sf --max-time 5 http://127.0.0.1:${PRODUCER_PORT}/v1/models | grep -q '\"max_model_len\":${MAX_MODEL_LEN}'"; then
      remote "${CONSUMER_HOST}" \
        "docker exec ${CONSUMER_CONTAINER} curl -sf --max-time 5 http://${PRODUCER_HOST}:${PRODUCER_PORT}/v1/models"
      return
    fi
    if ! remote "${PRODUCER_HOST}" \
      "docker exec ${PRODUCER_CONTAINER} pgrep -f 'vllm.entrypoints.cli.main' >/dev/null"; then
      remote "${PRODUCER_HOST}" \
        "docker exec ${PRODUCER_CONTAINER} tail -120 ${PRODUCER_LOG}"
      return 1
    fi
    sleep 10
  done
  remote "${PRODUCER_HOST}" \
    "docker exec ${PRODUCER_CONTAINER} tail -80 ${PRODUCER_LOG}"
  return 1
}

stop_stale_training() {
  remote "${CONSUMER_HOST}" \
    "docker exec ${CONSUMER_CONTAINER} bash ${REPO}/deploy/stop_cross_node_smoke.sh"
}

start_training() {
  if remote "${CONSUMER_HOST}" \
    "docker exec ${CONSUMER_CONTAINER} pgrep -f 'scripts/train.py.*dflash2_qwen38_500k_8k_fsdp_' >/dev/null"; then
    echo "A formal DFlash2 run is already active; refusing to start another training job." >&2
    return 1
  fi
  stop_stale_training
  remote "${CONSUMER_HOST}" \
    "docker exec -d -e TOTAL_SEQ_LEN=${TOTAL_SEQ_LEN} -e MAX_ANCHORS=${MAX_ANCHORS} -e MAX_STEPS=${MAX_STEPS} -e EPOCHS=${EPOCHS} -e TRAIN_DATA_RATIO=${TRAIN_DATA_RATIO} -e FSDP_SHARD=${FSDP_SHARD} -e CHECKPOINT_STEP_INTERVAL=${CHECKPOINT_STEP_INTERVAL} -e DATA_PATH=${DATA_PATH} -e SAVE_PATH=${SAVE_PATH} -e RUN_NAME=${RUN_NAME} -e RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT} ${CONSUMER_CONTAINER} bash ${REPO}/deploy/start_train_te_dflash2.sh ${PRODUCER_HOST} ${PRODUCER_PORT} 2,3 ${CONSUMER_HOST}; for _ in 1 2 3 4 5 6; do docker exec ${CONSUMER_CONTAINER} pgrep -af '[s]cripts/train.py' && exit 0; sleep 5; done; docker exec ${CONSUMER_CONTAINER} cat ${TRAIN_LOG}; exit 1"
}

sync_verifier() {
  bash "${REPO}/deploy/sync_minimal_verifier_118_to_119.sh"
}

wait_for_training() {
  for _ in $(seq 1 180); do
    if remote "${CONSUMER_HOST}" \
      "docker exec ${CONSUMER_CONTAINER} grep -q 'Training epoch 1/1 completed' ${TRAIN_LOG}" \
      && ! remote "${CONSUMER_HOST}" \
        "docker exec ${CONSUMER_CONTAINER} pgrep -f '[s]cripts/train.py' >/dev/null"; then
      remote "${CONSUMER_HOST}" \
        "docker exec ${CONSUMER_CONTAINER} grep -E 'get_sample|train/loss=|Training epoch 1/1 completed|Traceback|ERROR|Error' ${TRAIN_LOG}"
      return
    fi
    if ! remote "${CONSUMER_HOST}" \
      "docker exec ${CONSUMER_CONTAINER} pgrep -f '[s]cripts/train.py' >/dev/null"; then
      remote "${CONSUMER_HOST}" \
        "docker exec ${CONSUMER_CONTAINER} tail -120 ${TRAIN_LOG}"
      return 1
    fi
    sleep 10
  done
  remote "${CONSUMER_HOST}" \
    "docker exec ${CONSUMER_CONTAINER} tail -120 ${TRAIN_LOG}"
  return 1
}

status() {
  remote "${PRODUCER_HOST}" \
    "echo ===PRODUCER_HOST_MOUNT===; findmnt -T /mnt/share/weight || true; ls -ld ${MODEL_PATH} 2>&1 || true; ls -l ${MODEL_PATH}/config.json 2>&1 || true; docker inspect ${PRODUCER_CONTAINER} --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} propagation={{.Propagation}}{{println}}{{end}}'"
  remote "${PRODUCER_HOST}" \
    "docker exec ${PRODUCER_CONTAINER} sh -c 'echo ===PRODUCER_CONTAINER_MOUNT===; findmnt -T /mnt/share/weight || true; ls -ld ${MODEL_PATH} 2>&1 || true; ls -l ${MODEL_PATH}/config.json 2>&1 || true; echo ===PROCESS===; ps -eo pid,etime,stat,args | grep -E \"launch_vllm|vllm.entrypoints|EngineCore|Worker_TP\" | grep -v grep || true; echo ===NPU===; npu-smi info || true; echo ===PORT===; ss -ltn | grep :${PRODUCER_PORT} || true; echo ===API===; curl -sf --max-time 5 http://127.0.0.1:${PRODUCER_PORT}/v1/models || true; echo ===EVENTS===; grep -E \"POST /v1|put_sample|copy_to_send_buffer|TE send_hs|OutOfMemory|OOM|Traceback|ERROR|Error|RuntimeError|ValueError|HCCL|HCCl|MemoryError|EngineCore failed\" ${PRODUCER_LOG} || true; echo ===TAIL===; tail -240 ${PRODUCER_LOG}'"
  remote "${CONSUMER_HOST}" \
    "docker exec ${CONSUMER_CONTAINER} sh -c 'echo ===PROCESS===; pgrep -af \"torchrun|scripts/train.py\" || true; echo ===EVENTS===; grep -E \"HTTP Request|APIConnectionError|get_sample|batch_transfer|Traceback|ERROR|Error\" ${TRAIN_LOG} 2>/dev/null || true; echo ===TAIL===; tail -80 ${TRAIN_LOG} 2>/dev/null || true; cat /tmp/dflash2_train_save_path 2>/dev/null || true'"
}

status_training() {
  remote "${CONSUMER_HOST}" \
    "docker exec ${CONSUMER_CONTAINER} sh -c 'echo ===PROCESS===; ps -eo pid,ppid,etime,stat,args | grep -E \"torchrun|scripts/train.py\" | grep -v grep || true; echo ===MODEL_PATHS===; for path in /mnt/hcs/weights/Qwen3.8-27B /mnt/hcs/models/Qwen3.8-27B /mnt/hcs/y00917737/weights/Qwen3.8-27B; do test -f \"\$path/config.json\" && printf \"model=%s\\n\" \"\$path\"; done; echo ===DRAFT_CHECKPOINT===; draft=/mnt/hcs/weights/Qwen3.8-27B-DFlash2; test -d \"\$draft\" && find \"\$draft\" -maxdepth 1 -type f -printf \"%f %s bytes\\n\" | sort || true; test -f \"\$draft/config.json\" && python3 -c \"import json; c=json.load(open('\\\$draft/config.json')); print(c.get('architectures')); print(c.get('model_type')); print(c.get('speculators_config', {}).get('verifier', {}))\" || true; echo ===LOG_HEAD===; head -1 ${TRAIN_LOG} 2>/dev/null || true; echo ===RESULTS===; grep -E \"get_sample|train/loss=|Training epoch 1/1 completed|Traceback|ERROR|Error\" ${TRAIN_LOG} 2>/dev/null || true; echo ===CHECKPOINT===; save_path=\$(cat /tmp/dflash2_train_save_path 2>/dev/null || true); test -n \"\$save_path\" && find \"\$save_path\" -maxdepth 2 -type f -printf \"%f %s bytes\\n\" || true'"
}

case ${1:-smoke} in
  prepare-producer)
    prepare_producer
    ;;
  start-producer)
    start_producer
    ;;
  wait-producer)
    wait_for_producer
    ;;
  start-training)
    start_training
    ;;
  sync-verifier)
    sync_verifier
    ;;
  wait-training)
    wait_for_training
    ;;
  status)
    status
    ;;
  status-training)
    status_training
    ;;
  reset-consumer)
    stop_stale_training
    ;;
  smoke)
    prepare_producer
    start_producer
    wait_for_producer
    sync_verifier
    start_training
    wait_for_training
    status
    ;;
  *)
    echo "Usage: $0 {prepare-producer|start-producer|wait-producer|sync-verifier|start-training|wait-training|status|status-training|reset-consumer|smoke}" >&2
    exit 2
    ;;
esac
