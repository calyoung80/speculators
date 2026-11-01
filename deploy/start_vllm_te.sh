#!/bin/bash
set -euxo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

REPO=/mnt/hcs/y00917737/te_dspark_submission/speculators
NPU_IDS=${1:-1,2}
TP_SIZE=${2:-2}
PORT=${3:-18000}
LOCAL_HOST_IP=${4:-71.10.29.118}
MODEL_PATH=${MODEL_PATH:-/mnt/share/weight/Qwen/Qwen3.8-27B}
TE_META_DIR=${TE_META_DIR:-/mnt/hcs/y00917737/dflash2_te_meta_118_119}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
export TE_META_DIR

export PYTHONPATH=${REPO}/src:${REPO}/hs_connectors/src:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/python/site-packages
export VLLM_PLUGINS=ascend
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}
export MALLOC_CHECK_=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/hcs/cache/huggingface
export ASCEND_RT_VISIBLE_DEVICES=${NPU_IDS}
export ASCEND_TRANSFER_TIMEOUT=30000
export HCCL_NPU_SOCKET_PORT_RANGE=16000,25000
export MOONCAKE_LOCAL_HOSTNAME=${LOCAL_HOST_IP}
export VLLM_NO_USAGE_STATS=1
export TE_TIMING_LOG=${TE_TIMING_LOG:-1}
mkdir -p "${TE_META_DIR}"

cd "${REPO}"
test -f "${MODEL_PATH}/config.json"

printf 'producer_npus=%s tp_size=%s port=%s host=%s model=%s max_model_len=%s max_num_batched_tokens=%s max_num_seqs=%s prefix_caching=disabled chunked_prefill=disabled execute_model_timeout=%s\n' \
  "${NPU_IDS}" "${TP_SIZE}" "${PORT}" "${LOCAL_HOST_IP}" "${MODEL_PATH}" \
  "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" \
  "${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}" \
  > /tmp/dflash2_roce_producer.log

exec python3 scripts/launch_vllm.py \
  "${MODEL_PATH}" \
  --hidden-states-backend mooncake-te \
  --mooncake-te-zmq-port 9999 \
  --target-layer-ids 5 19 33 47 61 \
  -- \
  --api-server-count 1 \
  --renderer-num-workers 1 \
  --tensor-parallel-size "${TP_SIZE}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --enforce-eager \
  >> /tmp/dflash2_roce_producer.log 2>&1
