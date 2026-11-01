#!/bin/bash
# ============================================================
# vLLM producer startup (NPU 4,5, TP=2)
# Mooncake TransferEngine backend, ZMQ control plane on port 9999
# Env vars aligned with vllm-ascend PD deployment guide
# ============================================================
set -x

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

export PYTHONPATH=/mnt/hcs/y00917737:/mnt/hcs/y00917737/speculators/src:/mnt/hcs/y00917737/speculators/hs_connectors/src:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/python/site-packages
export VLLM_PLUGINS=ascend
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MALLOC_CHECK_=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/hcs/cache/huggingface
export ASCEND_RT_VISIBLE_DEVICES=4,5
export ASCEND_TRANSFER_TIMEOUT=30000
export HCCL_NPU_SOCKET_PORT_RANGE=16000,25000
export VLLM_NO_USAGE_STATS=1

cd /mnt/hcs/y00917737/speculators

python3 scripts/launch_vllm.py \
  /mnt/hcs/models/Qwen3.6-35B-A3B \
  --hidden-states-backend mooncake-te \
  --mooncake-te-zmq-port 9999 \
  --target-layer-ids 3 19 35 \
  -- \
  --tensor-parallel-size 2 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --max-num-seqs 4 \
  --max-num-batched-tokens 32768 \
  --enforce-eager \
  --no-enable-chunked-prefill
