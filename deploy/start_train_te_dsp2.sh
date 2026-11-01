#!/bin/bash
set -x
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

export PYTHONPATH=/mnt/hcs/y00917737:/mnt/hcs/y00917737/speculators/src:/mnt/hcs/y00917737/speculators/hs_connectors/src
export VLLM_PLUGINS=
export ASCEND_RT_VISIBLE_DEVICES=6,7
export ASCEND_TRANSFER_TIMEOUT=30000
export HCCL_NPU_SOCKET_PORT_RANGE=16000,25000
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

cd /mnt/hcs/y00917737/speculators

torchrun --nproc_per_node=2 --nnodes=1 scripts/train.py \
  --speculator-type dspark \
  --verifier-name-or-path /mnt/hcs/models/Qwen3.6-35B-A3B \
  --data-path /mnt/hcs/y00917737/training_data_32k \
  --hidden-states-backend mooncake-te \
  --mooncake-te-zmq-port 9999 \
  --mooncake-te-producer-ip 127.0.0.1 \
  --vllm-endpoint http://127.0.0.1:8000/v1 \
  --on-missing generate \
  --on-generate delete \
  --target-layer-ids 3 19 35 \
  --total-seq-len 8192 \
  --block-size 7 \
  --max-anchors 512 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --confidence-head-alpha 1.0 \
  --loss-fn '{ "ce":0.1,"tv":0.9}' \
  --dflash-decay-gamma 4.0 \
  --draft-attn-impl sdpa \
  --draft-arch qwen3 \
  --max-steps 1 \
  --epochs 1 \
  --log-freq 1 \
  --save-path /tmp/ckpt_$(date +%s) \
  --lr 1e-4 \
  --num-workers 0 \
  --prefetch-factor 2 \
  --request-timeout 600
