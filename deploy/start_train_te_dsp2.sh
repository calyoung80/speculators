#!/bin/bash
set -euxo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

REPO=/mnt/hcs/y00917737/te_dspark_submission/speculators
RUN_ID=$(date +%Y%m%d_%H%M%S)
SAVE_PATH=/mnt/hcs/y00917737/dflash2_phase3_smoke_ckpt/${RUN_ID}

export PYTHONPATH=${REPO}/src:${REPO}/hs_connectors/src
export VLLM_PLUGINS=
export ASCEND_RT_VISIBLE_DEVICES=6,7
export ASCEND_TRANSFER_TIMEOUT=30000
export HCCL_NPU_SOCKET_PORT_RANGE=16000,25000
export TE_META_DIR=/mnt/hcs/y00917737/dflash2_te_meta_128
export TE_PRE_INIT=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

cd "${REPO}"
printf '%s\n' "${SAVE_PATH}" > /tmp/dflash2_smoke_save_path

exec torchrun --nproc_per_node=2 --nnodes=1 scripts/train.py \
  --speculator-type dflash2 \
  --verifier-name-or-path /mnt/hcs/weights/Qwen3.8-27B \
  --data-path /mnt/hcs/y00917737/dflash2_data_27b/training_data_v2 \
  --hidden-states-backend mooncake-te \
  --mooncake-te-zmq-port 9999 \
  --mooncake-te-producer-ip 127.0.0.1 \
  --vllm-endpoint http://127.0.0.1:8000/v1 \
  --on-missing generate \
  --on-generate delete \
  --target-layer-ids 5 19 33 47 61 \
  --total-seq-len 4096 \
  --block-size 8 \
  --max-anchors 64 \
  --loss-implementation eager \
  --loss-fn ce \
  --optimizer muon \
  --lr 2e-4 \
  --muon-lr 2e-3 \
  --scheduler-type none \
  --draft-attn-impl sdpa \
  --draft-arch qwen3 \
  --max-steps 1 \
  --epochs 1 \
  --log-freq 1 \
  --save-path "${SAVE_PATH}" \
  --no-resume-from-checkpoint \
  --num-workers 0 \
  --prefetch-factor 2 \
  --request-timeout 600 \
  > /tmp/train_dflash2_smoke.log 2>&1
