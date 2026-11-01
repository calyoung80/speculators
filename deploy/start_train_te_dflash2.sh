#!/bin/bash
# ============================================================
# DFlash2 cross-node smoke training (Mooncake TE + RoCE, DDP=2).
#
# 验证过的参数:
#   optimizer=muon, loss=kl_div, lr=2e-4, max_anchors=256
#   Phase 2 (1000步) 全程稳定, loss 6.55->0.15
#   Chunked loss 补丁已应用, 支持 max_anchors=512 (但单卡内存上限 256)
#
# Cross-node topology:
#   - Producer: 71.10.29.118, NPU 1,2, TP=2
#   - Consumer: 71.10.29.119, NPU 2,3, DDP=2
# Inter-node TransferEngine selects RoCE automatically, while local DDP keeps
# its default HCCS path. Do not set HCCL_INTRA_ROCE_ENABLE here.
#
# 用法:
#   docker exec -d dflash2_train bash deploy/start_train_te_dflash2.sh \
#     71.10.29.118 18000 2,3 71.10.29.119
# ============================================================
set -euxo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

REPO=/mnt/hcs/y00917737/te_dspark_submission/speculators
VERIFIER_PATH=${VERIFIER_PATH:-/mnt/hcs/y00917737/dflash2_verifier_minimal}
PRODUCER_HOST=${1:-71.10.29.118}
PRODUCER_PORT=${2:-18000}
CONSUMER_NPUS=${3:-2,3}
LOCAL_HOST_IP=${4:-71.10.29.119}
RUN_ID=$(date +%Y%m%d_%H%M%S)
SAVE_PATH=${SAVE_PATH:-/mnt/hcs/y00917737/dflash2_cross_node_smoke_ckpt/${RUN_ID}}
RUN_NAME=${RUN_NAME:-dflash2_cross_node_${RUN_ID}}
DATA_PATH=${DATA_PATH:-/mnt/hcs/y00917737/dflash2_data_27b/training_data_v2}
TE_META_DIR=${TE_META_DIR:-/mnt/hcs/y00917737/dflash2_te_meta_118_119}
TOTAL_SEQ_LEN=${TOTAL_SEQ_LEN:-4096}
MAX_ANCHORS=${MAX_ANCHORS:-64}
MAX_STEPS=${MAX_STEPS:-1}
EPOCHS=${EPOCHS:-1}
TRAIN_DATA_RATIO=${TRAIN_DATA_RATIO:-0.9}
FSDP_SHARD=${FSDP_SHARD:-0}
CHECKPOINT_STEP_INTERVAL=${CHECKPOINT_STEP_INTERVAL:-0}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-0}
export TE_META_DIR

FSDP_ARGS=()
if [ "${FSDP_SHARD}" = "1" ]; then
  FSDP_ARGS+=(--fsdp-shard)
fi

CHECKPOINT_ARGS=()
if [ "${CHECKPOINT_STEP_INTERVAL}" -gt 0 ]; then
  CHECKPOINT_ARGS+=(--checkpoint-step-interval "${CHECKPOINT_STEP_INTERVAL}")
fi

RESUME_ARGS=(--no-resume-from-checkpoint)
if [ "${RESUME_FROM_CHECKPOINT}" = "1" ]; then
  RESUME_ARGS=()
fi

export PYTHONPATH=${REPO}/src:${REPO}/hs_connectors/src
export VLLM_PLUGINS=
export ASCEND_RT_VISIBLE_DEVICES=${CONSUMER_NPUS}
export ASCEND_TRANSFER_TIMEOUT=30000
export HCCL_NPU_SOCKET_PORT_RANGE=16000,25000
export MOONCAKE_LOCAL_HOSTNAME=${LOCAL_HOST_IP}
export TE_PRE_INIT=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29619
# Cross-node TE selects RoCE on its own. Keep local DDP on HCCS.
unset HCCL_INTRA_ROCE_ENABLE
unset HCCL_INTRA_PCIE_ENABLE
unset HCCL_IF_IP

mkdir -p "${SAVE_PATH}"
cd "${REPO}"
printf '%s\n' "${SAVE_PATH}" > /tmp/dflash2_train_save_path
printf 'producer=%s:%s consumer_npus=%s master_port=%s hccl_intra_roce=unset\n' \
  "${PRODUCER_HOST}" "${PRODUCER_PORT}" "${CONSUMER_NPUS}" "${MASTER_PORT}" \
  > /tmp/dflash2_cross_node_train.log
python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('${VERIFIER_PATH}')" \
  >> /tmp/dflash2_cross_node_train.log 2>&1

exec torchrun --nproc_per_node=2 --nnodes=1 \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}" \
  scripts/train.py \
  --speculator-type dflash2 \
  --verifier-name-or-path "${VERIFIER_PATH}" \
  --data-path "${DATA_PATH}" \
  --hidden-states-backend mooncake-te \
  --mooncake-te-zmq-port 9999 \
  --mooncake-te-producer-ip "${PRODUCER_HOST}" \
  --vllm-endpoint "http://${PRODUCER_HOST}:${PRODUCER_PORT}/v1" \
  --on-missing generate \
  --on-generate delete \
  --target-layer-ids 5 19 33 47 61 \
  --total-seq-len "${TOTAL_SEQ_LEN}" \
  --block-size 8 \
  --max-anchors "${MAX_ANCHORS}" \
  --loss-implementation eager \
  --loss-fn kl_div \
  --optimizer muon \
  --lr 2e-4 \
  --muon-lr 2e-3 \
  --scheduler-type none \
  --max-steps "${MAX_STEPS}" \
  --epochs "${EPOCHS}" \
  --train-data-ratio "${TRAIN_DATA_RATIO}" \
  "${FSDP_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  --log-freq 1 \
  --save-path "${SAVE_PATH}" \
  "${RESUME_ARGS[@]}" \
  --draft-attn-impl sdpa \
  --draft-arch qwen3 \
  --dflash-decay-gamma 4.0 \
  --conv-kernel-size 2 \
  --selector-rank 256 \
  --selector-top-k 16 \
  --num-workers 0 \
  --prefetch-factor 2 \
  --request-timeout 600 \
  --run-name "${RUN_NAME}" \
  >> /tmp/dflash2_cross_node_train.log 2>&1
