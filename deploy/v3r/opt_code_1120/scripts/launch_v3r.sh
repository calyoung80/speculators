#!/bin/bash
# launch_v3r.sh - v3 重训实验: 4600 v2巅峰权重(仅权重,无优化器状态) + train_1117_32k_v3 首训 (2-node DDP)
# 基于 launch_fresh32k.sh 适配, 差异仅:
#   SAVE/LOGDIR -> dspark_fresh32k_v3r/{checkpoints,log} (隔���守卫同步更换)
#   PORT 29520 -> 29521; TAG fresh32k_rN -> v3r_rN
#   --from-pretrained -> _v3r/init_weights (= archive_v2era/ckpt_step4600_v2final 的 config+model)
#   --seed 1: 采样序 = f(seed+epoch) = f(1) = 旧 run v3 段(epoch1,seed0)同序 -> 缓存温启动+可比性
#   新增 MALLOC_* 导出 (glibc 调优: 禁动态阈值棘轮+限arena, 压 worker 驻留高水位 ~510GB)
# 其余(数据/TSL/优化器/损失)与 fresh32k 逐字一致, 保证可比性。
# usage: bash launch_v3r.sh <rank> <nproc_per_node> <nnodes>
set -u
RANK=${1:?}; NPROC=${2:?}; NNODES=${3:?}
NW=${NW:-2}
PF=${PF:-2}
MTW=${MTW:-6}
REQ_TIMEOUT=${REQ_TIMEOUT:-900}
MAXRET=${MAXRET:-5}
MAXFAIL=${MAXFAIL:-20}
GRAD_CKPT=${GRAD_CKPT:-1}
TSL=${TSL:-32768}
CKPT_STEPS=${CKPT_STEPS:-100}
DATA=${DATA:-/mnt/cephfs/hw/zhujieming/spectrain/data/train_1117_32k_v3}
MAX_ANCHORS=${MAX_ANCHORS:-1024}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
SEED=${SEED:-1}

MASTER=10.100.1.70
PORT=29521
SAVE=${SAVE:-/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/checkpoints}
STATE=${STATE:-/mnt/cephfs/glm5_3_hidden}
EP=http://10.100.1.95:8080/v1
LOGDIR=${LOGDIR:-/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/log}
LAUNCHLOG=${LAUNCHLOG:-$LOGDIR/launcher_r${RANK}.log}
mkdir -p "$LOGDIR" 2>/dev/null
exec >>"$LAUNCHLOG" 2>&1

[[ "$SAVE" == *"/dspark_fresh32k_v3r/"* ]] || { echo "[ABORT] save path not isolated"; exit 1; }
[ -d "$DATA" ] || { echo "[ABORT] dataset not found: $DATA"; exit 1; }
[ -d "/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/init_weights" ] || { echo "[ABORT] init weights not found"; exit 1; }
[ "$TSL" -ge 32768 ] || echo "[WARN] TSL=$TSL < 32768: long samples will be truncated"

CKPT_ARGS=""
[ -n "$CKPT_STEPS" ] && [ "$CKPT_STEPS" != "0" ] && CKPT_ARGS="--checkpoint-step-interval $CKPT_STEPS"
GRAD_ARGS=""
[ "$GRAD_CKPT" != "0" ] && GRAD_ARGS="--gradient-checkpointing"

VIS=$(seq -s, 0 $((NPROC-1)))

export ASCEND_A3_ENABLE=1
export HCCL_IF_IP=10.100.1.$([ "$RANK" = "0" ] && echo 70 || echo 71)
export GLOO_SOCKET_IFNAME=bond0
export TP_SOCKET_IFNAME=bond0
export HCCL_SOCKET_IFNAME=bond0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=2048
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export PYTHONHASHSEED=0
export ASCEND_WORK_PATH=/data/hw/zhujieming/ascend_work
export ASCEND_RT_VISIBLE_DEVICES="$VIS"
export MALLOC_ARENA_MAX=2
export NUMPY_MADVISE_HUGEPAGE=0
export MALLOC_CONF=thp:never
export MALLOC_TRIM_THRESHOLD_=134217728
export MALLOC_MMAP_THRESHOLD_=134217728
export LD_PRELOAD=/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/jemalloc/libjemalloc53.so.2

mkdir -p "$SAVE" "$LOGDIR"
cd /vllm-workspace/speculators || exit 1
TAG=v3r_r${RANK}

echo "=== $(date '+%F %T') v3r: DATA=$DATA TSL=$TSL NW=$NW PF=$PF MTW=$MTW ckpt=$CKPT_STEPS grad_ckpt=$GRAD_CKPT seed=$SEED reqto=$REQ_TIMEOUT retries=$MAXRET maxfail=$MAXFAIL nproc=$NPROC anchors=$MAX_ANCHORS malloc=tuned-glibc+jemalloc ==="

nohup torchrun \
  --nnodes=$NNODES --nproc_per_node=$NPROC --node_rank=$RANK \
  --master_addr=$MASTER --master_port=$PORT \
  -m speculators.train \
  --verifier-name-or-path /mnt/cephfs/weight/GLM-5.3 \
  --from-pretrained /mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/init_weights \
  --trust-remote-code \
  --seed $SEED \
  --data-path $DATA \
  --train-data-ratio 0.99 --epochs ${EPOCHS:-8} \
  --lr 1.0e-4 --optimizer muon --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --total-seq-len $TSL --speculator-type dspark --block-size 8 --max-anchors $MAX_ANCHORS \
  --target-layer-ids 1 20 38 56 75 --draft-attn-impl $ATTN_IMPL \
  --markov-rank 256 --markov-head-type vanilla \
  --enable-confidence-head --confidence-head-with-markov --confidence-head-alpha 1.0 \
  --mask-token-id 154856 --scheduler-type cosine --scheduler-warmup-ratio 0.01 \
  --weight-decay 0.0 --dflash-decay-gamma 4.0 \
  --on-missing generate \
  --hidden-states-backend file --hidden-states-path $STATE \
  --vllm-endpoint $EP \
  --request-timeout $REQ_TIMEOUT --max-retries $MAXRET \
  --max-consecutive-generation-failures $MAXFAIL \
  --save-path $SAVE --log-dir $LOGDIR \
  $CKPT_ARGS $GRAD_ARGS \
  --num-workers $NW --prefetch-factor $PF --max-transfer-workers $MTW \
  --log-freq 1 --logger tensorboard \
  > $LOGDIR/train_$TAG.log 2>&1 &
echo "launched v3r $TAG pid=$! (nproc=$NPROC vis=$VIS NW=$NW PF=$PF MTW=$MTW tsl=$TSL ckpt=${CKPT_STEPS:-none} grad_ckpt=$GRAD_CKPT seed=$SEED anchors=$MAX_ANCHORS malloc=tuned-glibc)"