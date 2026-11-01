#!/bin/bash
# ============================================================
# DFlash2 训练容器创建脚本
# 镜像: quay.nju.edu.cn/ascend/vllm-ascend:nightly-main
# 关键: --ipc=host (修复 Mooncake TE ADXL rtsIpcMemGetExportKey 失败)
#
# 用法:
#   bash deploy/create_container_dflash2.sh [CONTAINER_NAME] [NPU_IDS]
#   bash deploy/create_container_dflash2.sh dflash2_roce_producer_118 1
# ============================================================
set -euxo pipefail

CONTAINER_NAME=${1:-dflash2_train}
NPU_IDS=${2:-0,1,2,3,4,5,6,7}
IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:nightly-main

DEVICES=()
for NPU_ID in ${NPU_IDS//,/ }; do
  DEVICES+=(--device "/dev/davinci${NPU_ID}")
done
DEVICES+=(
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)

docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

docker run -itd \
  --name ${CONTAINER_NAME} \
  --ipc=host \
  --privileged \
  --network=host \
  --shm-size=8g \
  "${DEVICES[@]}" \
  -v /mnt/hcs:/mnt/hcs \
  -v /mnt/share/weight:/mnt/share/weight:ro \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu/:/usr/slog \
  -e VLLM_PLUGINS=ascend \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/mnt/hcs/cache/huggingface \
  --entrypoint /bin/bash \
  ${IMAGE} \
  -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 && exec sleep infinity"

sleep 3

# pip 华为云镜像 + 安装训练依赖
docker exec ${CONTAINER_NAME} bash -c '
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip config set global.trusted-host repo.huaweicloud.com
pip install datasets multiprocess pyarrow pyarrow-hotfix 2>&1 | tail -3
'

echo ""
echo "Container ${CONTAINER_NAME} ready."
echo "  NPU: ${NPU_IDS}"
echo "  IPC: $(docker inspect ${CONTAINER_NAME} --format '{{.HostConfig.IpcMode}}')"
echo "  shm: $(docker exec ${CONTAINER_NAME} df -h /dev/shm | tail -1 | awk '{print $2}')"
echo "  npu: $(docker exec ${CONTAINER_NAME} npu-smi info 2>/dev/null | grep '910B3' | wc -l) x 910B3"
