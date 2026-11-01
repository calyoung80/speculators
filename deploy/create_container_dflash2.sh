#!/bin/bash
# ============================================================
# DFlash2 训练容器创建脚本
# 镜像: 默认使用 quay.io/ascend/vllm-ascend@sha256:866eda94f03689eebe48247d3683515fe4f0b6a81628fc8741788b519f501222。
# 跨机任务不能仅凭浮动 tag 判断版本，必须核对相同 immutable Image ID。
# 关键: --ipc=host (修复 Mooncake TE ADXL rtsIpcMemGetExportKey 失败)
#
# 用法:
#   bash deploy/create_container_dflash2.sh [CONTAINER_NAME] [NPU_IDS]
#   bash deploy/create_container_dflash2.sh dflash2_roce_producer_118 1
# ============================================================
set -euxo pipefail

CONTAINER_NAME=${1:-dflash2_train}
NPU_IDS=${2:-0,1,2,3,4,5,6,7}
IMAGE=${IMAGE:-quay.io/ascend/vllm-ascend@sha256:866eda94f03689eebe48247d3683515fe4f0b6a81628fc8741788b519f501222}
EXPECTED_IMAGE_ID=${EXPECTED_IMAGE_ID:-}

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
  -v /mnt:/mnt \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu/:/usr/slog \
  -e VLLM_PLUGINS=ascend \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/mnt/hcs/cache/huggingface \
  ${IMAGE} \
  sleep infinity

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
ACTUAL_IMAGE_ID=$(docker inspect ${CONTAINER_NAME} --format '{{.Image}}')
IMAGE_CREATED=$(docker image inspect "${IMAGE}" --format '{{.Created}}' 2>/dev/null || echo unknown)
IMAGE_OS=$(docker image inspect "${IMAGE}" --format '{{.Os}}' 2>/dev/null || echo unknown)
echo "  image: ${IMAGE}"
echo "  image_id: ${ACTUAL_IMAGE_ID}"
echo "  image_created: ${IMAGE_CREATED}"
echo "  image_os: ${IMAGE_OS}"
if [ -n "${EXPECTED_IMAGE_ID}" ] && [ "${ACTUAL_IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]; then
  echo "ERROR: image ID mismatch: expected ${EXPECTED_IMAGE_ID}, got ${ACTUAL_IMAGE_ID}" >&2
  exit 1
fi
echo "  IPC: $(docker inspect ${CONTAINER_NAME} --format '{{.HostConfig.IpcMode}}')"
echo "  shm: $(docker exec ${CONTAINER_NAME} df -h /dev/shm | tail -1 | awk '{print $2}')"
echo "  npu: $(docker exec ${CONTAINER_NAME} npu-smi info 2>/dev/null | grep '910B3' | wc -l) x 910B3"
