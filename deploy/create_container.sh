#!/bin/bash
# ============================================================
# dspark_train_smoke 容器创建脚本
# 镜像: quay.nju.edu.cn/ascend/vllm-ascend:nightly-main
# 用途: DSpark 训练 + vLLM 推理 (Mooncake TransferEngine NPU 直传)
# 参考: wxh_new 容器配置 (显式 --device /dev/davinci0-7 绕过 vNPU 限制)
# ============================================================

docker stop dspark_train_smoke 2>/dev/null
docker rm dspark_train_smoke 2>/dev/null

docker run -itd \
  --name dspark_train_smoke \
  --net=host \
  --privileged \
  --runtime ascend \
  --shm-size=8g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu/:/usr/slog \
  -v /mnt/hcs:/mnt/hcs \
  -e VLLM_PLUGINS=ascend \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/mnt/hcs/cache/huggingface \
  quay.nju.edu.cn/ascend/vllm-ascend:nightly-main \
  sleep infinity

sleep 3

# pip 华为云镜像
docker exec dspark_train_smoke bash -c '
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip config set global.trusted-host repo.huaweicloud.com
git config --global http.sslVerify false
'

# 安装训练依赖
docker exec dspark_train_smoke pip install datasets multiprocess pyarrow pyarrow-hotfix 2>&1 | tail -3

# 复制 sitecustomize.py (NPU patch: topk_softmax, is_monolithic, global patch)
docker cp /mnt/hcs/y00917737/sitecustomize.py \
  dspark_train_smoke:/usr/local/python3.12.13/lib/python3.12/site-packages/sitecustomize.py

echo "Container dspark_train_smoke ready."
echo "  shm: $(docker exec dspark_train_smoke df -h /dev/shm | tail -1 | awk '{print $2}')"
echo "  npu-smi: $(docker exec dspark_train_smoke npu-smi info -t board -i 0 2>&1 | grep 'NPU ID' || echo 'not available')"
echo "  driver: $(docker exec dspark_train_smoke ls /usr/local/Ascend/driver/lib64/driver/libascend_hal.so 2>&1)"
