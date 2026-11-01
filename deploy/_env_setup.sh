#!/bin/bash
set -e
mount -t tmpfs -o size=8G none /dev/shm 2>/dev/null || true
echo "127.0.0.1 localhost" >> /etc/hosts
echo "::1 localhost" >> /etc/hosts
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip config set global.trusted-host repo.huaweicloud.com
pip install "datasets<5.0" -q
rm -rf /mnt/hcs/y00917737/training_data_32k/checkpoints_te_v* 2>/dev/null
python3 -c "import datasets; print('datasets', datasets.__version__)"
echo "ENV_READY"
