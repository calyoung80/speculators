# DSpark + Mooncake TE 部署操作手册

## 概述

本手册用于在 Ascend NPU 上部署 DSpark 训练 + vLLM 推理，使用 Mooncake TransferEngine (TE) 直接传输 hidden states。

### 环境要求

- **硬件**: 8×Ascend 910B3 (60GB/卡)
- **镜像**: `quay.io/ascend/vllm-ascend:v0.23.0`
- **CANN**: 9.1.0
- **vLLM**: 0.23.0
- **vllm-ascend**: 0.23.0
- **Mooncake**: mooncake-transfer-engine-npu 0.3.11.post1 (镜像预装)
- **模型**: Qwen3.6-35B-A3B (MoE, 35B total / 3B active, hidden_size=2048)
- **NPU 分配**: NPU 4,5 = vLLM (TP=2), NPU 6,7 = 训练 (DDP=2)

### 涉及的三个仓库

| 仓库 | 分支 | 基于 | 说明 |
|------|------|------|------|
| `calyoung80/speculators` | `dspark-ascend-patches` | `main` | 训练框架 + TE connector + 部署脚本 |
| `calyoung80/vllm-ascend` | `dspark-ascend-patches` | `v0.23.0` | Ascend 适配补丁 |
| `calyoung80/vllm` | `dspark-ascend-patches` | `v0.23.0` | PR #51328 per-group slot_mapping 修复 |

---

## 步骤 1: 创建容器

```bash
docker run -itd \
  --name dspark_te \
  --net=host \
  --privileged \
  --runtime ascend \
  --shm-size=8g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /mnt/hcs:/mnt/hcs \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/mnt/hcs/cache/huggingface \
  quay.io/ascend/vllm-ascend:v0.23.0 \
  sleep infinity
```

> 如果只需要 4 卡 (NPU 4-7), 可去掉 davinci0-3。

---

## 步骤 2: 安装代码

### 2.1 克隆三个仓库

```bash
# speculators (训练框架 + connector + 部署脚本)
cd /mnt/hcs/y00917737
git clone -b dspark-ascend-patches git@github.com:calyoung80/speculators.git speculators

# vllm-ascend 补丁
cd /vllm-workspace/vllm-ascend
git fetch origin
git checkout dspark-ascend-patches
# 或直接用已有仓库 checkout 分支

# vllm 补丁 (PR #51328)
cd /vllm-workspace/vllm
git fetch origin
git checkout dspark-ascend-patches
```

> 容器内 `/vllm-workspace/vllm` 和 `/vllm-workspace/vllm-ascend` 已有 git 仓库，
> 只需 `git remote add fork git@github.com:calyoung80/xxx.git` 然后 `git fetch fork && git checkout fork/dspark-ascend-patches`。

### 2.2 安装 sitecustomize.py

```bash
docker cp /mnt/hcs/y00917737/speculators/deploy/sitecustomize.py \
  dspark_te:/usr/local/python3.12.13/lib/python3.12/site-packages/sitecustomize.py
```

sitecustomize.py 修复 3 个 vLLM-Ascend 兼容性问题:
1. `is_monolithic` AttributeError (FusedMoE)
2. `topk_softmax` 缺失 `_moe_C` C++ 扩展 → 纯 PyTorch 实现
3. global patch 未加载 → 强制 `import vllm_ascend.patch.platform`

### 2.3 验证代码安装

```bash
docker exec dspark_te python3 -c "
import vllm; print('vllm', vllm.__version__)
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
assert hasattr(ExtractHiddenStatesProposer, '_resolve_slot_mapping'), 'PR #51328 not applied'
print('PR #51328 OK')
from vllm_ascend.spec_decode.extract_hidden_states_proposer import AscendExtractHiddenStatesProposer
print('AscendExtractHiddenStatesProposer OK')
"
```

---

## 步骤 3: 环境准备

```bash
docker exec dspark_te bash -c 'bash /mnt/hcs/y00917737/speculators/deploy/_env_setup.sh'
```

期望输出:
```
datasets 4.x.x
ENV_READY
```

_env_setup.sh 做了:
- 挂载 8G tmpfs 到 /dev/shm
- 添加 localhost 到 /etc/hosts
- 配置 pip 华为云镜像
- 安装 datasets<5.0
- 清理旧 checkpoints

---

## 步骤 4: 启动 vLLM (Producer)

```bash
docker exec -d dspark_te bash -c \
  'bash /mnt/hcs/y00917737/speculators/deploy/start_vllm_te.sh > /tmp/vllm.log 2>&1'
```

等待 ~4 分钟，检查健康:

```bash
curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health
# 期望: 200
```

如果不是 200，检查日志:
```bash
docker exec dspark_te tail -50 /tmp/vllm.log
```

### vLLM 配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| ASCEND_RT_VISIBLE_DEVICES | 4,5 | NPU 4,5 |
| --tensor-parallel-size | 2 | TP=2 |
| --hidden-states-backend | mooncake-te | TE 后端 |
| --mooncake-te-zmq-port | 9999 | ZMQ 控制面端口 |
| --target-layer-ids | 3 19 35 | 提取 hidden states 的层 |
| --enforce-eager | (flag) | 禁用图模式 (调试用) |
| --no-enable-chunked-prefill | (flag) | 禁用 chunked prefill |
| --max-model-len | 32768 | 最大序列长度 |
| --gpu-memory-utilization | 0.8 | NPU 显存占用 |

---

## 步骤 5: 启动训练 (Consumer)

```bash
docker exec -d dspark_te bash -c \
  'bash /mnt/hcs/y00917737/speculators/deploy/start_train_te_dsp2.sh > /tmp/train.log 2>&1'
```

等待 ~3 分钟，检查结果:

```bash
docker exec dspark_te bash -c 'grep "train/loss\|step_ms\|NaN\|completed" /tmp/train.log'
```

期望输出:
```
train/loss=3.015, ...    (step 0, 无 NaN)
train/loss=2.240, ...    (step 1, 无 NaN)
Training epoch 1/1 completed
```

### 训练配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| ASCEND_RT_VISIBLE_DEVICES | 6,7 | NPU 6,7 |
| --speculator-type | dspark | DFlash base + Markov head |
| --hidden-states-backend | mooncake-te | TE 后端 |
| --mooncake-te-producer-ip | 127.0.0.1 | 本机 vLLM |
| --vllm-endpoint | http://127.0.0.1:8000/v1 | vLLM API |
| --target-layer-ids | 3 19 35 | 与 vLLM 一致 |
| --total-seq-len | 8192 | 序列长度 |
| --block-size | 7 | DFlash block 大小 |
| --markov-rank | 256 | Markov 矩阵秩 |
| --enable-confidence-head | (flag) | 启用 confidence head |
| --loss-fn | {"ce":0.1,"tv":0.9} | CE + TV loss |
| --max-steps | 1 | 最大步数 (改为 10 做更长训练) |
| --epochs | 1 | 训练轮数 |
| --lr | 1e-4 | 学习率 |

---

## 步骤 6: 验证性能

### 正常指标

| 指标 | 冷启动 (step 0) | 热运行 (step 1) |
|------|-----------------|----------------|
| step_ms | ~5000ms | ~1200ms |
| fetch_ms | ~2250ms (45%) | ~390ms (32%) |
| fwd_ms | ~2080ms | ~610ms |
| bwd_ms | ~580ms | ~155ms |
| tokens_per_s | ~1320 | ~1370 |
| loss | ~3.015 | ~2.240 |
| NaN | 无 | 无 |

### TE 传输指标

```
get_sample: ret=0, 26.00ms, 3984 MB/s  (冷)
get_sample: ret=0, 2.58ms, 10245 MB/s  (热)
```

### 异常排查

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| health=000 | vLLM 未启动完成 | 等待 4 分钟 |
| NaN in hidden states | PR #51328 未应用 | 检查 extract_hidden_states.py |
| NaN without PR warning | ACK sync 未生效 | 检查 connector 的 _prev_te_key |
| Timeout | handle 格式不匹配 | 检查 transfer.py _normalize_handle |
| HCCL conflict | ADXL/HCCL 初始化顺序 | 检查 distributed.py TE_PRE_INIT |
| aivec error | torch.tensor on NPU | 检查 CpuGpuBuffer patch |

---

## 附录: 容器重建后快速恢复

如果容器被删除，按以下顺序恢复:

```bash
# 1. 创建容器 (步骤 1)

# 2. 恢复代码
cd /vllm-workspace/vllm-ascend && git checkout dspark-ascend-patches
cd /vllm-workspace/vllm && git checkout dspark-ascend-patches
docker cp /mnt/hcs/y00917737/speculators/deploy/sitecustomize.py \
  dspark_te:/usr/local/python3.12.13/lib/python3.12/site-packages/sitecustomize.py

# 3. 环境准备
docker exec dspark_te bash -c 'bash /mnt/hcs/y00917737/speculators/deploy/_env_setup.sh'

# 4. 启动 vLLM
docker exec -d dspark_te bash -c \
  'bash /mnt/hcs/y00917737/speculators/deploy/start_vllm_te.sh > /tmp/vllm.log 2>&1'

# 5. 等待 health=200

# 6. 启动训练
docker exec -d dspark_te bash -c \
  'bash /mnt/hcs/y00917737/speculators/deploy/start_train_te_dsp2.sh > /tmp/train.log 2>&1'
```

---

## 附录: 代码改动清单

### speculators 仓库 (7 commits)

| Commit | 说明 |
|--------|------|
| `e8d4049` | feat: Mooncake TE NPU direct hidden-states transfer |
| `8f796db` | fix: ACK sync to prevent NaN race condition |
| `b42cd15` | fix: correct NaN root cause analysis |
| `4c3047a` | docs: NaN root cause documentation |
| `6a5b48d` | cleanup: remove debug traces, keep ACK sync |
| `c4ab638` | feat: DSpark training + NPU compatibility fixes |
| `b23a5a2` | deploy: add sitecustomize.py + startup scripts |

### vllm-ascend 仓库 (1 commit)

| 文件 | 改动 |
|------|------|
| `worker/model_runner_v1.py` | + AscendExtractHiddenStatesProposer isinstance check; + validate_same_kv_cache_group() 调用 |
| `spec_decode/extract_hidden_states_proposer.py` | CpuGpuBuffer 替换 torch.tensor (修复 aivec error) |

### vllm 仓库 (1 commit)

| 文件 | 改动 |
|------|------|
| `v1/spec_decode/extract_hidden_states.py` | + _resolve_slot_mapping() 方法; propose() 使用 per-group slot_mapping (PR #51328) |

### sitecustomize.py (独立文件)

| 修复 | 说明 |
|------|------|
| is_monolithic | `getattr(self.experts_cls, "is_monolithic", lambda: False)()` |
| topk_softmax | 纯 PyTorch softmax + topk 替代 `_moe_C` C++ 扩展 |
| global patch | 强制 `import vllm_ascend.patch.platform` + `_ensure_global_patch()` |
