#!/bin/bash
# ==============================================================================
# apply_patches.sh - 在容器内应用 DFlash2 训练所需的补丁
#
# 检查并应用以下补丁:
#   1. PR #51328: vLLM extract_hidden_states.py (per-group slot_mapping)
#      修复 Qwen3.8-27B hybrid 模型 (5 个 KV cache group) 的 NaN hidden states
#      GitHub: https://github.com/vllm-project/vllm/pull/51328
#
#   2. sitecustomize.py: NPU 兼容性补丁 (topk_softmax, global patch)
#      已在容器创建时安装，此处仅验证
#
# 用法:
#   bash apply_patches.sh                          # 检查并应用
#   bash apply_patches.sh --check                  # 仅检查，不应用
#
# 退出码:
#   0 = 所有补丁已就绪
#   1 = 有补丁未应用且无法修复
# ==============================================================================
set -euo pipefail

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

VLLM_FILE="/vllm-workspace/vllm/vllm/v1/spec_decode/extract_hidden_states.py"
PATCH_SRC="/mnt/hcs/y00917737/open_code_dflash2/scripts/06_patches/extract_hidden_states_proposer.py.patched"

echo "=========================================="
echo "  补丁检查 $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ── 1. PR #51328: per-group slot_mapping ─────────────────────────────────────
echo ""
echo "--- PR #51328: extract_hidden_states.py ---"

if [ ! -f "$VLLM_FILE" ]; then
    echo "  ERROR: $VLLM_FILE 不存在"
    exit 1
fi

# 检查是否已打补丁 (查找 kv_cache_gid 或 slot_mappings dict 的使用)
if grep -q "kv_cache_gid\|resolved_slot_mapping" "$VLLM_FILE" 2>/dev/null; then
    echo "  状态: ✅ 已应用"
else
    echo "  状态: ❌ 未应用"

    if [ "$CHECK_ONLY" = true ]; then
        echo "  (--check 模式，跳过应用)"
        exit 1
    fi

    if [ ! -f "$PATCH_SRC" ]; then
        echo "  ERROR: 补丁源文件不存在: $PATCH_SRC"
        exit 1
    fi

    # 备份原文件
    if [ ! -f "${VLLM_FILE}.orig" ]; then
        cp "$VLLM_FILE" "${VLLM_FILE}.orig"
        echo "  备份: ${VLLM_FILE}.orig"
    fi

    # 应用补丁
    cp "$PATCH_SRC" "$VLLM_FILE"

    # 验证
    if grep -q "kv_cache_gid\|resolved_slot_mapping" "$VLLM_FILE" 2>/dev/null; then
        echo "  应用: ✅ 成功"
    else
        echo "  应用: ❌ 失败，补丁内容不匹配"
        exit 1
    fi
fi

# ── 2. sitecustomize.py: NPU 兼容性 ──────────────────────────────────────────
echo ""
echo "--- sitecustomize.py: NPU patches ---"

SITECUSTOMIZE="/usr/local/python3.12.13/lib/python3.12/site-packages/sitecustomize.py"
if [ ! -f "$SITECUSTOMIZE" ]; then
    echo "  状态: ❌ 不存在"
    if [ "$CHECK_ONLY" = true ]; then
        exit 1
    fi
    DEPLOY_SC="/mnt/hcs/y00917737/te_dspark_submission/speculators/deploy/sitecustomize.py"
    if [ -f "$DEPLOY_SC" ]; then
        cp "$DEPLOY_SC" "$SITECUSTOMIZE"
        echo "  应用: ✅ 从 deploy/ 复制"
    else
        echo "  ERROR: 源文件不存在: $DEPLOY_SC"
        exit 1
    fi
else
    if grep -q "topk_softmax_pt\|_ensure_global_patch" "$SITECUSTOMIZE" 2>/dev/null; then
        echo "  状态: ✅ 已安装"
    else
        echo "  状态: ⚠️  存在但缺少 NPU 补丁"
        if [ "$CHECK_ONLY" = false ]; then
            DEPLOY_SC="/mnt/hcs/y00917737/te_dspark_submission/speculators/deploy/sitecustomize.py"
            cp "$DEPLOY_SC" "$SITECUSTOMIZE"
            echo "  应用: ✅ 从 deploy/ 覆盖"
        fi
    fi
fi

# ── 3. speculators + hs_connectors 安装验证 ────────────────────────────────────
echo ""
echo "--- speculators (dflash2-ascend) ---"
if python3 -c "from speculators.models.dflash2 import DFlash2DraftModel" 2>/dev/null; then
    echo "  状态: ✅ DFlash2 可用"
else
    echo "  状态: ❌ DFlash2 不可用"
    if [ "$CHECK_ONLY" = false ]; then
        cd /mnt/hcs/y00917737/te_dspark_submission/speculators && pip install -e . -q 2>/dev/null
        echo "  重新安装: $(python3 -c 'from speculators.models.dflash2 import DFlash2DraftModel; print("OK")' 2>/dev/null || echo 'FAILED')"
    fi
fi

echo ""
echo "--- hs_connectors (TE ADXL P2P) ---"
if python3 -c "from hs_connectors import HiddenStatesBackend" 2>/dev/null; then
    echo "  状态: ✅ hs_connectors 可用"
else
    echo "  状态: ❌ hs_connectors 不可用"
    if [ "$CHECK_ONLY" = false ]; then
        cd /mnt/hcs/y00917737/te_dspark_submission/speculators/hs_connectors && pip install -e . -q 2>/dev/null
        echo "  重新安装: $(python3 -c 'from hs_connectors import HiddenStatesBackend; print("OK")' 2>/dev/null || echo 'FAILED')"
    fi
fi

echo ""
echo "=========================================="
echo "  补丁检查完成"
echo "=========================================="
