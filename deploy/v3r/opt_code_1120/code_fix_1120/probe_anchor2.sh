cd /vllm-workspace/speculators/src/speculators || exit 9
echo "=== 1) 我们的训练命令（实际参数）==="
cat /mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/checkpoints/train_command.txt 2>/dev/null | head -40
echo "=== 2) select_anchors 的调用者（训练循环？）==="
grep -rn "select_anchors\|get_base_indices_for_anchored_blocks" --include="*.py" . 2>/dev/null | grep -v "def select_anchors\|def get_base" | head -10
echo "=== 3) dspark 路径的锚点/块用法 ==="
grep -rn "anchor\|block_size" --include="*.py" models/dspark/ train/trainer*.py train/loop*.py 2>/dev/null | head -20
