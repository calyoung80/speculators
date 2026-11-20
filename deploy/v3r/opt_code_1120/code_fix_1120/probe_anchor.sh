cd /vllm-workspace/speculators/src/speculators || exit 9
echo "=== 1) select_anchors / anchor 函数定义与调用 ==="
grep -rn "anchor" --include="*.py" train/data.py models/dflash/utils.py 2>/dev/null | head -25
echo "=== 2) hidden_states 的下标/切片用法 ==="
grep -rn "hidden_states\[" --include="*.py" train/ models/ 2>/dev/null | head -15
echo "=== 3) preprocess 在哪定义（数据侧 or 训练侧）==="
grep -rn "preprocess" --include="*.py" train/ | head -15
echo "=== 4) __getitem__ 主链 ==="
sed -n "/class SpeculatorsDataset/,/^class /p" train/data.py | head -80
