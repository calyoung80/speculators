echo "=== 1) 池内容（顶层前 10 项 + 计数）==="
ls /mnt/cephfs/glm5_3_hidden | head -10
ls /mnt/cephfs/glm5_3_hidden | wc -l
echo "=== 2) 找任意 safetensors（含子目录，浅层）==="
find /mnt/cephfs/glm5_3_hidden -maxdepth 2 -name "*.safetensors" -size +64M 2>/dev/null | head -3
echo "=== 3) transfer 导入位置与 views_enabled 状态 ==="
docker exec train_ddp python3 -c "
import sys
sys.path.insert(0, '/vllm-workspace/speculators/hs_connectors/src')
import hs_connectors.transfer as tr
import hs_connectors
print('transfer from:', tr.__file__)
print('hs_connectors from:', hs_connectors.__file__)
print('views_enabled (env clean):', tr.views_enabled())
import os
os.environ['HS_READONLY_VIEWS']='1'
print('views_enabled (env=1):', tr.views_enabled())
os.environ.pop('HS_READONLY_VIEWS')
print('readonly_views from:', __import__('hs_connectors.readonly_views', fromlist=['x']).__file__)
"
