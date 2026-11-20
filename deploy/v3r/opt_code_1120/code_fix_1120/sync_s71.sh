#!/bin/bash
set -e
DST=/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120
SRC=/vllm-workspace/speculators/hs_connectors/src/hs_connectors
docker exec train_ddp cp "$DST/page_cache_drop.py" "$SRC/page_cache_drop.py"
docker exec train_ddp cp "$DST/hs_readonly_views.py" "$SRC/readonly_views.py"
docker exec train_ddp python3 "$DST/patch_transfer_s71.py"
docker exec train_ddp python3 -m py_compile "$SRC/transfer.py" "$SRC/readonly_views.py" "$SRC/page_cache_drop.py"
echo S71_SYNC_OK
