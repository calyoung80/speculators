import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/vllm-workspace/speculators/hs_connectors/src")
sys.path.insert(0, "/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120")

import test_page_cache_drop as t

tmp = Path(tempfile.mkdtemp(prefix="pcd_ut8_"))
t.ut8_load_file_semantics(tmp)
failed = [r for r in t.RESULTS if not r[1]]
print(f"UT8 result: {t.RESULTS}")
sys.exit(1 if failed else 0)
