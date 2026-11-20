import os
import sys

sys.path.insert(0, "/vllm-workspace/speculators/hs_connectors/src")
os.environ["HS_READONLY_VIEWS"] = "1"
os.environ["HS_VIEWS_DEBUG"] = "1"
import hs_connectors.readonly_views as rv

r = rv.load_file_views("/mnt/cephfs/glm5_3_hidden/cmpl-becece10ba5111e2-0-849da952.safetensors")
print("RESULT:", "None" if r is None else list(r.keys()))
