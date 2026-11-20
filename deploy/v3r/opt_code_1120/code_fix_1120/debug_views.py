import json
import mmap
import os
import struct
import sys

sys.path.insert(0, "/vllm-workspace/speculators/hs_connectors/src")

p = sys.argv[1]
fd = os.open(p, os.O_RDONLY)
mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
hlen = struct.unpack("<Q", mm[:8])[0]
print("file:", os.path.basename(p), "size:", len(mm), "hlen:", hlen)
header = json.loads(bytes(mm[8 : 8 + hlen]))
for k, v in header.items():
    if isinstance(v, dict):
        print(f"  {k}: dtype={v.get('dtype')} shape={v.get('shape')} offsets={v.get('data_offsets')}")
    else:
        print(f"  {k}: {v}")

import numpy as np
import torch
import warnings

print("numpy", np.__version__, "torch", torch.__version__)
mv = memoryview(mm)
data_start = 8 + hlen
first = next(k for k, v in header.items() if isinstance(v, dict))
meta = header[first]
begin, end = meta["data_offsets"]
arr = np.frombuffer(mv, dtype=np.uint16, count=(end - begin) // 2, offset=data_start + begin)
print("np.frombuffer ok, writeable:", arr.flags.writeable)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # escalate warnings to see them
        t = torch.from_numpy(arr)
    print("torch.from_numpy OK (no warning)")
except Exception as e:
    print("torch.from_numpy raised/warned:", type(e).__name__, e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            t = torch.from_numpy(arr)
            print("torch.from_numpy OK when warnings ignored")
        except Exception as e2:
            print("torch.from_numpy FAILED even ignoring:", type(e2).__name__, e2)
mm.close()
os.close(fd)
