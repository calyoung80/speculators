"""Read-only zero-copy safetensors loading via PROT_READ mmap views.

Path-A fix (design doc S13/I, 2026-11-20): ``safetensors.torch.load_file``
COPIES every tensor into the worker heap. With ~1 GB per states sample that
copy is the dominant anonymous-memory term (Anon ~807G at the 11-19 22:36
outage). This module returns tensors that VIEW a read-only mapping instead,
so the sample's bytes exist exactly once, shared with the page cache.

Lessons baked in (community research, design doc S-F.2):
- PROT_READ only. safetensors' default private-writable mapping can be
  copy-on-write'd into anonymous memory that never drops under pressure
  (measured by sglang#37822); a read-only mapping cannot.
- Delayed page-cache drop: when the last tensor viewing the mapping dies,
  a weakref finalizer issues POSIX_FADV_DONTNEED for the file (only when
  HS_FADVISE_DONTNEED=1; orthogonal opt-in).

Opt-in via HS_READONLY_VIEWS=1 (default off = unchanged load_file behavior).
Any parse/IO error returns None so callers fall back to load_file. Never
raises. numpy is required (available in the training image).
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import warnings
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from hs_connectors.page_cache_drop import drop_page_cache
except Exception:  # pragma: no cover -- standalone testing

    def drop_page_cache(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore[misc]
        return False


_ENV_SWITCH = "HS_READONLY_VIEWS"

# safetensors dtype -> (numpy dtype, torch dtype or None when identity)
_DTYPES: dict[str, tuple[np.dtype, torch.dtype | None]] = {
    "BF16": (np.dtype(np.uint16), torch.bfloat16),
    "F16": (np.dtype(np.float16), None),
    "F32": (np.dtype(np.float32), None),
    "F64": (np.dtype(np.float64), None),
    "I64": (np.dtype(np.int64), None),
    "I32": (np.dtype(np.int32), None),
    "I16": (np.dtype(np.int16), None),
    "I8": (np.dtype(np.int8), None),
    "U8": (np.dtype(np.uint8), None),
    "BOOL": (np.dtype(np.bool_), None),
}


class _MMapHolder:
    """Weakref-able anchor keeping the mapping alive via tensor attributes."""

    __slots__ = ("mm", "fd", "__weakref__")


def views_enabled() -> bool:
    return os.environ.get(_ENV_SWITCH, "") == "1"


def _parse_header(mm: mmap.mmap) -> tuple[dict[str, Any], int]:
    if len(mm) < 8:
        raise ValueError("file too small for safetensors header")
    hlen = struct.unpack("<Q", mm[:8])[0]
    if hlen <= 0 or 8 + hlen > len(mm):
        raise ValueError(f"implausible header length {hlen}")
    header = json.loads(bytes(mm[8 : 8 + hlen]))
    return header, 8 + hlen


def load_file_views(path: Any) -> dict[str, torch.Tensor] | None:
    """Return zero-copy read-only views of every tensor in a safetensors file.

    Lifetime: each returned tensor keeps (via attribute) a holder that keeps
    the mmap open; when all tensors die the mapping closes and a weakref
    finalizer best-effort drops the page cache. Returns None on any error.
    """
    fd = -1
    try:
        if not views_enabled():
            return None
        p = Path(path)
        if not p.is_file():
            return None
        fd = os.open(str(p), os.O_RDONLY)
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        try:
            header, data_start = _parse_header(mm)
        except Exception:
            mm.close()
            return None

        holder = _MMapHolder()
        holder.mm = mm
        holder.fd = fd
        fd = -1  # ownership moved to holder
        weakref.finalize(holder, drop_page_cache, str(p))

        mv = memoryview(mm)
        out: dict[str, torch.Tensor] = {}
        for name, meta in header.items():
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                continue  # e.g. __metadata__
            np_dt, t_dt = _DTYPES.get(meta.get("dtype", ""))
            if np_dt is None:
                mm.close()
                return None  # exotic dtype -> caller falls back to load_file
            begin, end = meta["data_offsets"]
            nbytes = end - begin
            itemsize = np_dt.itemsize
            if nbytes % itemsize:
                mm.close()
                return None
            count = nbytes // itemsize
            arr = np.frombuffer(mv, dtype=np_dt, count=count, offset=data_start + begin)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # read-only numpy array warning
                t = torch.from_numpy(arr)
            if t_dt is not None:
                t = t.view(t_dt)
            shape = meta.get("shape") or []
            t = t.reshape(shape)
            try:
                setattr(t, "_hs_holder", holder)
            except Exception:
                pass  # attribute anchor unavailable; mmap still anchored by arr chain
            out[str(name)] = t
        return out
    except Exception:
        if os.environ.get("HS_VIEWS_DEBUG") == "1":
            import traceback

            traceback.print_exc()
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
