"""Best-effort page-cache drop for hidden-state files.

Motivation (2026-11-19 slide #8): states files read from the 2.3T cephfs pool
stay in the kernel page cache with no eviction policy, while worker copies
live in anonymous memory. Combined footprint (~1.3T) exceeds the 1T hosts and
any ceph osd flap pins the file LRU, collapsing MemAvailable in seconds.

POSIX_FADV_DONTNEED asks the kernel to drop the clean page-cache copy right
after each load. Each states file is read once per epoch (hours apart), so
dropping has no re-read throughput cost.

Opt-in via HS_FADVISE_DONTNEED=1 (default off = byte-identical behavior).
This module must never raise and never alter training data semantics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_ENV_SWITCH = "HS_FADVISE_DONTNEED"
_ENV_MIN_BYTES = "HS_FADVISE_MIN_BYTES"
_DEFAULT_MIN_BYTES = 64 * 1024 * 1024  # skip lock files / configs / tiny tensors

try:
    _FADV_DONTNEED = os.POSIX_FADV_DONTNEED
except AttributeError:  # non-POSIX platform: feature disabled
    _FADV_DONTNEED = None


def fadvise_enabled() -> bool:
    """True iff HS_FADVISE_DONTNEED == '1' (read fresh each call)."""
    return os.environ.get(_ENV_SWITCH, "") == "1"


def _resolve_min_bytes(min_bytes: int | None) -> int:
    if min_bytes is not None:
        return int(min_bytes)
    raw = os.environ.get(_ENV_MIN_BYTES, "")
    try:
        return int(raw) if raw else _DEFAULT_MIN_BYTES
    except ValueError:
        return _DEFAULT_MIN_BYTES


def drop_page_cache(path: Any, *, min_bytes: int | None = None) -> bool:
    """Request the kernel to drop clean page-cache pages of ``path``.

    Issues ``posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`` on the whole file.

    - Files smaller than ``min_bytes`` (default 64 MiB, env-overridable) are
      skipped to avoid syscall spam on small metadata files.
    - Best-effort: any error (missing file, unsupported fs, permission) is
      swallowed and ``False`` returned. Never raises, never blocks long.
    - ``True`` means the fadvise syscall was issued; the kernel may still
      keep pages that are mapped or dirty.

    No-op (returns False immediately) unless HS_FADVISE_DONTNEED=1.
    """
    if _FADV_DONTNEED is None or not fadvise_enabled():
        return False
    fd = -1
    try:
        p = Path(path)
        if p.stat().st_size < _resolve_min_bytes(min_bytes):
            return False
        fd = os.open(str(p), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, _FADV_DONTNEED)
        return True
    except Exception:
        return False
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
