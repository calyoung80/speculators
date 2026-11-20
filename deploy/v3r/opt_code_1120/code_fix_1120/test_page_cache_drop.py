"""Unit + integration tests for page_cache_drop (UT) and readonly_views (VUT).

Run (inside container, host idle):
    python3 test_page_cache_drop.py [cephfs_dir]

UT1-UT8: v1 fadvise patch (design doc S-C).
VUT1-VUT7: path-A read-only zero-copy views (design doc S-I).
VUT2-VUT6 need a real states file from /mnt/cephfs/glm5_3_hidden
(auto-discovered; FAIL if absent).
Exit code 0 = all pass.
"""

from __future__ import annotations

import gc
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RESULTS: list[tuple[str, bool, str]] = []

_orig_fadvise = getattr(os, "posix_fadvise", None)
_SRC = "/vllm-workspace/speculators/hs_connectors/src"
sys.path.insert(0, _SRC)

HS_POOL = "/mnt/cephfs/glm5_3_hidden"


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def with_fadvise_spy():
    calls: list[tuple] = []

    def spy(fd, off, ln, advice):
        calls.append((fd, off, ln, advice))

    os.posix_fadvise = spy

    def restore():
        if _orig_fadvise is None:
            del os.posix_fadvise
        else:
            os.posix_fadvise = _orig_fadvise

    return calls, restore


def cached_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return -1


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 1048576


def mapped_kb(path_s: str) -> tuple[int, bool]:
    """(sum of mapping Size KB for path, any mapping of path is read-only)."""
    total, readonly, cur = 0, False, None
    with open("/proc/self/smaps") as sf:
        for line in sf:
            if line[0].isdigit() or line[0] in "abcdef":
                parts = line.split()
                if len(parts) >= 6:
                    cur = parts[5]
                    if parts[1].startswith("r") and "w" not in parts[1] and cur == path_s:
                        readonly = True
            elif line.startswith("Size:") and cur == path_s:
                total += int(line.split()[1])
    return total, readonly


def make_file(dirpath: Path, size_mb: int) -> Path:
    p = dirpath / f"pcd_test_{os.getpid()}_{size_mb}mb.bin"
    chunk = b"x" * (1 << 20)
    with open(p, "wb") as f:
        for _ in range(size_mb):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    return p


def find_states_file() -> str | None:
    for pat in ("cmpl-*.safetensors", "hs_*.safetensors", "*.safetensors"):
        for p in sorted(glob.glob(f"{HS_POOL}/{pat}"))[:5]:
            if os.path.getsize(p) > 64 * 1024 * 1024:
                return p
    return None


# ---------------------------------------------------------------- UT1-UT5


def ut1_default_off(tmp: Path) -> None:
    os.environ.pop("HS_FADVISE_DONTNEED", None)
    import page_cache_drop as pcd

    f = make_file(tmp, 1)
    calls, restore = with_fadvise_spy()
    try:
        assert pcd.drop_page_cache(f) is False
        report("UT1_default_off", len(calls) == 0, f"calls={len(calls)}")
    finally:
        restore()


def ut2_enabled_calls(tmp: Path) -> None:
    os.environ["HS_FADVISE_DONTNEED"] = "1"
    import page_cache_drop as pcd

    f = make_file(tmp, 1)
    calls, restore = with_fadvise_spy()
    try:
        ok = pcd.drop_page_cache(f, min_bytes=0)
        good = (
            ok
            and len(calls) == 1
            and calls[0][1] == 0
            and calls[0][2] == 0
            and calls[0][3] == os.POSIX_FADV_DONTNEED
        )
        report("UT2_enabled_args", good, f"calls={calls}")
    finally:
        restore()
        os.environ.pop("HS_FADVISE_DONTNEED", None)


def ut3_exception_swallowed(tmp: Path) -> None:
    os.environ["HS_FADVISE_DONTNEED"] = "1"
    import page_cache_drop as pcd

    def boom(*a):
        raise OSError("injected")

    os.posix_fadvise = boom
    try:
        ok = pcd.drop_page_cache(make_file(tmp, 1), min_bytes=0)
        report("UT3_exception_swallowed", ok is False)
    finally:
        if _orig_fadvise is not None:
            os.posix_fadvise = _orig_fadvise
        os.environ.pop("HS_FADVISE_DONTNEED", None)


def ut4_missing_file(tmp: Path) -> None:
    os.environ["HS_FADVISE_DONTNEED"] = "1"
    import page_cache_drop as pcd

    try:
        r = pcd.drop_page_cache(tmp / "no_such_hs_file.safetensors")
        report("UT4_missing_file", r is False)
    finally:
        os.environ.pop("HS_FADVISE_DONTNEED", None)


def ut5_small_file_guard(tmp: Path) -> None:
    os.environ["HS_FADVISE_DONTNEED"] = "1"
    import page_cache_drop as pcd

    small = tmp / "tiny.bin"
    small.write_bytes(b"y" * 1024)
    calls, restore = with_fadvise_spy()
    try:
        r = pcd.drop_page_cache(small)
        report("UT5_small_file_guard", r is False and len(calls) == 0)
    finally:
        restore()
        os.environ.pop("HS_FADVISE_DONTNEED", None)


# ------------------------------------------------------------ UT6/UT7


def residency_drop_test(name: str, dirpath: Path, size_mb: int = 256) -> None:
    os.environ["HS_FADVISE_DONTNEED"] = "1"
    import page_cache_drop as pcd

    try:
        f = make_file(dirpath, size_mb)
        os.sync()
        with open(f, "rb") as fh:
            while fh.read(1 << 22):
                pass
        before = cached_kb()
        r = pcd.drop_page_cache(f)
        for _ in range(10):
            if cached_kb() <= before - size_mb * 1024 * 0.5:
                break
            import time

            time.sleep(0.2)
        after = cached_kb()
        dropped_mb = (before - after) / 1024
        report(name, bool(r and dropped_mb >= size_mb * 0.5),
               f"issued={r} dropped={dropped_mb:.0f}MB of {size_mb}MB")
        f.unlink(missing_ok=True)
    finally:
        os.environ.pop("HS_FADVISE_DONTNEED", None)


def ut8_load_file_semantics(tmp: Path) -> None:
    import torch
    from safetensors.torch import save_file

    os.environ["HS_FADVISE_DONTNEED"] = "1"
    import hs_connectors.transfer as tr

    try:
        p = tmp / "hs_test_sample.safetensors"
        save_file({"hidden_states": torch.zeros(64 * 1024 * 1024, dtype=torch.uint8)}, str(p))
        os.sync()
        with open(p, "rb") as fh:
            while fh.read(1 << 22):
                pass
        calls, restore = with_fadvise_spy()
        try:
            sample = tr._load_hs_file(p)
        finally:
            restore()
        tensors_ok = sample is not None and sample["hidden_states"].shape[0] == 64 * 1024 * 1024
        fsize = os.path.getsize(p)
        view_mode = mapped_kb(str(p))[0] >= fsize * 0.9 / 1024
        del sample
        gc.collect()
        after_gc = mapped_kb(str(p))[0]
        ok = tensors_ok and len(calls) == 1
        report("UT8_load_file_semantics", ok,
               f"tensors_ok={tensors_ok} fadvise_calls={len(calls)} "
               f"load_file={'mmap-view' if view_mode else 'copy'} mapped_after_gc={after_gc}KB")
        p.unlink(missing_ok=True)
    finally:
        os.environ.pop("HS_FADVISE_DONTNEED", None)


# ---------------------------------------------------------------- VUT1-7


def vut1_default_off(real: str) -> None:
    os.environ.pop("HS_READONLY_VIEWS", None)
    import hs_connectors.transfer as tr

    seen: list[str] = []
    orig = tr.load_file_views

    def spy(p):
        seen.append(str(p))
        return orig(p)

    tr.load_file_views = spy
    try:
        s = tr._load_hs_file(Path(real))
        ok = (not seen) and s is not None and "hidden_states" in s
        report("VUT1_default_off", ok, f"views_calls={len(seen)} sample_ok={s is not None}")
        del s
        gc.collect()
    finally:
        tr.load_file_views = orig


def vut2_value_equivalence(real: str) -> None:
    import torch
    from safetensors.torch import load_file
    import hs_connectors.readonly_views as rv

    os.environ["HS_READONLY_VIEWS"] = "1"
    try:
        sv = rv.load_file_views(real)
        sl = load_file(real)
        if sv is None:
            report("VUT2_value_equivalence", False, "views returned None")
            return
        keys_ok = set(sv) == set(sl)
        eq = keys_ok
        detail = []
        for k in sl:
            a, b = sv[k], sl[k]
            same = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            eq = eq and same
            detail.append(f"{k}:{same}")
        report("VUT2_value_equivalence", eq, " ".join(detail))
        del sv, sl
        gc.collect()
    finally:
        os.environ.pop("HS_READONLY_VIEWS", None)


def vut3_zero_copy(real: str) -> None:
    import hs_connectors.readonly_views as rv

    os.environ["HS_READONLY_VIEWS"] = "1"
    try:
        before = rss_mb()
        sv = rv.load_file_views(real)
        if sv is None:
            report("VUT3_zero_copy", False, "views returned None")
            return
        after = rss_mb()
        delta = after - before
        fsize_mb = os.path.getsize(real) / 1048576
        mkb, ro = mapped_kb(real)
        ok = delta < 32 and mkb >= fsize_mb * 0.9 / 1024 and ro
        report("VUT3_zero_copy", ok,
               f"rss_delta={delta:.1f}MB file={fsize_mb:.0f}MB mapped={mkb}KB readonly={ro}")
        del sv
        gc.collect()
    finally:
        os.environ.pop("HS_READONLY_VIEWS", None)


def vut4_lifetime_finalizer(real: str) -> None:
    import hs_connectors.readonly_views as rv

    os.environ["HS_READONLY_VIEWS"] = "1"
    os.environ["HS_FADVISE_DONTNEED"] = "1"
    calls, restore = with_fadvise_spy()
    try:
        sv = rv.load_file_views(real)
        if sv is None:
            report("VUT4_lifetime_finalizer", False, "views returned None")
            return
        before_map = mapped_kb(real)[0]
        c0 = len(calls)
        del sv
        gc.collect()
        import time

        time.sleep(0.5)
        gc.collect()
        after_map = mapped_kb(real)[0]
        c1 = len(calls)
        ok = before_map > 0 and after_map == 0 and c1 == c0 + 1
        report("VUT4_lifetime_finalizer", ok,
               f"mapped {before_map}KB->{after_map}KB fadvise {c0}->{c1}")
    finally:
        restore()
        os.environ.pop("HS_READONLY_VIEWS", None)
        os.environ.pop("HS_FADVISE_DONTNEED", None)


def vut5_write_protection(real: str) -> None:
    code = (
        "import sys;"
        f"sys.path.insert(0, '{_SRC}');"
        "import os; os.environ['HS_READONLY_VIEWS']='1';"
        "import hs_connectors.readonly_views as rv;"
        f"t = rv.load_file_views(r'{real}')['hidden_states'];"
        "t[0,0] += 1; print('SILENT_WRITE')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    silent = "SILENT_WRITE" in (r.stdout or "")
    segv = r.returncode < 0
    ok = (not silent) and r.returncode != 0
    report("VUT5_write_protection", ok,
           f"rc={r.returncode} mode={'segfault' if segv else 'exception'} "
           f"silent_write={silent}")


def vut6_end_to_end(real: str) -> None:
    import torch
    from safetensors.torch import load_file
    import hs_connectors.transfer as tr

    os.environ["HS_READONLY_VIEWS"] = "1"
    os.environ["HS_FADVISE_DONTNEED"] = "1"
    try:
        s = tr._load_hs_file(Path(real))
        sl = load_file(real)
        if s is None:
            report("VUT6_end_to_end", False, "_load_hs_file returned None")
            return
        eq = True
        for k in sl:
            eq = eq and torch.equal(s[k], sl[k])
        tid_ok = str(s.get("token_ids").dtype) == str(sl.get("token_ids").dtype)
        report("VUT6_end_to_end", eq and tid_ok, f"values_eq={eq} token_ids_dtype_ok={tid_ok}")
        del s, sl
        gc.collect()
    finally:
        os.environ.pop("HS_READONLY_VIEWS", None)
        os.environ.pop("HS_FADVISE_DONTNEED", None)


def vut7_corrupt_fallback(tmp: Path) -> None:
    import hs_connectors.readonly_views as rv

    os.environ["HS_READONLY_VIEWS"] = "1"
    try:
        bad = tmp / "corrupt.safetensors"
        bad.write_bytes(b"\xff" * 16 + b"\x00" * 64)
        r = rv.load_file_views(bad)
        report("VUT7_corrupt_fallback", r is None, f"returned={type(r)}")
    finally:
        os.environ.pop("HS_READONLY_VIEWS", None)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pcd_ut_"))
    ceph_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    ut1_default_off(tmp)
    ut2_enabled_calls(tmp)
    ut3_exception_swallowed(tmp)
    ut4_missing_file(tmp)
    ut5_small_file_guard(tmp)
    residency_drop_test("UT6_local_disk", tmp)
    if ceph_dir and ceph_dir.exists():
        residency_drop_test("UT7_cephfs", ceph_dir)
    else:
        report("UT7_cephfs", False, "SKIPPED: no cephfs dir argument")
    ut8_load_file_semantics(tmp)

    real = find_states_file()
    if real:
        vut1_default_off(real)
        vut2_value_equivalence(real)
        vut3_zero_copy(real)
        vut4_lifetime_finalizer(real)
        vut5_write_protection(real)
        vut6_end_to_end(real)
    else:
        report("VUT_suite", False, "SKIPPED: no states file in pool")
    vut7_corrupt_fallback(tmp)

    os.environ.pop("HS_FADVISE_DONTNEED", None)
    os.environ.pop("HS_READONLY_VIEWS", None)
    import importlib

    import page_cache_drop as pcd

    importlib.reload(pcd)
    report("UT0_module_import", callable(pcd.drop_page_cache))

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n=== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
