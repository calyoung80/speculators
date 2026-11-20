import shutil

SRC = "/vllm-workspace/speculators/hs_connectors/src/hs_connectors"
FIX = "/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120"
tp = f"{SRC}/transfer.py"
orig = open(tp).read()

# --- v1 -> v2: import block gains views symbols (with safe fallbacks) ---
old_imp = """try:  # 1120 fix: drop states page cache after load (opt-in via HS_FADVISE_DONTNEED=1)
    from hs_connectors.page_cache_drop import drop_page_cache
except Exception:  # pragma: no cover -- training must never die on this
    def drop_page_cache(*_args, **_kwargs):  # type: ignore[misc]
        return False"""
new_imp = """try:  # 1120 fix: page-cache drop + read-only views (opt-in env, default off)
    from hs_connectors.page_cache_drop import drop_page_cache
    from hs_connectors.readonly_views import load_file_views, views_enabled
except Exception:  # pragma: no cover -- training must never die on this
    def drop_page_cache(*_args, **_kwargs):  # type: ignore[misc]
        return False

    def load_file_views(*_args, **_kwargs):  # type: ignore[misc]
        return None

    def views_enabled() -> bool:  # type: ignore[misc]
        return False"""
assert orig.count(old_imp) == 1, "import anchor (v1) not unique"

# --- v1 -> v2: _load_hs_file gains the views branch with fallback ---
old_fn = """def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        sample = load_file(file_path)
        drop_page_cache(file_path)
        return sample

    return None"""
new_fn = """def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        if views_enabled():
            sample = load_file_views(file_path)
            if sample is not None:
                drop_page_cache(file_path)
                return sample
        sample = load_file(file_path)
        drop_page_cache(file_path)
        return sample

    return None"""
assert orig.count(old_fn) == 1, "function anchor (v1) not unique"

patched = orig.replace(old_imp, new_imp).replace(old_fn, new_fn)
open(tp, "w").write(patched)
shutil.copyfile(f"{FIX}/hs_readonly_views.py", f"{SRC}/readonly_views.py")
print("PATCH_V2_APPLIED")
