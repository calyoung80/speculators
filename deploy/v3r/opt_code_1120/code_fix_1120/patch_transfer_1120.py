import shutil

SRC = "/vllm-workspace/speculators/hs_connectors/src/hs_connectors"
FIX = "/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120"
tp = f"{SRC}/transfer.py"
orig = open(tp).read()

old_func = """def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

    return None"""
new_func = """def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        sample = load_file(file_path)
        drop_page_cache(file_path)
        return sample

    return None"""
assert orig.count(old_func) == 1, "func anchor not unique"
anchor = "from hs_connectors.mooncake_te_store import MooncakeTEConfig, MooncakeTEStore"
assert orig.count(anchor) == 1, "import anchor not unique"
guarded = anchor + """

try:  # 1120 fix: drop states page cache after load (opt-in via HS_FADVISE_DONTNEED=1)
    from hs_connectors.page_cache_drop import drop_page_cache
except Exception:  # pragma: no cover -- training must never die on this
    def drop_page_cache(*_args, **_kwargs):  # type: ignore[misc]
        return False"""
patched = orig.replace(anchor, guarded).replace(old_func, new_func)
open(tp, "w").write(patched)
shutil.copyfile(f"{FIX}/page_cache_drop.py", f"{SRC}/page_cache_drop.py")
print("PATCH_APPLIED")
