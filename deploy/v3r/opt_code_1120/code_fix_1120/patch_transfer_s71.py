"""Patch s71 container: ORIGINAL -> v2 (views + fadvise) in one step.

s71 was unreachable during both v1 and v2 deployments, so its container still
has the pristine hs_connectors. This patch applies the complete v2 change set
using the original anchors. Idempotence: refuses if v2 markers already exist.
"""

SRC = "/vllm-workspace/speculators/hs_connectors/src/hs_connectors"
tp = f"{SRC}/transfer.py"
src = open(tp).read()

if "views_enabled" in src:
    print("ALREADY_PATCHED_V2")
    raise SystemExit(0)

old_imp = "from hs_connectors.mooncake_te_store import MooncakeTEConfig, MooncakeTEStore"
new_imp = old_imp + """

try:  # 1120 fix: page-cache drop + read-only views (opt-in env, default off)
    from hs_connectors.page_cache_drop import drop_page_cache
    from hs_connectors.readonly_views import load_file_views, views_enabled
except Exception:  # pragma: no cover -- training must never die on this
    def drop_page_cache(*_args, **_kwargs):  # type: ignore[misc]
        return False

    def load_file_views(*_args, **_kwargs):  # type: ignore[misc]
        return None

    def views_enabled() -> bool:  # type: ignore[misc]
        return False"""

old_fn = """def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

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

assert src.count(old_imp) == 1, "import anchor not unique"
assert src.count(old_fn) == 1, "function anchor not unique"
open(tp, "w").write(src.replace(old_imp, new_imp).replace(old_fn, new_fn))
print("PATCH_S71_V2_APPLIED")
