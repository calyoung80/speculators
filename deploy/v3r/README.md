# v3r hidden-state memory optimization artifacts

This directory preserves the `6opt.zip` / `opt_code_1120` artifact bundle from the
v3r dual-node training run (2026-11-20). The runtime pieces have been integrated
into the repository:

| Artifact | Integrated location |
| --- | --- |
| `readonly_views.py` | `hs_connectors/src/hs_connectors/readonly_views.py` |
| `page_cache_drop.py` | `hs_connectors/src/hs_connectors/page_cache_drop.py` |
| `transfer.py` wiring | `hs_connectors/src/hs_connectors/transfer.py` |
| Regression tests | `tests/unit/hs_connectors/test_page_cache_drop.py` |

The remaining files under `opt_code_1120/` are the original deployment
snapshot, launch scripts, operational scripts, diagnostics, and design notes.
They are kept for provenance; `deploy/v3r/opt_code_1120/deployed_container/`
contains the pre-integration copies and is not imported by the package.

## Opt-in switches

Both optimizations are disabled by default and are safe no-ops unless enabled:

- `HS_READONLY_VIEWS=1` — load safetensors samples as read-only `mmap` views
  instead of copying tensors into the worker heap.
- `HS_FADVISE_DONTNEED=1` — issue `POSIX_FADV_DONTNEED` after each sample load.
  `HS_FADVISE_MIN_BYTES` overrides the default 64 MiB minimum file size.

Enable both together for the v3r memory profile:

```bash
export HS_READONLY_VIEWS=1
export HS_FADVISE_DONTNEED=1
```
