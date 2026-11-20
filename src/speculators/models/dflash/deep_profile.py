"""DFlash 深层打点 (env 开关, 默认关闭)。

用法: 在 dflash/model_definitions.py 与 core.py 里插入 begin/end:

    dp.begin("forward") / dp.end("forward")      # 每次 forward 一次, 控制开关与打印
    dp.begin("q_proj")   / dp.end("q_proj")      # 各子区域 (内部会 synchronize)

环境变量:
    DFLASH_DEEP_PROFILE=1        开启
    DFLASH_DEEP_PROFILE_EVERY=20 每 N 次 forward 打一次
    DFLASH_DEEP_PROFILE_SKIP=0   在第 (N*k+SKIP) 次 forward 打点

输出: 一行 `[deep] step#N name=xx.x name2=yy.y`，直接进 train_*.log。
"""

from __future__ import annotations

import contextlib
import os
import time

ENABLED = os.environ.get("DFLASH_DEEP_PROFILE", "0") == "1"
EVERY = max(int(os.environ.get("DFLASH_DEEP_PROFILE_EVERY", "20")), 1)
SKIP = int(os.environ.get("DFLASH_DEEP_PROFILE_SKIP", "0"))

_count = 0
_active = False
_t0 = 0.0
_acc: dict[str, float] = {}


def _sync() -> None:
    with contextlib.suppress(Exception):
        import torch_npu  # type: ignore[import-not-found]

        torch_npu.npu.synchronize()
        return
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.synchronize()


def begin(name: str) -> None:
    global _count, _active, _t0  # noqa: PLW0603
    if name == "forward":
        _count += 1
        _active = ENABLED and (_count % EVERY == SKIP)
        _acc.clear()
        if _active:
            _sync()
            _t0 = time.perf_counter()
        return
    if not _active:
        return
    _sync()
    _t0 = time.perf_counter()


def end(name: str) -> None:
    global _active, _t0  # noqa: PLW0603
    if name == "forward":
        if not _active:
            return
        _sync()
        if _acc:
            body = " ".join(f"{k}={v:.1f}" for k, v in sorted(_acc.items()))
            print(f"[deep] forward#{_count} {body}", flush=True)  # noqa: T201
        _active = False
        _t0 = 0.0
        return
    if not _active or _t0 == 0.0:
        return
    _sync()
    _acc[name] = _acc.get(name, 0.0) + (time.perf_counter() - _t0) * 1e3
    _t0 = time.perf_counter()  # 供下一个区域用 (区域之间是顺序的)
