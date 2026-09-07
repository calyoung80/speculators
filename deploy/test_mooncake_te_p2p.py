#!/usr/bin/env python3
"""Minimal same-host Ascend TransferEngine P2P diagnostic."""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from mooncake.engine import TransferEngine


def init_engine() -> TransferEngine:
    torch.npu.set_device(0)
    engine = TransferEngine()
    host = socket.gethostbyname(socket.gethostname())
    ret = engine.initialize(host, "P2PHANDSHAKE", "ascend", "")
    if ret != 0:
        raise RuntimeError(f"TransferEngine.initialize failed: ret={ret}")
    return engine


def run_producer(meta_path: Path) -> None:
    engine = init_engine()
    source = torch.full((1024,), 3.25, dtype=torch.bfloat16, device="npu:0")
    torch.npu.synchronize()
    ret = engine.register_memory(source.data_ptr(), source.nbytes)
    if ret != 0:
        raise RuntimeError(f"producer register_memory failed: ret={ret}")

    meta_path.write_text(
        json.dumps(
            {
                "session_id": (
                    f"{socket.gethostbyname(socket.gethostname())}:"
                    f"{engine.get_rpc_port()}"
                ),
                "ptr": source.data_ptr(),
                "nbytes": source.nbytes,
            }
        )
    )
    print(f"producer_ready {meta_path}", flush=True)

    done_path = meta_path.with_suffix(".done")
    deadline = time.monotonic() + 60
    while not done_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"consumer did not create {done_path}")
        time.sleep(0.1)


def run_consumer(meta_path: Path) -> None:
    deadline = time.monotonic() + 30
    while not meta_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"producer did not create {meta_path}")
        time.sleep(0.1)

    metadata = json.loads(meta_path.read_text())
    engine = init_engine()
    target = torch.empty(1024, dtype=torch.bfloat16, device="npu:0")
    ret = engine.register_memory(target.data_ptr(), target.nbytes)
    if ret != 0:
        raise RuntimeError(f"consumer register_memory failed: ret={ret}")

    ret = engine.batch_transfer_sync_read(
        metadata["session_id"],
        [target.data_ptr()],
        [metadata["ptr"]],
        [metadata["nbytes"]],
    )
    torch.npu.synchronize()
    if ret != 0:
        raise RuntimeError(f"batch_transfer_sync_read failed: ret={ret}")
    if not torch.all(target == 3.25).item():
        raise RuntimeError("transferred tensor contents do not match")

    meta_path.with_suffix(".done").write_text("ok\n")
    print("consumer_ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("producer", "consumer"))
    parser.add_argument("meta_path", type=Path)
    args = parser.parse_args()

    if args.role == "producer":
        run_producer(args.meta_path)
    else:
        run_consumer(args.meta_path)


if __name__ == "__main__":
    main()
