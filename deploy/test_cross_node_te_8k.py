#!/usr/bin/env python3
"""Verify one full 8K hidden-state transfer from the cross-node Producer."""

import json
import os
import urllib.request

import torch
import torch_npu  # noqa: F401

from hs_connectors.mooncake_te_store import MooncakeTEConfig, MooncakeTEStore


PRODUCER_ENDPOINT = os.environ.get(
    "PRODUCER_ENDPOINT", "http://71.10.29.118:18000/v1"
)
PRODUCER_IP = os.environ.get("PRODUCER_IP", "71.10.29.118")
CONSUMER_IP = os.environ.get("CONSUMER_IP", "71.10.29.119")
CONTEXT_TOKENS = 8191


def request_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        return json.load(response)


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("An NPU is required for the 8K TransferEngine check")
    torch.npu.set_device(0)

    model = request_json(f"{PRODUCER_ENDPOINT}/models")["data"][0]["id"]
    store = MooncakeTEStore(
        MooncakeTEConfig(
            local_hostname=CONSUMER_IP,
            protocol="ascend",
            producer_ip=PRODUCER_IP,
        )
    )
    store.setup()
    response = request_json(
        f"{PRODUCER_ENDPOINT}/completions",
        {
            "model": model,
            "prompt": [1] * CONTEXT_TOKENS,
            "max_tokens": 1,
            "return_token_ids": True,
        },
    )
    handle = response["kv_transfer_params"].get("hidden_states_path") or response[
        "kv_transfer_params"
    ].get("handle")
    if not handle:
        raise RuntimeError("Producer response did not include a hidden-state handle")

    try:
        sample = store.get_sample(handle, timeout=600)
        hidden_states = sample["hidden_states"]
        token_ids = sample["token_ids"]
        # vLLM extracts states for the prompt tokens; max_tokens=1 only triggers
        # decode and is not included in the returned prompt token IDs.
        if token_ids.numel() != CONTEXT_TOKENS:
            raise RuntimeError(
                f"Expected {CONTEXT_TOKENS} prompt tokens, got {token_ids.numel()}"
            )
        if hidden_states.shape[0] != token_ids.numel():
            raise RuntimeError(
                "Hidden-state and token lengths differ: "
                f"{hidden_states.shape[0]} != {token_ids.numel()}"
            )
        if hidden_states.dtype is not torch.bfloat16:
            raise RuntimeError(f"Expected bfloat16 hidden states, got {hidden_states.dtype}")
        if not torch.isfinite(hidden_states).all():
            raise RuntimeError("Transferred hidden states contain non-finite values")
        print(
            "cross_node_te_8k_ok "
            f"tokens={token_ids.numel()} shape={tuple(hidden_states.shape)} "
            f"bytes={hidden_states.nbytes}",
            flush=True,
        )
    finally:
        store.delete_sample(handle)


if __name__ == "__main__":
    main()
