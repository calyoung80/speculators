"""TransferEngine-backed vLLM hidden-states connector.

Same KVConnectorBase_V1 interface as ``MooncakeHiddenStatesConnector``,
but keeps hidden states on NPU and registers them with
``TransferEngine.register_memory`` instead of doing a DtoH copy +
``MooncakeDistributedStore.put_tensor``.

Loaded out-of-tree via ``kv_connector_module_path``; must be used with
the ``extract_hidden_states`` speculative method.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from hs_connectors.mooncake_te_store import MooncakeTEConfig, MooncakeTEStore

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def extract_from_kv_cache(
    kv_cache: torch.Tensor, slot_mapping: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    block_size = kv_cache.shape[1]
    return kv_cache[slot_mapping // block_size, slot_mapping % block_size][:num_tokens]


def sanitize_key(key: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    return ("k" + safe) if safe[:1].isdigit() else safe


@dataclass
class PendingSave:
    req_id: str
    te_key: str
    token_ids: torch.Tensor
    block_ids: list[int]
    slot_mapping: torch.Tensor


@dataclass
class ReqMeta:
    req_id: str
    token_ids: torch.Tensor

    @staticmethod
    def make_meta(req_id: str, token_ids: list[int]) -> "ReqMeta":
        return ReqMeta(req_id=req_id, token_ids=torch.tensor(token_ids))


@dataclass
class MooncakeTEConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)


class MooncakeTEHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """Stores hidden states on NPU via TransferEngine (no DtoH copy)."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    @classmethod
    def _find_cache_kv_group_id(cls, kv_cache_config: KVCacheConfig | None) -> int:
        if kv_cache_config is None:
            return 0
        from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

        groups = kv_cache_config.kv_cache_groups
        group_ids = [
            gid
            for gid, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        ]
        if len(group_ids) == 1:
            return group_ids[0]
        if not group_ids and len(groups) == 1:
            return 0
        raise ValueError(
            "Could not uniquely identify the extract-hidden-states KV cache "
            f"group among {len(groups)} groups."
        )

    @staticmethod
    def _get_cache_block_size(
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
        cache_kv_group_id: int,
    ) -> int:
        if kv_cache_config is None:
            return vllm_config.cache_config.block_size
        cache_group = kv_cache_config.kv_cache_groups[cache_kv_group_id]
        return cache_group.kv_cache_spec.block_size

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._hs_group_idx = self._find_cache_kv_group_id(kv_cache_config)
        self._block_size = self._get_cache_block_size(
            vllm_config, kv_cache_config, self._hs_group_idx
        )

        if (
            self._vllm_config.speculative_config is None
            or self._vllm_config.speculative_config.method != "extract_hidden_states"
        ):
            raise ValueError(
                "MooncakeTEHiddenStatesConnector requires the "
                "'extract_hidden_states' speculative method"
            )

        te_cfg = MooncakeTEConfig.from_dict(
            self._kv_transfer_config.get_from_extra_config("mooncake_te", {})
        )
        self._store = MooncakeTEStore(te_cfg)

        if role == KVConnectorRole.WORKER:
            tp_rank = get_tensor_model_parallel_rank()
            if tp_rank == 0:
                self._store.setup()
                self._store_ready = True
            else:
                self._store_ready = False
        else:
            self._store_ready = False

        self._request_keys: dict[str, str] = {}
        self._pending_saves: dict[str, PendingSave] = {}

        self._kv_cache: torch.Tensor | None = None
        self._is_tp_rank_zero: bool = True
        self._accumulated_finished_req_ids: set[str] = set()
        self._saves_done: bool = False
        self._prev_te_key: str | None = None

        if role == KVConnectorRole.WORKER:
            self._write_executor = ThreadPoolExecutor(max_workers=1)
        else:
            self._write_executor = None
        self._pending_futures: dict[str, Future] = {}

    @property
    def _stream_mod(self):
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu
        return torch.cuda

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        if self._store_ready and self._store._send_hs_buffer is not None:
            self._store.reset_send_buffer()

        # Option B: wait for previous request's ACK before this request's
        # unified_kv_cache_update overwrites KV cache blocks.
        # Only check on batches with new requests (prefill), not decode batches.
        prev_key = self._prev_te_key
        if prev_key is not None and self._store_ready:
            try:
                meta = self._get_connector_metadata()
                has_new = hasattr(meta, "requests") and meta.requests
            except Exception:
                has_new = False
            if has_new:
                # Skip ACK check if previous key's metadata no longer exists
                # (cleaned between training runs)
                prev_meta = f"/tmp/te_meta/{prev_key}.json"
                if not os.path.exists(prev_meta):
                    self._prev_te_key = None
                else:
                    ack_path = f"/tmp/te_meta/{prev_key}.ack"
                    t0 = time.perf_counter()
                    ack_timeout = float(os.environ.get("TE_ACK_TIMEOUT", "5"))
                    while not os.path.exists(ack_path):
                        if time.perf_counter() - t0 > ack_timeout:
                            logger.warning(
                                "ACK timeout for prev_key=%s after %.1fs",
                                prev_key, ack_timeout,
                            )
                            break
                        time.sleep(0.01)
                    else:
                        try:
                            os.remove(ack_path)
                        except OSError:
                            pass
                    self._prev_te_key = None

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self._cache_layers:
            return
        if not self._store_ready:
            return

        from vllm.forward_context import get_forward_context
        from vllm.model_executor.models.extract_hidden_states import (  # noqa: PLC0415
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        if not hasattr(connector_metadata, "requests") or not connector_metadata.requests:
            return

        slot_mapping = get_forward_context().slot_mapping[layer_name]
        offset = 0
        for request in connector_metadata.requests:
            num_tokens = request.token_ids.shape[0]
            req_slot_mapping = slot_mapping[offset : offset + num_tokens]
            offset += num_tokens

            req_id = request.req_id
            key = sanitize_key(req_id)

            blocks_per_token = (req_slot_mapping.cpu() // self._block_size).tolist()
            seen = set()
            block_ids = []
            for b in blocks_per_token:
                if b not in seen:
                    seen.add(b)
                    block_ids.append(b)
            pending = PendingSave(
                req_id=req_id,
                te_key=key,
                token_ids=request.token_ids,
                block_ids=block_ids,
                slot_mapping=req_slot_mapping,
            )
            if self._write_executor is not None:
                self._write_sample(pending)

    def wait_for_save(self) -> None:
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0

        from vllm.model_executor.models.extract_hidden_states import (  # noqa: PLC0415
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        cache_layers = list(layers.keys())
        assert len(cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(cache_layers)}"
        )
        self._kv_cache = kv_caches[cache_layers[0]]
        self._cache_layers = cache_layers

        # Register KV cache with TransferEngine (vllm-ascend pattern:
        # register once in register_kv_caches, use base_addr+offset for transfer)
        if self._store.is_setup:
            self._store.register_kv_cache(self._kv_cache)

    def _ensure_store(self) -> None:
        if not self._store_ready:
            self._store.setup()
            self._store_ready = True
            # If kv_cache already available (register_kv_caches called before setup),
            # register it now
            if self._kv_cache is not None:
                self._store.register_kv_cache(self._kv_cache)

    def _write_sample(self, pending: PendingSave) -> None:
        num_tokens = len(pending.token_ids)

        # Block-based approach: write KV cache block addresses directly
        block_size = self._kv_cache.shape[1]
        slots = pending.slot_mapping
        block_ids = (slots // block_size).unique().tolist()

        blocks = []
        per_token_bytes = self._kv_cache.element_size() * self._kv_cache.shape[2:].numel()
        for bid in block_ids:
            block_base = self._kv_cache.data_ptr() + bid * block_size * per_token_bytes
            block_bytes = block_size * per_token_bytes
            blocks.append({"ptr": block_base, "size": block_bytes})

        tensor_specs = {
            "hidden_states": {
                "blocks": blocks,
                "per_token_bytes": per_token_bytes,
                "shape": [num_tokens, self._kv_cache.shape[2], self._kv_cache.shape[3]],
                "dtype": str(self._kv_cache.dtype),
            },
        }

        tid_list = pending.token_ids.detach().contiguous().view(-1).to(torch.long).cpu().tolist()
        tensor_specs["token_ids"] = {
            "data": tid_list,
            "shape": list(pending.token_ids.shape), "dtype": str(pending.token_ids.dtype),
        }

        self._store.put_sample(pending.te_key, tensor_specs)
        self._prev_te_key = pending.te_key

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        self._accumulated_finished_req_ids.update(finished_req_ids)
        done = set(self._accumulated_finished_req_ids)
        self._accumulated_finished_req_ids.clear()
        return done or None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: Request,  # noqa: ARG002
        num_computed_tokens: int,  # noqa: ARG002
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,  # noqa: ARG002
        blocks: KVCacheBlocks,  # noqa: ARG002
        num_external_tokens: int,
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = MooncakeTEConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            meta.requests.append(ReqMeta.make_meta(new_req.req_id, token_ids))
            self._request_keys[new_req.req_id] = sanitize_key(new_req.req_id)
        return meta

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        key = sanitize_key(request.request_id)
        return True, {"handle": key}

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[self._hs_group_idx])

    @classmethod
    def get_required_kvcache_layout(
        cls,
        vllm_config: VllmConfig,  # noqa: ARG003
    ) -> str | None:
        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called on the base class"
            )
        return "NHD"
