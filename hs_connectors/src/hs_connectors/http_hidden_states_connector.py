"""HTTP-backed sibling of ``MooncakeHiddenStatesConnector``.

Inherits all scheduler hooks and hidden-state extraction logic from the
mooncake connector; only the store is swapped for
:class:`~hs_connectors.http_store.HttpStore`, which PUTs packed bytes to
a plain HTTP key-value server. This avoids the Ascend fabric-memory
requirement entirely while keeping the same transfer-manifest format.

Loaded out-of-tree via ``kv_connector_module_path``; must be used with
the ``extract_hidden_states`` speculative method.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)

from hs_connectors.http_store import HttpStore, HttpStoreConfig
from hs_connectors.mooncake_hidden_states_connector import (
    MooncakeHiddenStatesConnector,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

__all__ = ["HttpHiddenStatesConnector"]

_DEFAULT_WRITER_THREADS = 4


class HttpHiddenStatesConnector(MooncakeHiddenStatesConnector):
    """Put hidden states on an HTTP key-value server instead of mooncake.

    All behavior (scheduler hooks, extraction, async writer threads) is
    inherited from :class:`MooncakeHiddenStatesConnector`; only
    ``__init__`` differs so that an :class:`HttpStore` is created from
    the ``http`` section of ``kv_connector_extra_config``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # Skip MooncakeHiddenStatesConnector.__init__ (which builds a
        # MooncakeHiddenStatesStore) and initialise from the grandparent
        # so we can substitute our own store below.
        super(MooncakeHiddenStatesConnector, self).__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        # ---- common setup (mirrors MooncakeHiddenStatesConnector) ----
        self._hs_group_idx = self._find_cache_kv_group_id(kv_cache_config)
        self._block_size = self._get_cache_block_size(
            vllm_config, kv_cache_config, self._hs_group_idx
        )
        (
            self._num_hidden_states,
            self._hidden_size,
            self._hidden_state_dtype,
        ) = self._get_hidden_state_layout(vllm_config)

        if (
            self._vllm_config.speculative_config is None
            or self._vllm_config.speculative_config.method
            != "extract_hidden_states"
        ):
            raise ValueError(
                "HttpHiddenStatesConnector requires the "
                "'extract_hidden_states' speculative method"
            )

        # ---- HTTP store instead of mooncake ---------------------------
        extra_cfg = self._kv_transfer_config.get_from_extra_config("http", {})
        http_cfg = HttpStoreConfig.from_dict(extra_cfg)
        self._store = HttpStore(http_cfg)

        # Scheduler-side state.
        self._request_keys: dict[str, str] = {}
        self._pending_saves: dict[str, Any] = {}

        # Worker-side state (set in register_kv_caches).
        self._kv_cache: Any | None = None
        self._is_tp_rank_zero: bool = True
        self._store_ready: bool = False
        self._copy_stream: Any | None = None
        self._device_module: Any | None = None
        self._num_writer_threads = int(
            extra_cfg.get("num_writer_threads", _DEFAULT_WRITER_THREADS)
        )
        self._executor: ThreadPoolExecutor | None = None
        self._req_futures: dict[str, Any] = {}
        self._accumulated_finished_req_ids: set[str] = set()
