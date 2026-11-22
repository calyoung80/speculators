"""HTTP-backed hidden-states store (optimized).

Stores the tensor payload as packed bytes on a remote HTTP key-value
server. Shares the transfer-manifest format with the mooncake backend
(``packed_layout``/``VERSION``), so the vLLM producer and the trainer
only differ in which store they talk to.

Optimizations over the naive version:
  * Bounded connection pool with keep-alive reuse (no per-request TCP
    handshake).
  * Chunked parallel PUT/GET: payloads above ``max_chunk_bytes`` are
    split into chunks of that size and all chunks are transferred
    concurrently (bounded by the connection pool size).
  * Zero-copy staging on the consumer side (``torch.frombuffer``).
  * The HTTP server evicts by total bytes (``--max-bytes``), not just
    entry count, and GET is read-once (auto-delete).

Use it with ``scripts/hs_http_server.py``.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import http.client
import numpy as np
import torch

from hs_connectors.mooncake_store import (
    VERSION,
    MooncakeIntegrityError,
    _parse_dtype,
    assert_finite,  # noqa: F401  (re-exported for the connector)
    packed_layout,  # noqa: F401  (re-exported for the connector)
)

__all__ = [
    "HttpStoreConfig",
    "HttpStore",
    "HttpStoreError",
    "assert_finite",
    "packed_layout",
    "VERSION",
]

_HTTP_TIMEOUT = 60.0
_HTTP_TIMEOUT_DELETE = 10.0


class HttpStoreError(RuntimeError):
    """Raised when the HTTP key-value server rejects or loses a sample."""


# ---------------------------------------------------------------------------
# Connection pool (keep-alive reuse, bounded concurrency)
# ---------------------------------------------------------------------------


class _ConnPool:
    """Thread-safe bounded pool of reusable ``http.client`` connections.

    Each thread borrows a connection via :meth:`connection` and returns
    it to the pool on success.  Stale connections are detected on use
    and transparently re-established by the caller's retry logic.
    """

    def __init__(self, endpoint: str, max_size: int = 8):
        host_port = endpoint.removeprefix("http://").removeprefix("https://")
        host, _, port = host_port.partition(":")
        self._host = host
        self._port = int(port) if port else 80
        self._max_size = max_size
        self._conns: list[http.client.HTTPConnection] = []
        self._lock = threading.Lock()

    def _new_conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self._host, self._port, timeout=_HTTP_TIMEOUT)

    @contextmanager
    def connection(self):
        with self._lock:
            conn = self._conns.pop() if self._conns else None
        if conn is None:
            conn = self._new_conn()
        pooled = False
        try:
            yield conn
            # Return healthy connection to the pool.
            with self._lock:
                if len(self._conns) < self._max_size:
                    self._conns.append(conn)
                    pooled = True
        finally:
            if not pooled:
                conn.close()

    def close_all(self) -> None:
        with self._lock:
            for c in self._conns:
                c.close()
            self._conns.clear()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class HttpStoreConfig:
    """Configuration for :class:`HttpStore`.

    Args:
        endpoint: Base URL of the HTTP key-value server, e.g.
            ``http://10.100.1.67:50088`` (bare ``host:port`` also works).
        max_connections: Size of the keep-alive connection pool.
        chunk_threshold_bytes: Payloads larger than this are split into
            chunks and transferred in parallel.  0 disables chunking.
        chunk_size_bytes: Size of each chunk for parallel transfer.
        parallel_chunks: Max concurrent chunk connections.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:9020",
        max_connections: int = 8,
        chunk_size_bytes: int = 512 * 1024 * 1024,  # 512 MiB per chunk
    ):
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
        self.endpoint = endpoint.rstrip("/")
        self.max_connections = max_connections
        self.chunk_size_bytes = chunk_size_bytes

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HttpStoreConfig":
        return cls(
            endpoint=d.get("endpoint", "http://127.0.0.1:9020"),
            max_connections=int(d.get("max_connections", 8)),
            chunk_size_bytes=int(d.get("chunk_size_bytes", 512 * 1024 * 1024)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "max_connections": self.max_connections,
            "chunk_size_bytes": self.chunk_size_bytes,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class HttpStore:
    """HTTP PUT/GET/DELETE hidden-states store (pooled + chunked).

    Presents the same interface as
    :class:`~hs_connectors.mooncake_store.MooncakeHiddenStatesStore` so
    it can back :class:`~hs_connectors.http_hidden_states_connector.
    HttpHiddenStatesConnector` on the vLLM side and
    :class:`~hs_connectors.transfer.HttpTransfer` on the trainer side.
    """

    def __init__(self, config: HttpStoreConfig):
        self.config = config
        self._ready = False
        self._pool = _ConnPool(config.endpoint, max_size=config.max_connections)
        self._chunk_executor: ThreadPoolExecutor | None = None

    def __getstate__(self) -> dict:
        """Picklable state for DataLoader workers: drop the connection
        pool (contains a threading.Lock) and the chunk executor; both
        are recreated lazily on the other side."""
        return {"config": self.config}

    def __setstate__(self, state: dict) -> None:
        self.config = state["config"]
        self._ready = False
        self._pool = _ConnPool(self.config.endpoint, max_size=self.config.max_connections)
        self._chunk_executor = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_setup(self) -> bool:
        return self._ready

    def setup(self) -> "HttpStore":
        if self._ready:
            return self
        self._ready = True
        return self

    def close(self) -> None:
        self._pool.close_all()
        if self._chunk_executor is not None:
            self._chunk_executor.shutdown(wait=True)
            self._chunk_executor = None

    def _get_chunk_executor(self) -> ThreadPoolExecutor:
        if self._chunk_executor is None:
            self._chunk_executor = ThreadPoolExecutor(
                max_workers=self.config.max_connections,
                thread_name_prefix="hs-http-chunk",
            )
        return self._chunk_executor

    # -- low-level request helpers (pooled, with stale-connection retry) ---

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        """One pooled HTTP request; retries once on stale connection."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                with self._pool.connection() as conn:
                    conn.request(method, path, body=body, headers=headers or {})
                    resp = conn.getresponse()
                    data = resp.read()
                    return resp.status, data
            except (http.client.HTTPException, OSError) as e:
                last_exc = e
                continue  # stale connection; retry with a fresh one
        raise HttpStoreError(f"HTTP {method} {path}: {last_exc}")

    def _url(self, handle: str) -> str:
        return f"/hs/{handle}"

    # -- producer side ------------------------------------------------------

    def put_sample(
        self, transfer_manifest: dict[str, Any], tensors: dict[str, Any]
    ) -> None:
        """Publish ``tensors`` (possibly on the accelerator) via HTTP PUT.

        Accepts either a single tensor or a list of tensor chunks (as
        produced by the connector's chunked gather for sequences
        > 8192 tokens); chunks are validated by summed byte coverage.

        Payloads above ``chunk_threshold_bytes`` are split further and
        uploaded with parallel connections.
        """
        handle = transfer_manifest["handle"]
        tensor_spec = transfer_manifest["tensors"]
        total_bytes = transfer_manifest["metadata"]["total_aligned_bytes"]

        buf = np.zeros(total_bytes, dtype=np.uint8)
        for name, spec in tensor_spec.items():
            tensor = tensors[name]
            if isinstance(tensor, list):
                if not tensor:
                    raise HttpStoreError(
                        f"Producer tensor {name!r} chunk list is empty"
                    )
                off = 0
                for chunk in tensor:
                    host_t = chunk.cpu().contiguous()
                    raw = host_t.view(torch.uint8).reshape(-1).numpy()
                    n = raw.shape[0]
                    buf[spec["offset"] + off : spec["offset"] + off + n] = raw
                    off += n
                if off != spec["nbytes"]:
                    raise HttpStoreError(
                        f"Producer tensor {name!r} chunks covered {off} bytes, "
                        f"but the manifest reserves {spec['nbytes']}"
                    )
            else:
                host_t = tensor.cpu().contiguous()
                raw = host_t.view(torch.uint8).reshape(-1).numpy()
                buf[spec["offset"] : spec["offset"] + spec["nbytes"]] = raw

        data = buf.tobytes()

        if len(data) > self.config.chunk_size_bytes:
            self._chunked_put(handle, data)
        else:
            status, _ = self._request(
                "PUT",
                self._url(handle),
                body=data,
                headers={"Content-Type": "application/octet-stream"},
            )
            if status != 200:
                raise HttpStoreError(f"HTTP PUT {handle}: status {status}")

    def _chunked_put(self, handle: str, data: bytes) -> None:
        """Split ``data`` into chunks and upload them in parallel."""
        csz = self.config.chunk_size_bytes
        nchunks = (len(data) + csz - 1) // csz

        def _put_chunk(idx: int) -> None:
            chunk = data[idx * csz : (idx + 1) * csz]
            status, _ = self._request(
                "PUT",
                self._url(f"{handle}/{idx}"),
                body=chunk,
                headers={"Content-Type": "application/octet-stream"},
            )
            if status != 200:
                raise HttpStoreError(f"HTTP PUT chunk {handle}/{idx}: status {status}")

        ex = self._get_chunk_executor()
        list(ex.map(_put_chunk, range(nchunks)))

        # Completion marker tells the consumer how many chunks to fetch.
        status, _ = self._request(
            "PUT",
            self._url(f"{handle}/nchunks"),
            body=str(nchunks).encode(),
            headers={"Content-Type": "text/plain"},
        )
        if status != 200:
            raise HttpStoreError(f"HTTP PUT nchunks {handle}: status {status}")

    def put_error(self, transfer_manifest: dict[str, Any], error: str) -> None:
        """Publish a small terminal marker so consumers fail fast."""
        handle = transfer_manifest["handle"] + ":error"
        payload = json.dumps({"error": error[:4096]}).encode("utf-8")
        try:
            self._request(
                "PUT",
                self._url(handle),
                body=payload,
                headers={"Content-Type": "application/json"},
            )
        except Exception:  # noqa: BLE001 - best-effort error marker
            pass

    def delete_sample(self, key: str) -> None:
        """Remove the data (all chunks) and error objects for a sample."""
        # If a chunked marker exists, delete all chunks too.
        try:
            status, data = self._request(
                "GET", self._url(f"{key}/nchunks"), headers={"Connection": "close"}
            )
            if status == 200:
                nchunks = int(data)
                for i in range(nchunks):
                    self._request("DELETE", self._url(f"{key}/{i}"))
                self._request("DELETE", self._url(f"{key}/nchunks"))
        except Exception:  # noqa: BLE001
            pass
        for k in (key, key + ":error"):
            try:
                self._request("DELETE", self._url(k))
            except Exception:  # noqa: BLE001
                pass

    # -- consumer side ------------------------------------------------------

    def get_sample(
        self,
        transfer_manifest: dict[str, Any],
        timeout: float = 120.0,
        poll_interval: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Wait for and decode a sample described by ``transfer_manifest``."""
        manifest_version = transfer_manifest.get("version", 0)
        if manifest_version != VERSION:
            raise MooncakeIntegrityError(
                f"HTTP manifest version {manifest_version} didn't match "
                f"local version {VERSION}. Manifest: {transfer_manifest}"
            )

        handle = transfer_manifest["handle"]
        tensor_specs = transfer_manifest["tensors"]

        # Try single-shot GET first; fall back to chunked protocol.
        data = self._poll_get(handle, timeout, poll_interval, prefer_chunks=True)
        if data is None:
            error = self._get_error(handle)
            if error is not None:
                raise MooncakeIntegrityError(
                    f"HTTP producer error for {handle}: {error}"
                )
            raise TimeoutError(
                f"HTTP fetch timed out for {handle} after {timeout}s"
            )

        # Zero-copy staging: torch.frombuffer shares memory with ``data``.
        # _extract_tensor().clone() makes the returned tensors independent.
        staging = torch.frombuffer(data, dtype=torch.uint8)
        return {
            name: self._extract_tensor(handle, name, staging, spec)
            for name, spec in tensor_specs.items()
        }

    def _poll_get(
        self,
        handle: str,
        timeout: float,
        poll_interval: float,
        prefer_chunks: bool = False,
    ) -> bytes | None:
        """Poll for a sample: chunked transfer for large payloads.

        Returns ``None`` on timeout.  Raises on hard errors.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Check for chunked marker (fast path for large samples).
            if prefer_chunks:
                try:
                    status, data = self._request("GET", self._url(f"{handle}/nchunks"))
                    if status == 200:
                        return self._chunked_get(handle, int(data))
                except HttpStoreError:
                    pass  # no marker; try single GET below

            # Single-shot GET.
            try:
                status, data = self._request("GET", self._url(handle))
                if status == 200:
                    return data
            except HttpStoreError:
                pass

            time.sleep(poll_interval)
        return None

    def _chunked_get(self, handle: str, nchunks: int) -> bytes:
        """Fetch all chunks in parallel and reassemble."""
        ex = self._get_chunk_executor()

        def _get_chunk(idx: int) -> bytes:
            status, data = self._request("GET", self._url(f"{handle}/{idx}"))
            if status != 200:
                raise HttpStoreError(f"HTTP GET chunk {handle}/{idx}: status {status}")
            return data

        chunks = list(ex.map(_get_chunk, range(nchunks)))
        # Best-effort cleanup of chunk entries after successful read.
        for i in range(nchunks):
            try:
                ex.submit(self._request, "DELETE", self._url(f"{handle}/{i}"))
            except Exception:  # noqa: BLE001
                pass
        return b"".join(chunks)

    def _get_error(self, handle: str) -> str | None:
        try:
            status, data = self._request("GET", self._url(handle + ":error"))
            if status == 200:
                return json.loads(data).get("error")
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _extract_tensor(
        key: str, name: str, obj: torch.Tensor, spec: Any
    ) -> torch.Tensor:
        """Copy one tensor out of the staging byte buffer."""
        try:
            shape = tuple(int(d) for d in spec["shape"])
            dtype = _parse_dtype(key, name, spec["dtype"])
            offset, nbytes = int(spec["offset"]), int(spec["nbytes"])
        except (KeyError, TypeError, ValueError) as e:
            raise MooncakeIntegrityError(
                f"Invalid tensor manifest for key={key}:{name}: {spec!r}"
            ) from e

        expected = torch.Size(shape).numel() * dtype.itemsize
        if nbytes != expected or offset < 0 or offset + nbytes > obj.numel():
            raise MooncakeIntegrityError(
                f"Tensor manifest mismatch for key={key}:{name}: "
                f"nbytes={nbytes} expected={expected} "
                f"offset={offset} obj_numel={obj.numel()}"
            )
        view = obj[offset : offset + nbytes].view(torch.uint8)
        return view.view(dtype).view(shape).clone()
