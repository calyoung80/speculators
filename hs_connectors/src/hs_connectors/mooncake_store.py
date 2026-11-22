"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem. This backend stores the tensor payload as a
raw packed Mooncake object instead, while the versioned transfer manifest
(tensor shapes, dtypes, and byte offsets) travels separately in the vLLM
response so the two sides can run on different nodes.
"""

from __future__ import annotations

import ctypes
import json
import logging
import threading
import time
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
import torch

from hs_connectors.device import ensure_accelerator_context

logger = logging.getLogger(__name__)


VERSION = 1

_TENSOR_ALIGN = 64
_OBJECT_NOT_FOUND = -704
_MIN_POLL_INTERVAL = 0.001


class MooncakeIntegrityError(RuntimeError):
    """A Mooncake object was present but failed an integrity check."""


class NonFiniteTensorError(MooncakeIntegrityError):
    """A producer attempted to publish a tensor containing NaN or infinity."""


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise ``NonFiniteTensorError`` if ``tensor`` holds NaN or infinity.

    ``amin``/``amax`` propagate NaN and surface either infinity, so the whole
    tensor is screened in two reductions with no full-size boolean temporary.
    Call this while the data is still on the accelerator: on the host it is an
    extra pass over the whole sample on the critical path of a producer write.
    """
    if not tensor.is_floating_point() or tensor.numel() == 0:
        return

    bounds = torch.stack((tensor.amin(), tensor.amax()))
    if bool(torch.isfinite(bounds).all()):
        return

    # Only pay for the exact counts once we already know the sample is bad.
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    raise NonFiniteTensorError(
        f"Non-finite producer tensor {name!r}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, nan_count={nan_count}, inf_count={inf_count}"
    )


def _check_store_result(operation: str, key: str, result: Any) -> None:
    """Mooncake's Python API returns negative status codes for failures."""
    if not isinstance(result, int) or result < 0:
        raise RuntimeError(
            f"Mooncake {operation} failed for key={key} with status={result}"
        )


def _align(offset: int) -> int:
    return -(-offset // _TENSOR_ALIGN) * _TENSOR_ALIGN


def packed_layout(
    tensors: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> tuple[dict[str, Any], int]:
    """Assign aligned offsets; return the tensor spec and total size"""
    specs: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, tensor_metadata in tensors.items():
        shape, dtype = tensor_metadata
        nbytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
        specs[name] = {
            "shape": shape,
            "dtype": str(dtype),
            "nbytes": nbytes,
            "offset": offset,  # byte offset in packed tensor
        }
        offset = _align(offset + nbytes)
    return specs, offset


def _parse_dtype(key: str, name: str, value: Any) -> torch.dtype:
    dtype = getattr(torch, str(value).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise MooncakeIntegrityError(f"Unknown dtype for key={key}:{name}: {value!r}")
    return dtype


def _lease_tensor(lease: Any, nbytes: int) -> torch.Tensor:
    """Wrap a staging lease as a writable CPU ``torch.uint8`` view."""
    buf = (ctypes.c_ubyte * nbytes).from_address(int(lease.ptr))
    return torch.from_numpy(np.ctypeslib.as_array(buf))


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "P2PHANDSHAKE"
    master_server_address: str = "127.0.0.1:50051"
    global_segment_size: int = 1 * 1024 * 1024 * 1024
    local_buffer_size: int = 512 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 4

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            logger.warning("Unknown MooncakeStoreConfig keys ignored: %s", unknown)
        return cls(**{k: v for k, v in d.items() if k in known})


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store.

    Each sample is one raw object containing 64-byte-aligned tensor bytes. The
    versioned transfer manifest is supplied separately and describes the layout
    of those bytes. A Mooncake put becomes visible atomically, so the object's
    presence marks the sample complete. Staging leases are sub-allocated from
    the store's registered client buffer, so ``put_from``/``get_into`` use them
    directly without a client-side bounce copy.
    """

    def __init__(self, config: MooncakeStoreConfig):
        self.config = config
        self._store = None
        self._pool = None
        self._pool_lock = threading.Lock()

    def __getstate__(self):
        # BufferPool, locks, and the C++ store are process-local;
        # DataLoader workers recreate them via lazy setup().
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_pool_lock"] = None
        state["_store"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pool = None
        self._pool_lock = threading.Lock()

    @property
    def is_setup(self):
        return self._store is not None

    def setup(self) -> MooncakeHiddenStatesStore:
        if self._store is not None:
            return self
        try:
            from mooncake.store import (  # type: ignore[import-not-found] # noqa: PLC0415
                MooncakeDistributedStore,
            )
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Mooncake is required for the Mooncake hidden-states backend. "
                "Install it with `pip install 'mooncake-transfer-engine>=0.3.12'` or "
                "`pip install 'mooncake-transfer-engine-cuda13>=0.3.12'`."
            ) from e

        try:
            import torch_npu  # noqa: F401 # noqa: PLC0415
            if torch.npu.is_available():
                try:
                    torch.npu.set_device(torch.npu.current_device())
                except Exception:
                    torch.npu.set_device(0)
                logger.info(
                    "torch_npu initialized for Mooncake ACL context, device=%s",
                    torch.npu.current_device(),
                )
        except ImportError:
            pass

        store = MooncakeDistributedStore()
        ensure_accelerator_context()
        result = store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            self.config.local_buffer_size,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )
        _check_store_result("setup", self.config.local_hostname, result)
        self._store = store
        return self

    def _require_store(self) -> Any:
        if self._store is None:
            raise RuntimeError("call setup() first")
        return self._store

    def _staging_pool(self):
        """Lazily create the BufferPool over the store's registered client buffer."""
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    from mooncake.store import (  # type: ignore[import-not-found] # noqa: PLC0415
                        BufferPool,
                    )

                    self._pool = BufferPool(
                        self._require_store(), max_bytes=self.config.local_buffer_size
                    )
        return self._pool

    def put_sample(
        self, transfer_manifest: dict[str, Any], tensors: dict[str, torch.Tensor]
    ) -> None:
        """Publish ``tensors``, which may live on the accelerator.

        The transfer manifest supplies the packed byte layout shared with the
        consumer. Device tensors are copied into a staging lease on the current
        stream, then published zero-copy via ``put_from``.
        """
        tensor_spec = transfer_manifest["tensors"]
        total_bytes = transfer_manifest["metadata"]["total_aligned_bytes"]

        expected_names = set(tensor_spec)
        actual_names = set(tensors)
        if actual_names != expected_names:
            raise MooncakeIntegrityError(
                "Producer tensor names do not match manifest: "
                f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
            )

        for name, spec in tensor_spec.items():
            tensor = tensors[name]
            expected_dtype = _parse_dtype(
                transfer_manifest["handle"], name, spec["dtype"]
            )
            if isinstance(tensor, list):
                # Chunked device tensor (payload-size fix): gathered in token
                # slices by the connector; validated by total byte size.
                if not tensor:
                    raise MooncakeIntegrityError(
                        f"Producer tensor {name!r} chunk list is empty"
                    )
                if tensor[0].dtype != expected_dtype:
                    raise MooncakeIntegrityError(
                        f"Producer tensor {name!r} dtype {tensor[0].dtype} does not "
                        f"match manifest dtype {expected_dtype}"
                    )
                actual_nbytes = sum(
                    c.numel() * c.element_size() for c in tensor
                )
            else:
                expected_shape = tuple(spec["shape"])
                if tuple(tensor.shape) != expected_shape:
                    raise MooncakeIntegrityError(
                        f"Producer tensor {name!r} shape {tuple(tensor.shape)} does not "
                        f"match manifest shape {expected_shape}"
                    )
                if tensor.dtype != expected_dtype:
                    raise MooncakeIntegrityError(
                        f"Producer tensor {name!r} dtype {tensor.dtype} does not "
                        f"match manifest dtype {expected_dtype}"
                    )
                actual_nbytes = tensor.numel() * tensor.element_size()
            if actual_nbytes != spec["nbytes"]:
                raise MooncakeIntegrityError(
                    f"Producer tensor {name!r} has {actual_nbytes} bytes, but the "
                    f"manifest reserves {spec['nbytes']}"
                )

        store = self._require_store()
        # Lease memory lives in the registered client buffer, so put_from needs
        # no register_buffer (caller memory fails with -600 on Ascend fabric).
        lease = self._staging_pool().acquire(total_bytes)
        try:
            staging = _lease_tensor(lease, total_bytes)

            for name, spec in tensor_spec.items():
                tensor = tensors[name]
                region = staging[spec["offset"] : spec["offset"] + spec["nbytes"]]
                if isinstance(tensor, list):
                    # Copy each chunk into its slice of the packed region so every
                    # DtoH stays small (a single ~2.4 GB pinned copy hangs AICPU).
                    off = 0
                    step = 8 * 1024 * 1024  # 8M elements, node82-proven DtoH slice
                    for chunk in tensor:
                        n = chunk.numel() * chunk.element_size()
                        dst = region[off : off + n].view(chunk.dtype).reshape(-1)
                        src_flat = chunk.reshape(-1)
                        for s_ in range(0, src_flat.numel(), step):
                            e = min(s_ + step, src_flat.numel())
                            dst[s_:e].copy_(src_flat[s_:e])
                        off += n
                    if off != spec["nbytes"]:
                        raise MooncakeIntegrityError(
                            f"Producer tensor {name!r} chunks covered {off} bytes, "
                            f"but the manifest reserves {spec['nbytes']}"
                        )
                else:
                    region.view(tensor.dtype).view(tensor.shape).copy_(tensor)

            handle = transfer_manifest["handle"]
            result = store.put_from(handle, lease.ptr, total_bytes)
            _check_store_result("put_from", handle, result)
        finally:
            lease.release()

    def put_error(self, transfer_manifest: dict[str, Any], error: str) -> None:
        """Publish a small terminal marker so consumers fail fast and can retry."""
        store = self._require_store()
        error_dict = {
            "error": error[:4096],
        }
        error_handle = transfer_manifest["handle"] + ":error"

        _check_store_result(
            "put",
            error_handle,
            store.put(error_handle, json.dumps(error_dict).encode("utf-8")),
        )

    def delete_sample(self, key: str) -> None:
        """Remove the data and error objects for a sample.

        A missing object is not an error.
        """
        error_handle = key + ":error"

        for k in (key, error_handle):
            result = self._require_store().remove(k, force=True)
            if result != _OBJECT_NOT_FOUND:
                _check_store_result("remove", k, result)

    def get_sample(
        self,
        transfer_manifest: dict[str, Any],
        timeout: float = 120.0,
        poll_interval: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Wait for and decode a sample described by ``transfer_manifest``.

        Mooncake contains only the packed tensor bytes. The manifest returned
        by vLLM supplies the layout needed to reconstruct each tensor.
        """

        if (manifest_version := transfer_manifest.get("version", 0)) != VERSION:
            raise MooncakeIntegrityError(
                f"Mooncake manifest version {manifest_version} didn't match "
                f"local version {VERSION}"
                f"Transfer manifest: {transfer_manifest}\n"
            )

        self._wait_for(transfer_manifest, timeout, poll_interval)

        handle = transfer_manifest["handle"]
        tensor_specs = transfer_manifest["tensors"]
        expected_payload_size = transfer_manifest["metadata"]["total_aligned_bytes"]
        staging_tensor, lease = self._read_from_store(handle, expected_payload_size)

        try:
            return {
                name: self._extract_tensor(handle, name, staging_tensor, spec)
                for name, spec in tensor_specs.items()
            }
        finally:
            lease.release()

    def _read_from_store(self, key, expected_payload_size: int = 0):
        """Read one raw Mooncake object into a staging lease via ``get_into``.

        Returns ``(tensor, lease)``; the view is invalid after ``lease.release()``.
        """
        store = self._require_store()
        size = store.get_size(key)
        if not isinstance(size, int) or size < 0:
            raise MooncakeIntegrityError(
                f"Mooncake value unavailable for key={key} (status={size})"
            )
        if expected_payload_size > 0 and size != expected_payload_size:
            raise MooncakeIntegrityError(
                f"Mooncake payload size {size} doesn't match expected payload "
                f"size {expected_payload_size}"
            )

        lease = self._staging_pool().acquire(size)
        try:
            received = store.get_into(key, lease.ptr, size)
            if received != size:
                raise MooncakeIntegrityError(
                    f"Mooncake get_into for key={key} returned {received}, "
                    f"expected {size} bytes"
                )
            return _lease_tensor(lease, size), lease
        except BaseException:
            lease.release()
            raise

    def _process_error(self, error_handle, transfer_manifest: dict[str, Any]):
        """Read a producer error marker and raise it as an integrity error."""
        obj, lease = self._read_from_store(error_handle)
        try:
            raw = obj.numpy().tobytes()
        finally:
            lease.release()

        try:
            error_dict = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MooncakeIntegrityError(
                f"Corrupt Mooncake json for key={error_handle}: {e}"
            ) from e

        if not isinstance(error_dict, dict):
            raise MooncakeIntegrityError(
                f"Invalid Mooncake error marker for key={error_handle}: "
                f"Received: {error_dict}\n"
                f"Transfer manifest: {transfer_manifest}\n"
            )
        raise MooncakeIntegrityError(
            f"Mooncake producer error for key={error_handle}: "
            f"{error_dict.get('error', 'unknown producer error')}\n"
            f"Transfer manifest: {transfer_manifest}\n"
        )

    @staticmethod
    def _extract_tensor(
        key: str, name: str, obj: torch.Tensor, spec: Any
    ) -> torch.Tensor:
        """Copy one tensor out of the staging lease before it is released."""
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
                f"Mooncake tensor out of bounds for key={key}:{name}: "
                f"offset={offset}, nbytes={nbytes}, expected_nbytes={expected}, "
                f"object={obj.numel()} bytes"
            )
        return obj[offset : offset + nbytes].view(dtype).view(shape).clone()

    def _wait_for(
        self,
        transfer_manifest: dict[str, Any],
        timeout: float,
        poll_interval: float,
    ) -> None:
        """Wait for the data object or its terminal error marker.

        vLLM answers the HTTP request before the connector's put completes, so
        the sample usually lands a few milliseconds after the consumer asks.
        The error marker is checked first so producer failures take precedence
        if both keys happen to be visible.
        """
        store = self._require_store()

        def _check_key(key: str) -> bool:
            exists = store.is_exist(key)
            if exists == 1:
                return True
            if exists != 0:
                raise RuntimeError(
                    f"Mooncake is_exist failed for key={key} with status={exists}"
                )
            return False

        handle = transfer_manifest["handle"]
        error_handle = handle + ":error"

        deadline = time.monotonic() + timeout
        delay = min(_MIN_POLL_INTERVAL, poll_interval)
        while True:
            # Prefer the terminal error marker over the data object.
            if _check_key(error_handle):
                self._process_error(error_handle, transfer_manifest)
            if _check_key(handle):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {handle}")
            time.sleep(delay)
            delay = min(delay * 2, poll_interval)
