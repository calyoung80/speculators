"""Mooncake TransferEngine store for NPU direct hidden-states transfer.

Architecture:
- Producer (vLLM): copies hidden states to persistent send buffer, writes TE metadata
- Consumer (training): polls files, reads metadata, calls batch_transfer_sync_read
- No ZMQ (avoids deadlock: put_sample blocked vLLM response)
- No per-request register_memory / unregister_memory
- Persistent send buffer prevents NaN from KV cache block reuse after request completion
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def _log_timing(message: str, *args: object) -> None:
    """Keep T0 timing visible when vLLM filters connector INFO logs."""
    if os.environ.get("TE_TIMING_LOG", "0") == "1":
        logger.warning(message, *args)
    else:
        logger.info(message, *args)

MAX_TOKEN_NUMEL = 512 * 1024  # 512K elements * 8 bytes = 4MB
MAX_HS_NUMEL = (
    32 * 1024 * 4 * 2048
)  # 32K tokens * 4 layers * 2048 hidden = 256M elements * 2 bytes = 512MB
TE_COPY_TOKEN_CHUNK_SIZE = max(
    1, int(os.environ.get("TE_COPY_TOKEN_CHUNK_SIZE", "128"))
)
TE_META_DIR = os.environ.get("TE_META_DIR", "/tmp/te_meta")

# Global TE engine — shared between distributed.py pre-init and MooncakeTEStore
_global_te_engine = None


def pre_init_te_engine():
    """Initialize TransferEngine (ADXL) BEFORE dist.init_process_group.

    Called from distributed.py on rank 0 when TE_PRE_INIT=1 env var is set.
    This ensures ADXL is initialized before HCCL, preventing the HCCL conflict
    where ADXL init disrupts the existing DDP HCCL connection.
    """
    global _global_te_engine
    if _global_te_engine is not None:
        return _global_te_engine

    import torch_npu  # noqa: F401

    if torch.npu.is_available():
        torch.npu.set_device(torch.npu.current_device())

    hostname = socket.gethostbyname(socket.gethostname())
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    protocol = os.environ.get("MOONCAKE_TE_PROTOCOL", "tcp")
    ret = engine.initialize(hostname, "P2PHANDSHAKE", protocol, "")
    if ret != 0:
        raise RuntimeError(f"pre_init_te_engine: initialize failed: ret={ret}")
    _global_te_engine = engine
    logger.info("pre_init_te_engine: ADXL initialized on %s (before HCCL)", hostname)
    return engine


@dataclass
class MooncakeTEConfig:
    local_hostname: str = ""
    protocol: str = os.environ.get("MOONCAKE_TE_PROTOCOL", "tcp")
    device_name: str = ""
    zmq_port: int = 9999
    producer_ip: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeTEConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            logger.warning("Unknown MooncakeTEConfig keys ignored: %s", unknown)
        return cls(**{k: v for k, v in d.items() if k in known})


class MooncakeTEStore:
    """TransferEngine store — register once, transfer with offsets.

    Producer (vLLM): gathers hidden states into a persistent registered buffer;
    token_ids are inlined in metadata.
    Consumer (training): pre-allocates + registers receive buffers.
    Metadata exchanged via files (no ZMQ, no deadlock).
    """

    def __init__(self, config: MooncakeTEConfig):
        self.config = config
        self._engine = None
        self._is_producer = config.producer_ip == ""
        self._is_setup = False
        self._local_hostname = ""
        self._token_buffer = None
        self._token_buffer_ptr = 0
        self._recv_hs_buffer = None
        self._recv_hs_ptr = 0
        self._send_hs_buffer = None
        self._send_hs_ptr = 0
        self._send_offset = 0
        self._generation_by_key: dict[str, int] = {}
        self._producer_epoch = f"{time.time_ns():x}-{os.getpid():x}"

    @property
    def is_setup(self) -> bool:
        return self._is_setup

    def setup(self) -> MooncakeTEStore:
        if self._is_setup:
            return self

        import torch_npu  # noqa: F401, PLC0415

        if torch.npu.is_available():
            try:
                torch.npu.set_device(torch.npu.current_device())
            except Exception:
                torch.npu.set_device(0)

        self._local_hostname = self.config.local_hostname or socket.gethostbyname(
            socket.gethostname()
        )

        # Reuse global engine if pre-initialized (before HCCL), else create new
        global _global_te_engine
        if _global_te_engine is not None:
            engine = _global_te_engine
            logger.info("Reusing pre-initialized TE engine")
        else:
            from mooncake.engine import TransferEngine  # noqa: PLC0415

            engine = TransferEngine()
            ret = engine.initialize(
                self._local_hostname,
                "P2PHANDSHAKE",
                self.config.protocol,
                self.config.device_name,
            )
            if ret != 0:
                raise RuntimeError(f"TransferEngine.initialize failed: ret={ret}")
        self._engine = engine

        self._token_buffer = torch.empty(
            MAX_TOKEN_NUMEL, dtype=torch.long, pin_memory=True
        )
        self._token_buffer_ptr = self._token_buffer.data_ptr()
        ret = engine.register_memory(self._token_buffer_ptr, self._token_buffer.nbytes)
        if ret != 0:
            raise RuntimeError(f"register_memory failed for token_buffer: ret={ret}")

        if self._is_producer:
            # TEMPORARILY DISABLED: _send_hs_buffer causes HcclCommPrepare fail
            # self._send_hs_buffer = torch.empty(
            #     MAX_HS_NUMEL,
            #     dtype=torch.bfloat16, device=device,
            # )
            # self._send_hs_ptr = self._send_hs_buffer.data_ptr()
            # ret = engine.register_memory(self._send_hs_ptr, self._send_hs_buffer.nbytes)
            # if ret != 0:
            #     raise RuntimeError(f"register_memory failed for send_hs: ret={ret}")
            pass

        if not self._is_producer:
            device = f"npu:{torch.npu.current_device()}"
            self._recv_hs_buffer = torch.empty(
                MAX_HS_NUMEL,
                dtype=torch.bfloat16,
                device=device,
            )
            self._recv_hs_ptr = self._recv_hs_buffer.data_ptr()
            ret = engine.register_memory(self._recv_hs_ptr, self._recv_hs_buffer.nbytes)
            if ret != 0:
                raise RuntimeError(f"register_memory failed for recv_hs: ret={ret}")

        os.makedirs(TE_META_DIR, exist_ok=True)
        if self._is_producer:
            self._cleanup_stale_samples()

        logger.info(
            "TE buffers registered (token=%d elems @0x%x)",
            self._token_buffer.numel(),
            self._token_buffer_ptr,
        )
        if self._is_producer and self._send_hs_buffer is not None:
            logger.info(
                "TE send_hs buffer: %d elems @0x%x",
                self._send_hs_buffer.numel(),
                self._send_hs_ptr,
            )
        logger.info(
            "TE store setup complete (producer=%s, hostname=%s)",
            self._is_producer,
            self._local_hostname,
        )

        self._is_setup = True
        return self

    def _close_engine(self):
        """Destroy ADXL engine to free NPU HCCL resources."""
        if self._engine is not None:
            del self._engine
            self._engine = None
            import gc

            gc.collect()
            logger.info("TE engine closed (ADXL destroyed)")

    def _ensure_engine(self):
        """Re-create engine if it was closed."""
        if self._engine is None:
            from mooncake.engine import TransferEngine

            engine = TransferEngine()
            ret = engine.initialize(
                self._local_hostname,
                "P2PHANDSHAKE",
                self.config.protocol,
                self.config.device_name,
            )
            if ret != 0:
                raise RuntimeError(f"TransferEngine re-init failed: ret={ret}")
            engine.register_memory(self._token_buffer_ptr, self._token_buffer.nbytes)
            if self._recv_hs_ptr:
                engine.register_memory(self._recv_hs_ptr, self._recv_hs_buffer.nbytes)
            if self._send_hs_ptr:
                engine.register_memory(self._send_hs_ptr, self._send_hs_buffer.nbytes)
            self._engine = engine
            logger.info("TE engine re-initialized")

    def register_kv_cache(self, kv_cache: torch.Tensor) -> None:
        ptr = kv_cache.data_ptr()
        size = kv_cache.nbytes
        ret = self._engine.register_memory(ptr, size)
        if ret != 0:
            raise RuntimeError(f"register_memory failed for kv_cache: ret={ret}")
        logger.info("KV cache registered: %d bytes @0x%x", size, ptr)

    def reset_send_buffer(self) -> None:
        self._send_offset = 0

    def has_send_capacity(self, kv_cache: torch.Tensor, num_tokens: int) -> bool:
        """Check whether another lease fits without overwriting live data."""
        if self._send_hs_buffer is None:
            return True
        required_numel = num_tokens * math.prod(kv_cache.shape[2:])
        return self._send_offset + required_numel <= self._send_hs_buffer.numel()

    def copy_to_send_buffer(
        self, kv_cache: torch.Tensor, slot_mapping: torch.Tensor, num_tokens: int
    ) -> tuple[int, int, list[int]]:
        # Lazy allocation: allocate _send_hs_buffer on first call,
        # AFTER setup() and ADXL engine init (avoids HcclCommPrepare conflict).
        if self._send_hs_buffer is None:
            import torch_npu  # noqa: F401

            device = f"npu:{torch.npu.current_device()}"
            self._send_hs_buffer = torch.empty(
                MAX_HS_NUMEL,
                dtype=torch.bfloat16,
                device=device,
            )
            self._send_hs_ptr = self._send_hs_buffer.data_ptr()
            ret = self._engine.register_memory(
                self._send_hs_ptr, self._send_hs_buffer.nbytes
            )
            if ret != 0:
                raise RuntimeError(
                    f"register_memory failed for send_hs (lazy): ret={ret}"
                )
            logger.info(
                "TE send_hs buffer lazily allocated: %d elems @0x%x",
                self._send_hs_buffer.numel(),
                self._send_hs_ptr,
            )

        block_size = kv_cache.shape[1]
        slot_mapping = slot_mapping.reshape(-1)[:num_tokens]
        per_token_numel = math.prod(kv_cache.shape[2:])
        numel = slot_mapping.numel() * per_token_numel
        if self._send_offset + numel > self._send_hs_buffer.numel():
            raise RuntimeError(
                f"send_hs_buffer overflow: need {self._send_offset + numel}, "
                f"have {self._send_hs_buffer.numel()}"
            )

        # Advanced indexing of the complete request materializes a temporary
        # tensor proportional to the whole prompt. With Qwen3.8-27B and a
        # 5K-token request that temporary was ~2.4 GiB and killed the Producer.
        # Gather bounded token chunks and write them directly into the already
        # registered send buffer instead.
        chunk_tokens = TE_COPY_TOKEN_CHUNK_SIZE
        copy_start = time.perf_counter()
        write_offset = self._send_offset
        for start in range(0, slot_mapping.numel(), chunk_tokens):
            stop = min(start + chunk_tokens, slot_mapping.numel())
            chunk_slots = slot_mapping[start:stop]
            extracted = kv_cache[
                chunk_slots // block_size, chunk_slots % block_size
            ].contiguous()
            chunk_numel = extracted.numel()
            self._send_hs_buffer[write_offset : write_offset + chunk_numel].copy_(
                extracted.view(-1)
            )
            write_offset += chunk_numel
            del extracted
        torch.npu.current_stream().synchronize()
        element_size = kv_cache.element_size()
        copy_ms = (time.perf_counter() - copy_start) * 1000
        ptr = self._send_hs_ptr + self._send_offset * element_size
        size = numel * element_size
        shape = [slot_mapping.numel(), *kv_cache.shape[2:]]
        self._send_offset += numel
        _log_timing(
            "copy_to_send_buffer: %d tokens, %d bytes, offset=%d, ptr=0x%x, "
            "chunk_tokens=%d, copy_ms=%.2f",
            num_tokens,
            size,
            self._send_offset - numel,
            ptr,
            chunk_tokens,
            copy_ms,
        )
        return ptr, size, shape

    def _meta_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(TE_META_DIR, f"{safe_key}.json")

    def _ack_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(TE_META_DIR, f"{safe_key}.ack")

    @staticmethod
    def _write_json_atomically(path: str, payload: dict) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)

    @staticmethod
    def _read_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _cleanup_stale_samples(self) -> None:
        """Remove abandoned files only when an operator explicitly enables TTL."""
        ttl = float(os.environ.get("TE_ACK_STALE_TTL", "0"))
        if ttl <= 0:
            return
        cutoff = time.time() - ttl
        for entry in os.scandir(TE_META_DIR):
            if not entry.name.endswith(".json"):
                continue
            ack_path = f"{entry.path[:-5]}.ack"
            try:
                if (
                    os.path.exists(ack_path)
                    and max(entry.stat().st_mtime, os.stat(ack_path).st_mtime) < cutoff
                ):
                    os.remove(entry.path)
                    os.remove(ack_path)
                    logger.warning("Removed stale TE lease: %s", entry.path[:-5])
            except FileNotFoundError:
                pass
        for entry in os.scandir(TE_META_DIR):
            if not entry.name.endswith(".ack"):
                continue
            metadata_path = f"{entry.path[:-4]}.json"
            try:
                if not os.path.exists(metadata_path) and entry.stat().st_mtime < cutoff:
                    os.remove(entry.path)
                    logger.warning("Removed stale orphan TE ACK: %s", entry.path)
            except FileNotFoundError:
                pass

    def put_sample(self, key: str, tensor_specs: dict[str, dict]) -> str:
        """Write TE metadata to file. Consumer polls and reads it.

        No ZMQ, no blocking, no deadlock.
        Only device-memory tensors go through batch_transfer_sync_read.
        Host-memory tensors (token_ids) are inlined into the metadata file.
        """
        if not self._is_setup:
            raise RuntimeError("call setup() first")

        publish_start = time.perf_counter()
        generation_index = self._generation_by_key.get(key, 0) + 1
        self._generation_by_key[key] = generation_index
        generation = f"{self._producer_epoch}-{generation_index}"
        metadata = {
            "version": 1,
            "request_key": key,
            "generation": generation,
            "producer_ip": self._local_hostname,
            "rpc_port": self._engine.get_rpc_port(),
            "tensors": tensor_specs,
        }

        path = self._meta_path(key)
        self._write_json_atomically(path, metadata)
        _log_timing(
            "put_sample key=%s generation=%s, metadata written to %s, publish_ms=%.2f",
            key,
            generation,
            path,
            (time.perf_counter() - publish_start) * 1000,
        )
        return generation

    def copy_token_ids(self, token_ids: torch.Tensor) -> tuple[int, int]:
        t = token_ids.detach().contiguous().view(-1).to(torch.long).cpu()
        if t.numel() > self._token_buffer.numel():
            raise RuntimeError(
                f"token_ids too large: {t.numel()} > {self._token_buffer.numel()}"
            )
        self._token_buffer[: t.numel()].copy_(t)
        return self._token_buffer_ptr, t.nbytes

    def get_sample(
        self, key: str, timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
        """Poll for metadata file, then TE transfer device-memory tensors only.

        Host-memory tensors (token_ids with 'data' field) are read from the
        metadata file directly — the ADXL engine does not support host-to-host
        transfers.
        """
        if not self._is_setup:
            raise RuntimeError("call setup() first")

        path = self._meta_path(key)
        t0 = time.perf_counter()
        while not os.path.exists(path):
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"Timeout waiting for metadata file: {path}")
            time.sleep(poll_interval)

        metadata_wait = time.perf_counter() - t0
        logger.info(
            "get_sample_metadata key=%s wait_ms=%.2f",
            key,
            metadata_wait * 1000,
        )

        metadata = self._read_json(path)
        if metadata.get("version") != 1 or metadata.get("request_key") != key:
            raise RuntimeError(f"Invalid TE metadata for key={key}")
        generation = metadata.get("generation")
        if not isinstance(generation, str) or not generation:
            raise RuntimeError(f"Invalid TE metadata generation for key={key}")

        session_id = f"{metadata['producer_ip']}:{metadata['rpc_port']}"

        # Separate device-memory tensors (TE transfer) from inline data
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        meta_info: dict[str, dict] = {}
        result: dict[str, torch.Tensor] = {}

        for name, info in metadata["tensors"].items():
            meta_info[name] = info

            # Inline data (e.g. token_ids) — read directly, no TE transfer
            if "data" in info:
                dtype_str = info["dtype"].replace("torch.", "")
                dtype = getattr(torch, dtype_str)
                shape = info["shape"]
                flat = info["data"]
                tensor = torch.tensor(flat, dtype=dtype).view(shape)
                result[name] = tensor
                continue

            if "blocks" in info:
                per_token_bytes = info["per_token_bytes"]
                offset = 0
                for block in info["blocks"]:
                    src_list.append(block["ptr"])
                    length_list.append(block["size"])
                    if name == "hidden_states":
                        dst_list.append(self._recv_hs_ptr + offset)
                    else:
                        raise ValueError(f"blocks not supported for {name}")
                    offset += block["size"]
            else:
                src_list.append(info["ptr"])
                length_list.append(info["size"])
                if name == "hidden_states":
                    dst_list.append(self._recv_hs_ptr)
                else:
                    raise ValueError(f"Unknown tensor name: {name}")

        # TE transfer for device-memory tensors only
        # API: batch_transfer_sync_read(session_id, local_dst_list, remote_src_list, length_list)
        # src_list = remote (producer) addresses, dst_list = local (consumer) addresses
        if src_list:
            # Each rank does its own TE transfer independently.
            # No dist.broadcast — DDP gives different samples (different shapes)
            # to each rank, so broadcasting rank 0's data would cause
            # HcclBroadcast EI0005 (parameter count mismatch) → EI0006.
            torch.npu.set_device(torch.npu.current_device())
            self._ensure_engine()

            skip_transfer = os.environ.get("SKIP_TE_TRANSFER", "0") == "1"

            if skip_transfer:
                logger.info("SKIP_TE_TRANSFER=1, skipping batch_transfer_sync_read")
                ret = 0
                elapsed = 0
                total_bytes = sum(length_list)
            else:
                t1 = time.perf_counter()
                ret = self._engine.batch_transfer_sync_read(
                    session_id, dst_list, src_list, length_list
                )
                torch.npu.synchronize()
                elapsed = time.perf_counter() - t1

                total_bytes = sum(length_list)
                throughput = total_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                logger.info(
                    "get_sample key=%s, ret=%d, %.2fms, %.1f MB/s, %d bytes",
                    key,
                    ret,
                    elapsed * 1000,
                    throughput,
                    total_bytes,
                )

            if ret < 0:
                raise RuntimeError(f"batch_transfer_sync_read failed: ret={ret}")

            for name, info in meta_info.items():
                if "data" in info:
                    continue
                dtype_str = info["dtype"].replace("torch.", "")
                dtype = getattr(torch, dtype_str)
                shape = info["shape"]
                expected_numel = 1
                for dim in shape:
                    expected_numel *= dim
                if name == "hidden_states":
                    tensor = self._recv_hs_buffer[:expected_numel].view(shape).clone()
                else:
                    tensor = self._token_buffer[:expected_numel].view(shape).clone()
                result[name] = tensor

        # Publish a durable, versioned ACK only after transfer and cloning finish.
        # Producer owns ACK consumption and source-buffer release.
        self._write_json_atomically(
            self._ack_path(key),
            {
                "version": 1,
                "request_key": key,
                "generation": generation,
                "consumer_id": f"{self._local_hostname}:{os.getpid()}",
            },
        )

        return result

    def wait_for_ack(self, key: str, generation: str, timeout: float) -> float:
        """Wait for and validate the Consumer ACK for one published generation."""
        ack_path = self._ack_path(key)
        started = time.perf_counter()
        while not os.path.exists(ack_path):
            if time.perf_counter() - started > timeout:
                raise TimeoutError(
                    f"ACK timeout for key={key} generation={generation} after {timeout}s"
                )
            time.sleep(0.01)

        ack = self._read_json(ack_path)
        if (
            ack.get("version") != 1
            or ack.get("request_key") != key
            or ack.get("generation") != generation
        ):
            raise RuntimeError(
                f"ACK does not match key={key} generation={generation}: {ack}"
            )
        return time.perf_counter() - started

    def release_sample(self, key: str, generation: str) -> None:
        """Consume a matching ACK and release Producer-owned control-plane state."""
        metadata_path = self._meta_path(key)
        metadata = self._read_json(metadata_path)
        if (
            metadata.get("version") != 1
            or metadata.get("request_key") != key
            or metadata.get("generation") != generation
        ):
            raise RuntimeError(
                f"Metadata does not match key={key} generation={generation}: {metadata}"
            )
        ack_path = self._ack_path(key)
        ack = self._read_json(ack_path)
        if ack.get("request_key") != key or ack.get("generation") != generation:
            raise RuntimeError(
                f"ACK does not match key={key} generation={generation}: {ack}"
            )
        os.remove(metadata_path)
        os.remove(ack_path)

    def delete_sample(self, key: str) -> None:
        """Consumer-side lifecycle hook; Producer owns metadata and ACK removal."""

    def close(self) -> None:
        pass
