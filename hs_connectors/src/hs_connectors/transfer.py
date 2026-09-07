"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import dataclasses
import fcntl
import os
import shutil
import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file

from hs_connectors.mooncake_store import MooncakeHiddenStatesStore, MooncakeStoreConfig
from hs_connectors.mooncake_te_store import MooncakeTEConfig, MooncakeTEStore

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


def wait_for_lock(lock_path: str, timeout: float = 10.0, poll_interval: float = 0.1):
    fd = os.open(lock_path, os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

    return None


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return _load_hs_file(path)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return _load_hs_file(Path(handle))

    def cache(self, handle: str, file_idx: int) -> None:
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        target = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        shutil.move(handle, target)

    def delete(self, handle: str) -> None:
        Path(handle).unlink()


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }


# ---------------------------------------------------------------------------
# Mooncake-based backend (distributed store)
# ---------------------------------------------------------------------------


class MooncakeTransfer(HiddenStatesTransfer):
    """Mooncake distributed store based hidden-states transfer."""

    def __init__(self, store: MooncakeHiddenStatesStore):
        self.store = store

    def setup(self) -> None:
        if not self.store.is_setup:
            self.store.setup()

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:  # noqa: ARG002
        return None

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return self.store.get_sample(handle)

    def delete(self, handle: str) -> None:
        self.store.delete_sample(handle)


@HiddenStatesBackend.register("mooncake")
class MooncakeBackend(HiddenStatesBackend):
    """Mooncake distributed store backend (no shared filesystem required)."""

    @staticmethod
    def _add_mooncake_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mooncake-master",
            type=str,
            default="127.0.0.1:50051",
            help="Mooncake master server address. Used with backend=mooncake.",
        )
        parser.add_argument(
            "--mooncake-metadata-server",
            type=str,
            default="P2PHANDSHAKE",
            help=(
                "Mooncake metadata server (or P2PHANDSHAKE). "
                "Used with backend=mooncake."
            ),
        )
        parser.add_argument(
            "--mooncake-protocol",
            choices=["tcp", "rdma"],
            default="tcp",
            help="Mooncake transport protocol. Used with backend=mooncake.",
        )
        parser.add_argument(
            "--mooncake-device",
            type=str,
            default="",
            help="Mooncake RDMA device name (e.g. mlx5_0, auto-discovery). Used with backend=mooncake.",
        )

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,  # noqa: ARG004
    ) -> MooncakeTransfer:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        store = MooncakeHiddenStatesStore(
            MooncakeStoreConfig(
                local_hostname=local_hostname,
                metadata_server=args.mooncake_metadata_server,
                master_server_address=args.mooncake_master,
                protocol=args.mooncake_protocol,
                device_name=getattr(args, "mooncake_device", "") or "",
            )
        )
        return MooncakeTransfer(store)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        mooncake_cfg = MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=args.mooncake_metadata_server,
            master_server_address=args.mooncake_master,
            protocol=args.mooncake_protocol,
            device_name=getattr(args, "mooncake_device", "") or "",
        )

        return {
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": (
                "hs_connectors.mooncake_hidden_states_connector"
            ),
            "kv_connector_extra_config": {
                "mooncake": dataclasses.asdict(mooncake_cfg),
            },
        }


# ---------------------------------------------------------------------------
# Mooncake TransferEngine backend (NPU direct, no DtoH copy)
# ---------------------------------------------------------------------------


class MooncakeTETransfer(HiddenStatesTransfer):
    """TransferEngine NPU direct hidden-states transfer (consumer side)."""

    def __init__(self, store: MooncakeTEStore):
        self.store = store

    def setup(self) -> None:
        if not self.store.is_setup:
            self.store.setup()

    def _normalize_handle(self, handle: str) -> str:
        import os

        if "/" in handle or handle.endswith(".safetensors"):
            handle = os.path.basename(handle)
            if handle.endswith(".safetensors"):
                handle = handle[: -len(".safetensors")]
        return handle

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:  # noqa: ARG002
        return None

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return self.store.get_sample(self._normalize_handle(handle))

    def delete(self, handle: str) -> None:
        self.store.delete_sample(self._normalize_handle(handle))


@HiddenStatesBackend.register("mooncake-te")
class MooncakeTEBackend(HiddenStatesBackend):
    """TransferEngine NPU direct backend (HCCS/RoCE, no DtoH copy)."""

    @staticmethod
    def _add_te_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mooncake-te-zmq-port",
            type=int,
            default=9999,
            help="ZMQ port for TransferEngine control plane. Used with backend=mooncake-te.",
        )
        parser.add_argument(
            "--mooncake-te-producer-ip",
            type=str,
            default="",
            help=(
                "Producer (vLLM) IP for TransferEngine. If empty, extracted "
                "from --vllm-endpoint. Used with backend=mooncake-te."
            ),
        )

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        MooncakeTEBackend._add_te_args(parser)

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        MooncakeTEBackend._add_te_args(parser)

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,  # noqa: ARG004
    ) -> MooncakeTETransfer:
        producer_ip = getattr(args, "mooncake_te_producer_ip", "") or ""
        if not producer_ip:
            from urllib.parse import urlparse

            endpoint = getattr(args, "vllm_endpoint", "")
            if endpoint:
                producer_ip = urlparse(endpoint).hostname or ""

        store = MooncakeTEStore(
            MooncakeTEConfig(
                zmq_port=getattr(args, "mooncake_te_zmq_port", 9999),
                producer_ip=producer_ip,
            )
        )
        return MooncakeTETransfer(store)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        te_cfg = MooncakeTEConfig(
            local_hostname=local_hostname,
            protocol="ascend",
            zmq_port=getattr(args, "mooncake_te_zmq_port", 9999),
            producer_ip="",  # producer: empty = ZMQ REP bind
        )

        return {
            "kv_connector": "MooncakeTEHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": "hs_connectors.mooncake_te_connector",
            "kv_connector_extra_config": {
                "mooncake_te": dataclasses.asdict(te_cfg),
            },
        }
