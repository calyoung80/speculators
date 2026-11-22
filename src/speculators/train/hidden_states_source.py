"""Hidden-state acquisition for training data."""

import asyncio
import contextlib
import gc
import logging
import uuid
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol, cast

import openai
import torch
from hs_connectors.device import ensure_thread_context

from hs_connectors import HiddenStatesTransfer
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    generate_hidden_states_async,
    has_multimodal_content,
)
from speculators.train.data_types import DatasetRecord, GenerationReceipt
from speculators.train.recovery import GenerationRecoveryGuard, SampleUnavailable

logger = logging.getLogger("speculators")

HiddenStateSample = dict[str, torch.Tensor]
HiddenStateResult = HiddenStateSample | SampleUnavailable
OfflineMissingPolicy = Literal["skip", "warn", "raise"]


class HiddenStatesSource(Protocol):
    """Source of hidden states for one training record."""

    def get_one(self, record: DatasetRecord) -> HiddenStateResult:
        """Return hidden states or a structured unavailable result."""

    def get_batch(self, records: Sequence[DatasetRecord]) -> list[HiddenStateResult]:
        """Return one ordered result for each input record."""

    def close(self) -> None:
        """Release source resources."""


def build_client_item(record: DatasetRecord) -> ClientItem:
    """Build a vLLM request payload from a dataset record.

    Text-only records use their preprocessed token IDs. Multimodal records
    also forward their messages so vLLM can resolve image, video, or audio
    inputs through the Chat Completions API.
    """
    out_dict: dict[str, Any] = {"input_ids": record.input_ids.tolist()}

    if record.messages is not None and has_multimodal_content(record.messages):
        out_dict["messages"] = record.messages

    return cast("ClientItem", out_dict)


class OfflineHiddenStatesSource:
    """Load hidden states that were already written to the transfer backend."""

    def __init__(
        self,
        *,
        transfer: HiddenStatesTransfer,
        on_missing: OfflineMissingPolicy,
    ):
        self.transfer = transfer
        self.on_missing = on_missing

    def get_one(self, record: DatasetRecord) -> HiddenStateResult:
        """Load one cached sample by its storage index."""
        cached = self.transfer.get_cached(record.storage_index)
        if cached is not None:
            return cached

        if self.on_missing == "skip":
            return SampleUnavailable()
        if self.on_missing == "warn":
            warnings.warn(
                "Failed to load hidden states for sample "
                f"{record.storage_index}. Skipping...",
                stacklevel=1,
            )
            return SampleUnavailable(
                reason=f"Hidden states unavailable for sample {record.storage_index}"
            )
        raise RuntimeError(
            f"Failed to load hidden states for sample {record.storage_index}."
        )

    def get_batch(self, records: Sequence[DatasetRecord]) -> list[HiddenStateResult]:
        """Load cached samples in input order."""
        return [self.get_one(record) for record in records]

    def close(self) -> None:
        """Offline transfers do not currently own closeable resources."""


class VllmHiddenStatesSource:
    """Generate and retrieve hidden states from a vLLM endpoint.

    This source always generates a sample. Cache-miss handling belongs to
    :class:`OfflineHiddenStatesSource`; this object owns the vLLM round trip,
    validation, cleanup, and generation recovery.
    """

    def __init__(
        self,
        *,
        transfer: HiddenStatesTransfer,
        model: str,
        vllm_endpoint: str = "http://localhost:8000/v1",
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        generation_validation_retries: int = 2,
        max_consecutive_generation_failures: int = 20,
        max_transfer_workers: int = 4,
    ):
        if max_transfer_workers <= 0:
            raise ValueError("max_transfer_workers must be > 0")

        self.transfer = transfer
        self.vllm_endpoint = vllm_endpoint
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.max_transfer_workers = max_transfer_workers
        self._transfer_executor: ThreadPoolExecutor | None = None
        self._is_setup = False
        # r8 (2026-11-06): bound Mooncake engine host-memory retention.
        # Each get_into on a producer-originated object can leave one
        # payload-sized anonymous mapping that the engine only returns
        # after the calling threads exit and a fresh event loop is created
        # and closed (probes E/O1 on s96). Recycle the transfer executor
        # every executor_recycle_interval loaded samples.
        self._loads_since_recycle = 0
        self.executor_recycle_interval = 4
        # Streaming-path client (mooncake-integration): the streaming loader
        # drives generate()/load_generated() on its own long-lived event loop
        # and fetch threads, so it needs a persistent client here. The torch
        # path (get_batch) keeps its r7 per-batch throwaway clients and never
        # touches this attribute.
        self._client: openai.AsyncOpenAI | None = None
        # NPU device index captured at setup time; pool threads bind this
        # context via ensure_thread_context (CANN contexts are thread-local
        # and mooncake's adxl engine runs fabric transfers on the calling
        # thread; fresh threads otherwise default to device 0).
        self._device_index: int | None = None
        self.generation_recovery = GenerationRecoveryGuard(
            retries=generation_validation_retries,
            max_consecutive_failures=max_consecutive_generation_failures,
        )

    def __getstate__(self):
        # AsyncOpenAI client and ThreadPoolExecutor are process-local;
        # DataLoader workers recreate them via lazy _setup().
        state = self.__dict__.copy()
        state["_client"] = None
        state["_transfer_executor"] = None
        state["_is_setup"] = False
        state["_loads_since_recycle"] = 0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = None
        self._transfer_executor = None
        self._is_setup = False
        self._loads_since_recycle = 0

    def _setup(self) -> None:
        if self._is_setup:
            return

        self.transfer.setup()
        if self._device_index is None:
            with contextlib.suppress(Exception):  # CPU-only hosts
                if torch.accelerator.is_available():
                    self._device_index = torch.accelerator.current_device_index()
        if self._transfer_executor is None:
            self._transfer_executor = ThreadPoolExecutor(
                max_workers=self.max_transfer_workers,
                thread_name_prefix="speculators-hidden-state-transfer",
                initializer=lambda: ensure_thread_context(self._device_index),
            )
        self._is_setup = True

    def setup(self) -> None:
        """Public setup for non-DataLoader consumers (streaming loader).

        Primes the persistent client on the caller's thread: the streaming
        loader calls this from its main thread before starting the pipeline
        event loop, so the client is created under the trackable context.
        """
        self._setup()
        if self._client is None:
            self._client = self._make_client()

    def _recycle_transfer_executor(self) -> None:
        """Return leaked Mooncake receive buffers to the OS (r8).

        Probes on s96 showed each ``get_into`` can retain one payload-sized
        anonymous mapping in the engine's native layer. Dropping or closing
        the engine object does NOT free them (probes M/N); only retiring the
        executor threads and then creating+closing a fresh event loop does
        (probes E/O1: -all retained maps). Cheap (thread churn only), safe
        between batches: no transfer is in flight once get_batch returned.
        Recreated threads re-bind the NPU context via the initializer
        (mooncake-integration: CANN contexts are thread-local).
        """
        if self._transfer_executor is not None:
            self._transfer_executor.shutdown(wait=True)
            self._transfer_executor = None
        asyncio.Runner().close()
        gc.collect()
        self._transfer_executor = ThreadPoolExecutor(
            max_workers=self.max_transfer_workers,
            thread_name_prefix="speculators-hidden-state-transfer",
            initializer=lambda: ensure_thread_context(self._device_index),
        )
        self._loads_since_recycle = 0

    def _make_client(self) -> openai.AsyncOpenAI:
        """Create the short-lived client used for a single batch."""
        return openai.AsyncOpenAI(
            base_url=self.vllm_endpoint,
            api_key="EMPTY",
            max_retries=0,
        )

    async def generate(
        self,
        record: DatasetRecord,
        extra_headers: dict[str, str] | None = None,
    ) -> GenerationReceipt:
        """Send one request and return its receipt without fetching the payload.

        Used by the streaming loader, which decouples request submission from
        batch assembly; ``load_generated`` completes the round trip. Runs on
        the caller's event loop with the persistent client created in
        ``_setup`` (the streaming loader owns one long-lived loop; the torch
        path never uses this method).
        """
        self._setup()
        if self._client is None:  # pragma: no cover - defensive
            self._client = self._make_client()
        handle, transfer_params = await generate_hidden_states_async(
            self._client,
            self.model,
            build_client_item(record),
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            extra_headers=extra_headers,
        )
        return GenerationReceipt(handle, transfer_params)

    def load_generated(
        self,
        record: DatasetRecord,
        receipt: GenerationReceipt,
    ) -> dict[str, torch.Tensor]:
        try:
            loaded_hs = self.transfer.get_generated(receipt.transfer_params)
            if loaded_hs is None:
                raise ValueError(
                    f"Failed to load hidden states for handle {receipt.handle}"
                )

            # Covers token/shape mismatches and non-finite values. The Mooncake
            # transfer performs manifest validation first.
            check_hidden_states(loaded_hs, record.input_ids.tolist())

            try:
                self.transfer.delete(receipt.handle)
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup is best effort
                logger.warning(
                    "Loaded a valid hidden-state sample but failed to delete "
                    "handle %s: %s",
                    receipt.handle,
                    cleanup_error,
                )
            return loaded_hs
        except Exception:
            try:
                self.transfer.delete(receipt.handle)
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup is best effort
                logger.warning(
                    "Failed to clean generated hidden-state handle %s: %s",
                    receipt.handle,
                    cleanup_error,
                )
            raise

    def get_one(
        self,
        record: DatasetRecord,
    ) -> HiddenStateResult:
        """Generate one sample through the batch path."""
        return self.get_batch([record])[0]

    def get_batch(self, records: Sequence[DatasetRecord]) -> list[HiddenStateResult]:
        """Generate samples concurrently and return results in input order."""
        if not records:
            return []
        if not self._is_setup:
            self._setup()
        # r7 (2026-11-06): run each batch on a throwaway client + event loop.
        # One long-lived loop retains roughly payload-sized native buffers
        # per run_until_complete call on this build (~1.15 GB per generated
        # sample per rank), which OOM'd the host at global step 2030 when the
        # source lived for a whole epoch. Closing the loop after every batch
        # bounds host RSS. The transfer store and its registered staging
        # buffers are loop-independent and keep being reused across batches.
        client = self._make_client()
        runner = asyncio.Runner()
        try:
            result = runner.run(self._get_batch_async(client, records))
            runner.run(client.close())
            return result
        finally:
            runner.close()
            self._loads_since_recycle += len(records)
            if self._loads_since_recycle >= self.executor_recycle_interval:
                self._recycle_transfer_executor()

    async def _get_batch_async(
        self,
        client: openai.AsyncOpenAI,
        records: Sequence[DatasetRecord],
    ) -> list[HiddenStateResult]:
        """Run one independent worker for each record in the batch.

        All requests of one batch share an ``X-Batch-Id`` header so a
        batch-affinity load balancer can pin them to the same vLLM instance
        (the pack budget matches vLLM's ``max_num_batched_tokens``, letting
        the scheduler co-batch them into a single engine step).
        """
        if self._transfer_executor is None:
            raise RuntimeError("Hidden-state source was not set up")
        batch_headers = {"X-Batch-Id": uuid.uuid4().hex}
        tasks = [
            asyncio.create_task(
                self._worker(
                    client,
                    record,
                    self._transfer_executor,
                    batch_headers,
                )
            )
            for record in records
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for result in results:
            if isinstance(result, BaseException):
                raise result
        return cast("list[HiddenStateResult]", results)

    async def _worker(
        self,
        client: openai.AsyncOpenAI,
        record: DatasetRecord,
        transfer_executor: ThreadPoolExecutor,
        batch_headers: dict[str, str] | None = None,
    ) -> HiddenStateResult:
        """Generate, transfer, and validate one sample with recovery."""
        description = (
            f"Hidden-state round trip failed for storage index {record.storage_index}"
        )

        async def generate_and_transfer() -> dict[str, torch.Tensor]:
            handle, transfer_params = await generate_hidden_states_async(
                client,
                self.model,
                build_client_item(record),
                timeout=self.request_timeout,
                max_retries=self.max_retries,
                extra_headers=batch_headers,
            )
            receipt = GenerationReceipt(handle, transfer_params)
            future = transfer_executor.submit(self.load_generated, record, receipt)
            while not future.done():  # noqa: ASYNC110 - keep request tasks running
                await asyncio.sleep(0.001)
            return future.result()

        return await self.generation_recovery.run_async(
            generate_and_transfer,
            description=description,
        )

    def close(self) -> None:
        """Shut down the shared transfer executor and streaming client.

        Torch-path clients and event loops are created per batch and closed
        there (see ``get_batch``); the streaming client and the executor
        outlive a batch and are released here.
        """
        if self._client is not None:
            runner = asyncio.Runner()
            try:
                with contextlib.suppress(Exception):  # best-effort close
                    runner.run(self._client.close())
            finally:
                runner.close()
            self._client = None
        if self._transfer_executor is not None:
            self._transfer_executor.shutdown(wait=True, cancel_futures=True)
            self._transfer_executor = None
        self._is_setup = False
