"""Streaming training loader: requests decoupled from batch boundaries.

The torch path (``create_train_val_loaders``) pre-packs the whole dataset with
``MultipackDistributedBatchSamplerV2`` and fetches each batch's hidden states
inside a DataLoader worker, so a batch's requests are only issued once the
sampler reaches it and one slow response holds up every already-ready batch
behind it. This loader issues requests up front, packs closed batches as
responses arrive, and yields them in completion order.

Two packings cooperate:

- *Send batches* — maximal token prefixes of the rank slice, deterministic
  and therefore known before any request leaves. All requests of one send
  batch carry the same ``X-Batch-Id`` header, so a batch-affinity load
  balancer can pin them to a single vLLM instance and co-batch them into one
  engine step (the budget matches ``max_num_batched_tokens``). A batch-level
  in-flight gate bounds outstanding send batches.
- *Yield batches* — packed by response arrival order (the send-batch
  boundaries are ignored on receive), so a slow sample never holds up
  already-arrived samples: zero head-of-line blocking where it matters.

Discovery is response-side (``ResponseManifestSource`` wraps
``VllmHiddenStatesSource.generate``): the vLLM response carries the transfer
manifest, so any ``HiddenStatesTransfer`` backend works unchanged. Payload
fetch, validation, and delete-after-read reuse
``VllmHiddenStatesSource.load_generated``; batch assembly reuses
``SpeculatorsDataset.collate_loaded``, keeping the yielded ``BatchType``
field-for-field identical to the torch path.

Resume is sample-level: the loader records which slice records were consumed
(``state_dict``); batch boundaries depend on response arrival order, so the
exact batch sequence of a crashed run is not reproducible, but no consumed
record is trained twice.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from hs_connectors.device import ensure_thread_context
from speculators.train.recovery import GenerationRecoveryGuard, SampleUnavailable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from speculators.train.data import BatchType
    from speculators.train.data_types import DatasetRecord, GenerationReceipt
    from speculators.train.hidden_states_source import VllmHiddenStatesSource

logger = logging.getLogger(__name__)

# Batches buffered ahead of the training loop. Depths beyond a few batches buy
# jitter tolerance, not throughput; steady state is governed by
# ``in_flight_batches`` and the vLLM service rate.
DEFAULT_YIELD_QUEUE_DEPTH = 4
_MIN_QUEUED_BATCHES = 2
_MAX_SAMPLE_LEN = int(os.environ.get("SPECULATORS_MAX_SAMPLE_LEN", "0"))

_SENTINEL = object()


@dataclass
class _PendingEntry:
    """One manifest-arrived sample awaiting batch closure."""

    record: DatasetRecord
    receipt: GenerationReceipt


@dataclass
class _EpochController:
    """``batch_sampler`` shim so the trainer's ``set_epoch`` calls re-seed us."""

    loader: StreamingDataLoader
    _cached_generated_batches: tuple[int, list] = field(default=(-1, []))

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.loader._epoch:  # noqa: SLF001
            self.loader._epoch = epoch  # noqa: SLF001
            # A new epoch re-slices the dataset; the previous epoch's
            # consumed bitmap refers to a different slice and must go.
            self.loader._consumed.clear()  # noqa: SLF001


class StreamingPacker:
    """Arrival-order first-fit packing into a token-budgeted open bin.

    Offline Multipack packs the full dataset LPT-optimally; here samples arrive
    one by one, so a single open bin is closed as soon as the next sample does
    not fit (``add`` returns the closed entries). Packing uses only
    ``record.seq_len``, known before the response arrives, so the bin fills
    while requests are still in flight. A sample longer than the budget closes
    the bin and becomes its own batch, matching the sampler's
    truncate-long-samples behaviour of never failing the epoch.
    """

    def __init__(self, batch_max_length: int):
        self.batch_max_length = batch_max_length
        self._entries: list[_PendingEntry] = []
        self._fill = 0

    def add(
        self, record: DatasetRecord, receipt: GenerationReceipt
    ) -> list[_PendingEntry] | None:
        """Add one arrived sample; return the closed batch, if any."""
        closed = None
        if self._fill + record.seq_len > self.batch_max_length and self._entries:
            closed = self._entries
            self._entries = []
            self._fill = 0
        # Oversized samples still get their own batch rather than an exception:
        # the collate path pads to ``max_len`` and the trainer's error-record
        # accounting handles the rest.
        self._entries.append(_PendingEntry(record, receipt))
        self._fill += min(record.seq_len, self.batch_max_length)
        return closed

    def flush(self) -> list[_PendingEntry] | None:
        """Close the trailing partial bin (drop_last is the trainer's choice)."""
        if not self._entries:
            return None
        closed = self._entries
        self._entries = []
        self._fill = 0
        return closed


class StreamingDataLoader:
    """Backend-agnostic streaming loader, verl ``StatefulPromptDataLoader`` shaped.

    Contract: ``__iter__`` (required), ``state_dict`` / ``load_state_dict``
    (sample-level resume), ``close``. The trainer-facing attributes it touches
    are provided: ``batch_sampler`` (a ``set_epoch`` shim without the
    sampler's fast-skip API) and ``__len__`` (a packing estimate).

    A background thread runs the asyncio loop: it keeps at most
    ``in_flight_batches`` send batches outstanding (the LB-style
    one-slot-per-batch accounting — a batch's slot is held until its last
    request completes, so in-flight requests are bounded by the batch gate,
    not by a request-level cap) while ``max_queued_batches`` bounds
    total pipeline occupancy (submission coupled to consumption),
    packs arrived manifests, and submits closed batches to a fetch thread
    pool. Assembled batches land in a bounded queue; when it fills, the loop
    stops submitting new requests, which is the backpressure onto vLLM.
    """

    def __init__(
        self,
        dataset: Any,  # SpeculatorsDataset
        source: VllmHiddenStatesSource,
        *,
        batch_max_length: int,
        rank: int,
        world_size: int,
        in_flight_batches: int = 2,
        max_queued_batches: int = 4,
        fetch_workers: int = 4,
        yield_queue_depth: int = DEFAULT_YIELD_QUEUE_DEPTH,
        on_missing: str = "skip",
        max_consecutive_failures: int = 20,
        seed: int = 0,
        max_batches: int | None = None,
    ):
        if in_flight_batches <= 0:
            raise ValueError("in_flight_batches must be positive")
        if max_queued_batches < _MIN_QUEUED_BATCHES:
            # Below 2 the open packer bin (1) plus one outstanding send batch
            # can pin the occupancy at the limit and stall the pipeline.
            raise ValueError("max_queued_batches must be at least 2")
        if fetch_workers <= 0 or yield_queue_depth <= 0:
            raise ValueError("fetch_workers and yield_queue_depth must be positive")
        if not (0 <= rank < max(world_size, 1)):
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        if on_missing not in ("generate", "skip", "warn", "raise"):
            raise ValueError("on_missing must be one of generate/skip/warn/raise")

        self.dataset = dataset
        self.source = source
        self.batch_max_length = batch_max_length
        self.rank = rank
        self.world_size = world_size
        self.in_flight_batches = in_flight_batches
        self.max_queued_batches = max_queued_batches
        self.fetch_workers = fetch_workers
        self.yield_queue_depth = yield_queue_depth
        self.seed = seed
        self.max_batches = max_batches
        # ``on_missing`` mirrors the CLI knob. For a streaming loader every
        # sample is generated at runtime, so the values describe the failure
        # policy instead of the data-acquisition mode:
        #   generate - regenerate a failed sample through the same
        #              GenerationRecoveryGuard the torch path uses (same
        #              retry count, same consecutive-failure circuit breaker),
        #              then degrade to a structured result
        #   skip     - degrade to SampleUnavailable without logging
        #   warn     - degrade and log a warning per failed sample
        #   raise    - abort the iteration on the first failure
        self.on_missing = on_missing
        self._generation_guard: GenerationRecoveryGuard | None = None
        if on_missing == "generate":
            # Reuse the source's guard so the retry count is literally the
            # baseline's ``generation_validation_retries`` configuration.
            self._generation_guard = getattr(
                source, "generation_recovery", None
            ) or GenerationRecoveryGuard(
                retries=2, max_consecutive_failures=max_consecutive_failures
            )
        self.batch_sampler = _EpochController(self)
        self._epoch = 0
        self._consumed: set[int] = set()
        self._batches_yielded = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._fetch_executor: ThreadPoolExecutor | None = None
        self._device_index: int | None = None
        self._fetch_recycle_interval = 0
        self._fetches_since_recycle = 0
        self._queue: Queue = Queue(maxsize=yield_queue_depth)
        self._closed = False
        self._iterating = False
        self._pipeline_error: BaseException | None = None

    # -- slice and resume ---------------------------------------------------

    def _slice_indices(self, epoch: int) -> np.ndarray:
        """Rank slice of the shuffled dataset (same seeding as Multipack)."""
        lengths = np.asarray(self.dataset.approx_lengths)
        valid = np.nonzero(
            lengths <= max(self.batch_max_length, _MAX_SAMPLE_LEN)
        )[0]
        rng = np.random.default_rng(seed=self.seed + epoch)
        shuffled = rng.permutation(valid)
        # Multipack balances replicas via LPT; arrival-order packing cannot,
        # so take a strided split, which balances total tokens in expectation.
        return shuffled[self.rank :: self.world_size]

    def state_dict(self) -> dict[str, Any]:
        return {"consumed": sorted(self._consumed), "epoch": self._epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        slice_indices = set(self._slice_indices(int(state.get("epoch", 0))).tolist())
        self._consumed = slice_indices.intersection(state.get("consumed", ()))
        self._epoch = int(state.get("epoch", 0))

    # -- iteration ------------------------------------------------------------

    def __len__(self) -> int:
        slice_indices = self._slice_indices(self._epoch)
        remaining = len(slice_indices) - len(self._consumed)
        lengths = np.asarray(self.dataset.approx_lengths)[slice_indices]
        if remaining <= 0:
            return 0
        avg = max(int(lengths.sum()) // max(remaining, 1), 1)
        per_batch = max(self.batch_max_length // avg, 1)
        return (remaining + per_batch - 1) // per_batch

    def __iter__(self) -> Iterator[BatchType]:
        if self._iterating:
            raise RuntimeError(
                "StreamingDataLoader does not support re-entrant iteration"
            )
        return self._iterate()

    def _iterate(self) -> Iterator[BatchType]:  # noqa: C901
        self._iterating = True
        try:
            self._pipeline_error = None
            self.source.setup()
            # Capture the rank's device index in this (main) thread so pool
            # threads bind the correct NPU context (fresh threads default
            # to device 0). CANN contexts are thread-local and mooncake's
            # adxl engine executes fabric transfers in the calling thread.
            _device_index = None
            with contextlib.suppress(Exception):
                if torch.accelerator.is_available():
                    _device_index = torch.accelerator.current_device_index()
            self._device_index = _device_index

            self._fetch_executor = ThreadPoolExecutor(
                max_workers=self.fetch_workers,
                thread_name_prefix="speculators-streaming-fetch",
                initializer=lambda: ensure_thread_context(_device_index),
            )
            # Optional r8-adapted fetch recycling (off by default): each
            # get_into can leave one payload-sized anonymous mapping that the
            # mooncake engine only returns after the calling threads exit and
            # a fresh event loop is created and closed (s96 probes E/O1).
            # Enable with STREAMING_FETCH_RECYCLE=<samples per recycle>.
            self._fetch_recycle_interval = int(
                os.environ.get("STREAMING_FETCH_RECYCLE", "0") or 0
            )
            self._fetches_since_recycle = 0
            self._queue = Queue(maxsize=self.yield_queue_depth)
            self._loop = asyncio.new_event_loop()
            self._closed = False
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="speculators-streaming-loop"
            )
            self._thread.start()
            self._loop.call_soon_threadsafe(
                asyncio.run_coroutine_threadsafe, self._run(), self._loop
            )
            try:
                while True:
                    try:
                        item = self._queue.get(timeout=1.0)
                    except Empty:
                        if self._thread is not None and not self._thread.is_alive():
                            raise RuntimeError(
                                "streaming loader worker thread died"
                            ) from None
                        continue
                    if item is _SENTINEL:
                        if self._pipeline_error is not None:
                            # Surface pipeline failures (on_missing="raise"
                            # or unexpected errors) instead of silently
                            # ending the epoch short.
                            raise self._pipeline_error
                        break
                    entries, batch = item
                    if batch is None:
                        # Assembly failed entirely; records already counted as
                        # consumed so a resume will not regenerate them.
                        for entry in entries:
                            self._consumed.add(entry.record.storage_index)
                        continue
                    for entry in entries:
                        self._consumed.add(entry.record.storage_index)
                    self._batches_yielded += 1
                    yield batch
                    if (
                        self.max_batches is not None
                        and self._batches_yielded >= self.max_batches
                    ):
                        self.close()
                        break
            finally:
                self.close()
        finally:
            self._iterating = False

    def close(self) -> None:
        """Cancel in-flight work; safe to call repeatedly."""
        loop = self._loop
        if loop is not None and self._thread is not None and self._thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=10)
        self._loop = None
        self._thread = None
        if self._fetch_executor is not None:
            self._fetch_executor.shutdown(wait=True, cancel_futures=True)
            self._fetch_executor = None

    # -- pipeline internals ---------------------------------------------------

    async def _run(self) -> None:
        """Drive the request pool, packer, and fetch submissions."""
        try:
            await self._run_inner()
        except BaseException as exc:
            logger.exception("Streaming loader pipeline failed")
            self._pipeline_error = exc
        finally:
            self._closed = True
            self._queue.put(_SENTINEL)

    async def _run_inner(self) -> None:  # noqa: C901
        slice_indices = self._slice_indices(self._epoch)
        packer = StreamingPacker(self.batch_max_length)
        pending: set[asyncio.Task] = set()
        fetches: list[Future] = []
        consumed = set(self._consumed)
        window: list[int] = [i for i in slice_indices.tolist() if i not in consumed]

        # Send batches: maximal prefixes of the slice (deterministic, so the
        # batch id each request must carry is known at submission time).
        # Requests of one send batch share an X-Batch-Id header so a
        # batch-affinity load balancer can pin them to one vLLM instance and
        # co-batch them into a single engine step; the receive side ignores
        # these boundaries and re-packs by arrival order (zero head-of-line
        # blocking). The batch-level in-flight gate keeps at most
        # ``in_flight_batches`` send batches outstanding.
        send_batches = self._send_batches(window)
        batch_pos = 0
        outstanding: dict[str, int] = {}
        task_batch: dict[asyncio.Task, str] = {}
        fetch_pending = 0

        def submit_more() -> None:
            # Pipeline-depth gate: admission is coupled to consumption. Each
            # station is counted in its own batch unit — outstanding send
            # batches, the open packer bin, closed batches awaiting fetch,
            # and assembled batches in the yield queue. When the trainer
            # stalls, occupancy stops draining and no new requests leave, so
            # payloads do not pile up in the transfer storage and vLLM does
            # not compute ahead of a consumer that is not there.
            nonlocal batch_pos, fetch_pending
            while batch_pos < len(send_batches):
                occupancy = (
                    len(outstanding)
                    + (1 if packer._entries else 0)  # noqa: SLF001 - same module
                    + fetch_pending
                    + self._queue.qsize()
                )
                if occupancy >= self.max_queued_batches:
                    break
                # Co-batch gate: a send batch's requests must queue together
                # at the instance, so outstanding send batches are bounded.
                if len(outstanding) >= self.in_flight_batches:
                    break
                indices = send_batches[batch_pos]
                batch_pos += 1
                batch_id = uuid.uuid4().hex
                outstanding[batch_id] = len(indices)
                for index in indices:
                    record = self.dataset.records[index]
                    task = asyncio.create_task(self._generate(index, record, batch_id))
                    pending.add(task)
                    task_batch[task] = batch_id

        submit_more()
        while pending or batch_pos < len(send_batches):
            submit_more()
            if pending:
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                # Submission is blocked by the depth gate with nothing in
                # flight: occupancy can only drop from another thread (a
                # fetch retiring, the trainer consuming). Poll cheaply and
                # re-check instead of awaiting an empty task set.
                await asyncio.sleep(0.05)
                done = set()
            for task in done:
                pending.discard(task)
                batch_id = task_batch.pop(task, None)
                if batch_id is not None:
                    remaining = outstanding.get(batch_id, 0) - 1
                    if remaining > 0:
                        outstanding[batch_id] = remaining
                    else:
                        outstanding.pop(batch_id, None)
                result = task.result()
                if result is None:
                    continue
                index, record, receipt = result
                closed = packer.add(record, receipt)
                if closed is not None:
                    fetches.append(self._submit_fetch(closed))
                    fetch_pending += 1
            # Retire completed fetches; a failed one surfaces immediately.
            for future in fetches:
                if future.done():
                    fetch_pending -= 1
                    self._fetches_since_recycle += 1
                    if future.exception() is not None:
                        future.result()
            fetches = [future for future in fetches if not future.done()]
            if (
                self._fetch_recycle_interval > 0
                and not fetches
                and self._fetches_since_recycle >= self._fetch_recycle_interval
            ):
                self._recycle_fetch_executor()

        final = packer.flush()
        if final is not None:
            fetches.append(self._submit_fetch(final))
            fetch_pending += 1
        for future in fetches:
            future.result()

    def _send_batches(self, window: list[int]) -> list[list[int]]:
        """Split the slice into send batches by maximal token prefix.

        Deterministic (lengths are known offline), so each batch's member set
        is fixed before any request leaves. Each send batch totals at most
        ``batch_max_length`` tokens — the budget a batch-affinity load
        balancer assumes a vLLM instance can co-batch in one engine step.
        """
        lengths = self.dataset.approx_lengths
        batches: list[list[int]] = []
        current: list[int] = []
        fill = 0
        for index in window:
            seq = min(int(lengths[index]), self.batch_max_length)
            if current and fill + seq > self.batch_max_length:
                batches.append(current)
                current, fill = [], 0
            current.append(index)
            fill += seq
        if current:
            batches.append(current)
        return batches

    async def _generate(
        self, index: int, record: DatasetRecord, batch_id: str | None = None
    ) -> tuple[int, DatasetRecord, GenerationReceipt | SampleUnavailable] | None:
        """One request; failures degrade per the ``on_missing`` policy.

        Degrading always produces a structured ``SampleUnavailable`` that
        flows into the batch (never a silent drop), so ``error_records`` and
        ``RecoveryMetadata`` stay observable downstream. ``batch_id`` rides
        along as the X-Batch-Id header so a batch-affinity load balancer can
        co-batch this request's send batch on one vLLM instance.
        """
        headers = {"X-Batch-Id": batch_id} if batch_id is not None else None
        try:
            if self._generation_guard is not None:
                result = await self._generation_guard.run_async(
                    lambda: self.source.generate(record, extra_headers=headers),
                    description=(
                        "Streaming generation failed for storage index "
                        f"{record.storage_index}"
                    ),
                )
                if isinstance(result, SampleUnavailable):
                    return index, record, result
                return index, record, result
            receipt = await self.source.generate(record, extra_headers=headers)
        except Exception as exc:
            if self.on_missing == "raise":
                raise
            if self.on_missing == "warn":
                logger.warning(
                    "Streaming generation failed for storage index %s: %r; skipping",
                    record.storage_index,
                    exc,
                )
            return (
                index,
                record,
                SampleUnavailable(
                    reason=f"Streaming generation failed: {exc!r}",
                    counts_as_failure=True,
                ),
            )
        return index, record, receipt

    def _submit_fetch(self, entries: list[_PendingEntry]) -> Future:
        if self._fetch_executor is None:
            raise RuntimeError("streaming fetch executor is not running")
        return self._fetch_executor.submit(self._fetch_and_assemble, entries)

    def _recycle_fetch_executor(self) -> None:
        """Return leaked mooncake receive buffers to the OS (r8, streaming).

        Same recipe as ``VllmHiddenStatesSource._recycle_transfer_executor``:
        retire the fetch threads (each ``get_into`` can retain one
        payload-sized anonymous mapping), create+close a fresh event loop on
        this thread to release the engine's per-thread native state, then
        recreate the pool; the initializer re-binds the NPU context (CANN
        contexts are thread-local). Only called with no fetch in flight.
        """
        if self._fetch_executor is not None:
            self._fetch_executor.shutdown(wait=True)
            self._fetch_executor = None
        fresh_loop = asyncio.new_event_loop()
        fresh_loop.close()
        gc.collect()
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=self.fetch_workers,
            thread_name_prefix="speculators-streaming-fetch",
            initializer=lambda: ensure_thread_context(self._device_index),
        )
        self._fetches_since_recycle = 0

    def _fetch_and_assemble(
        self, entries: list[_PendingEntry]
    ) -> tuple[list[_PendingEntry], BatchType | None]:
        """Fetch payloads and collate; failures degrade per-sample."""
        loaded: list[Any] = []
        for entry in entries:
            if isinstance(entry.receipt, SampleUnavailable):
                # Generation already degraded upstream: carry the structured
                # result straight into the collate path.
                loaded.append(entry.receipt)
                continue
            # Mirror the generation guard's retry count (baseline retries the
            # whole round trip; here the fetch half gets the same budget).
            attempts = (
                self._generation_guard.retries + 1
                if self._generation_guard is not None
                else 1
            )
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    loaded.append(
                        self.source.load_generated(entry.record, entry.receipt)
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if self.on_missing == "raise":
                        raise
                    if self.on_missing in ("warn", "generate") and attempt == attempts:
                        logger.warning(
                            "Streaming fetch failed for handle %s: %r; skipping sample",
                            entry.receipt.handle,
                            exc,
                        )
            if last_exc is not None:
                loaded.append(
                    SampleUnavailable(
                        reason=f"Streaming fetch failed: {last_exc!r}",
                        counts_as_failure=True,
                    )
                )
        indices = [entry.record.storage_index for entry in entries]
        records = [entry.record for entry in entries]
        batch = self.dataset.collate_loaded(indices, records, loaded)
        item = (entries, batch)
        while not self._closed:
            try:
                self._queue.put(item, timeout=1.0)
                break
            except Full:
                continue
        return item
