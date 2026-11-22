"""Unit tests for the streaming dataloader (mocked vLLM source and transfer).

The streaming loader's contract pieces are exercised without network or Ray:
packing correctness under out-of-order arrival, sample-level resume, error
skipping, and parity of the collated batch with ``SpeculatorsDataset.collate_loaded``.
"""

from __future__ import annotations

import pytest
import torch

from speculators.train.data_types import DatasetRecord, GenerationReceipt
from speculators.train.streaming_dataloader import StreamingDataLoader, StreamingPacker


def _record(index: int, seq_len: int) -> DatasetRecord:
    return DatasetRecord(
        input_ids=torch.arange(seq_len, dtype=torch.long) + index,
        loss_mask=torch.ones(seq_len, dtype=torch.bool),
        seq_len=seq_len,
        storage_index=index,
        messages=None,
    )


def _hidden_states(record: DatasetRecord, num_layers: int = 2, hidden: int = 4):
    seq_len = record.input_ids.shape[0]
    return {
        "hidden_states": torch.randn(seq_len, num_layers, hidden),
        "token_ids": record.input_ids.clone(),
    }


class _FakeSource:
    """Stands in for VllmHiddenStatesSource: controllable delays and failures."""

    def __init__(
        self,
        *,
        delay_index: dict[int, float] | None = None,
        fail_generate: set[int] | None = None,
        fail_load: set[int] | None = None,
    ):
        self.delay_index = delay_index or {}
        self.fail_generate = fail_generate or set()
        self.fail_load = fail_load or set()
        self.records: dict[int, DatasetRecord] = {}
        self.setup_called = False
        self.deleted: list[str] = []
        self.headers: dict[int, str | None] = {}
        self._inflight: dict[str, int] = {}
        self.max_inflight_batches = 0

    def setup(self) -> None:
        self.setup_called = True

    async def generate(
        self, record: DatasetRecord, extra_headers: dict | None = None
    ) -> GenerationReceipt:
        import asyncio  # noqa: PLC0415

        batch_id = (extra_headers or {}).get("X-Batch-Id")
        self.headers[record.storage_index] = batch_id
        if batch_id is not None:
            self._inflight[batch_id] = self._inflight.get(batch_id, 0) + 1
            self.max_inflight_batches = max(
                self.max_inflight_batches,
                sum(1 for v in self._inflight.values() if v > 0),
            )
        delay = self.delay_index.get(record.storage_index)
        if delay is not None:
            await asyncio.sleep(delay)
        if batch_id is not None:
            self._inflight[batch_id] -= 1
        if record.storage_index in self.fail_generate:
            raise RuntimeError(f"generate failed for {record.storage_index}")
        self.records[record.storage_index] = record
        return GenerationReceipt(f"handle-{record.storage_index}", {})

    def load_generated(
        self, record: DatasetRecord, receipt: GenerationReceipt
    ) -> dict[str, torch.Tensor]:
        if record.storage_index in self.fail_load:
            raise RuntimeError(f"load failed for {record.storage_index}")
        assert receipt.handle == f"handle-{record.storage_index}"
        return _hidden_states(record)


class _FakeRecords:
    """Arrow-shaped indexable backing store."""

    def __init__(self, records: list[DatasetRecord]):
        self._records = records

    def __getitem__(self, index: int) -> DatasetRecord:
        return self._records[index]

    def __getitems__(self, indices: list[int]) -> list[DatasetRecord]:
        return [self._records[i] for i in indices]


class _FakeDataset:
    """SpeculatorsDataset-shaped: reuses the real collate path."""

    def __init__(self, records: list[DatasetRecord], max_len: int):
        from speculators.train.data import SpeculatorsDataset  # noqa: PLC0415

        self.records = _FakeRecords(records)
        self.approx_lengths = [r.seq_len for r in records]
        self.max_len = max_len
        self._inner = SpeculatorsDataset.__new__(SpeculatorsDataset)
        self._inner.max_len = max_len
        self._inner.hidden_size = 4
        self._inner.num_target_layers = 1
        self._inner.hidden_states_dtype = torch.bfloat16
        self._inner.preprocess = None

    def collate_loaded(self, indices, records, loaded_hs):
        return self._inner.collate_loaded(indices, records, loaded_hs)


def _make_loader(records, *, source=None, **kwargs):
    defaults = {
        "batch_max_length": 64,
        "rank": 0,
        "world_size": 1,
        "max_queued_batches": 8,
        "fetch_workers": 2,
        "yield_queue_depth": 2,
        "seed": 0,
    }
    defaults.update(kwargs)
    source = source or _FakeSource()
    dataset = _FakeDataset(records, max_len=defaults["batch_max_length"])
    return StreamingDataLoader(dataset, source, **defaults), source


class TestStreamingPacker:
    def test_arrival_order_packing_respects_budget(self):
        packer = StreamingPacker(100)
        closed = []
        for seq_len in (40, 40, 30, 30, 30):
            out = packer.add(_record(0, seq_len), None)
            if out is not None:
                closed.append(out)
        out = packer.flush()
        if out is not None:
            closed.append(out)
        sums = [sum(e.record.seq_len for e in batch) for batch in closed]
        assert all(s <= 100 for s in sums)
        assert sum(sums) == 170

    def test_oversized_sample_gets_own_batch(self):
        packer = StreamingPacker(100)
        packer.add(_record(0, 50), None)
        closed = packer.add(_record(1, 120), None)
        assert closed is not None  # previous bin closed
        assert len(closed) == 1
        assert packer.flush() is not None  # oversized sample alone

    def test_flush_empty_returns_none(self):
        assert StreamingPacker(100).flush() is None


class TestStreamingDataLoader:
    def test_yields_all_samples_once_in_any_order(self):
        records = [_record(i, 10 + i) for i in range(8)]
        loader, _ = _make_loader(records)
        batches = list(loader)
        total = sum(int(batch["loss_mask"].sum()) for batch in batches)
        assert total == sum(r.seq_len for r in records)
        assert loader.state_dict()["consumed"] == [r.storage_index for r in records]
        loader.close()

    def test_out_of_order_arrival_loses_nothing(self):
        # Long delays on the first submitted records force later arrivals to
        # pack first; the sample set must still come out complete.
        records = [_record(i, 10) for i in range(8)]
        source = _FakeSource(delay_index={0: 0.3, 1: 0.2})
        loader, _ = _make_loader(records, source=source)
        batches = list(loader)
        total = sum(int(batch["loss_mask"].sum()) for batch in batches)
        assert total == 80
        loader.close()

    def test_generation_failure_skips_sample(self):
        records = [_record(i, 10) for i in range(8)]
        source = _FakeSource(fail_generate={3})
        loader, _ = _make_loader(records, source=source)
        batches = list(loader)
        total = sum(int(batch["loss_mask"].sum()) for batch in batches)
        assert total == 70
        assert 3 in loader.state_dict()["consumed"]  # counted, not retried
        loader.close()

    def test_load_failure_degrades_to_error_records(self):
        from speculators.train.recovery import RECOVERY_METADATA_KEY  # noqa: PLC0415

        records = [_record(i, 10) for i in range(8)]
        source = _FakeSource(fail_load={5})
        loader, _ = _make_loader(records, source=source)
        batches = list(loader)
        failing = [b for b in batches if RECOVERY_METADATA_KEY in b]
        assert failing, "expected recovery metadata on the degraded batch"
        total = sum(int(batch["loss_mask"].sum()) for batch in batches)
        assert total == 70
        loader.close()

    def test_sample_level_resume_skips_consumed(self):
        records = [_record(i, 10) for i in range(8)]
        loader, _ = _make_loader(records)
        it = iter(loader)
        first = next(it)
        consumed_before = set(loader.state_dict()["consumed"])
        loader.close()

        loader2, _ = _make_loader(records)
        state = {"consumed": sorted(consumed_before), "epoch": 0}
        loader2.load_state_dict(state)
        batches = list(loader2)
        total = sum(int(batch["loss_mask"].sum()) for batch in batches)
        remaining = sum(r.seq_len for r in records) - sum(
            records[i].seq_len for i in consumed_before
        )
        assert total == remaining
        # No consumed record is regenerated.
        assert consumed_before.issubset(set(loader2.state_dict()["consumed"]))
        loader2.close()
        assert first is not None

    def test_distributed_slice_partitions_records(self):
        records = [_record(i, 10 + (i % 7)) for i in range(20)]
        seen: set[int] = set()
        for rank in range(4):
            loader, _ = _make_loader(records, rank=rank, world_size=4)
            batches = list(loader)
            for _batch in batches:
                pass
            seen.update(loader.state_dict()["consumed"])
            loader.close()
        assert seen == set(range(20))

    def test_backpressure_bounds_inflight_requests(self):
        # No request-level cap: in-flight requests are bounded by the
        # batch-level gate (in-flight send batches x batch size), matching
        # the LB's one-slot-per-batch accounting.
        import asyncio  # noqa: PLC0415

        records = [_record(i, 10) for i in range(24)]
        in_flight = 0
        peak = 0

        class _TrackingSource(_FakeSource):
            async def generate(self, record, extra_headers=None):
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.02)
                in_flight -= 1
                return await super().generate(record, extra_headers=extra_headers)

        # 24 samples of 10 tokens, budget 64 -> send batches of 6; two
        # outstanding send batches bound peak requests at 12.
        loader, _ = _make_loader(
            records,
            source=_TrackingSource(),
            in_flight_batches=2,
            max_queued_batches=24,
        )
        list(loader)
        assert peak <= 12
        loader.close()

    def test_batch_matches_collate_loaded_golden(self):
        # Identical payloads: arrival order decides in-batch order, so the
        # golden comparison must be order-insensitive.
        ids = torch.arange(8)
        records = [
            DatasetRecord(
                input_ids=ids.clone(),
                loss_mask=torch.ones(8, dtype=torch.bool),
                seq_len=8,
                storage_index=i,
            )
            for i in range(2)
        ]
        loader, _ = _make_loader(records, batch_max_length=16)
        streaming_batch = list(loader)[0]
        loader.close()

        dataset = _FakeDataset(records, 16)
        loaded = [_hidden_states(r) for r in records]
        golden = dataset.collate_loaded(
            [r.storage_index for r in records], records, loaded
        )
        assert set(streaming_batch.keys()) == set(golden.keys())
        assert int(streaming_batch["loss_mask"].sum()) == int(golden["loss_mask"].sum())
        assert streaming_batch["hidden_states"].shape == golden["hidden_states"].shape

        # Packed rows are the concatenation of per-sample tokens; sort each
        # row-block so packing order cannot fail the comparison.
        def _row_blocks(batch):
            mask = batch["loss_mask"][0]
            blocks = []
            start = 0
            for flag in mask.tolist():
                if not flag:
                    blocks.append(batch["input_ids"][0][start:start].tolist())
                    start = batch["input_ids"][0].shape[0]
            blocks.append(batch["input_ids"][0][start:].tolist())
            return sorted(b for b in blocks if b)

        assert _row_blocks(streaming_batch) == _row_blocks(golden)
        assert streaming_batch["error_records"] == golden["error_records"]

    def test_len_is_positive_and_decreasing(self):
        records = [_record(i, 16) for i in range(8)]
        loader, _ = _make_loader(records, batch_max_length=32)
        assert len(loader) > 0
        loader.load_state_dict({"consumed": [0, 1, 2, 3], "epoch": 0})
        assert len(loader) < 8
        loader.close()

    def test_reentrant_iteration_rejected(self):
        records = [_record(i, 10) for i in range(4)]
        loader, _ = _make_loader(records)
        it = iter(loader)
        next(it)
        with pytest.raises(RuntimeError):
            iter(loader)
        loader.close()

    def test_raise_policy_propagates_generation_failure(self):
        records = [_record(i, 10) for i in range(4)]
        source = _FakeSource(fail_generate={1})
        loader, _ = _make_loader(records, source=source, on_missing="raise")
        with pytest.raises(RuntimeError, match="generate failed for 1"):
            list(loader)
        loader.close()

    def test_raise_policy_propagates_fetch_failure(self):
        records = [_record(i, 10) for i in range(4)]
        source = _FakeSource(fail_load={2})
        loader, _ = _make_loader(records, source=source, on_missing="raise")
        with pytest.raises(RuntimeError, match="load failed for 2"):
            list(loader)
        loader.close()

    def test_set_epoch_clears_consumed_bitmap(self):
        records = [_record(i, 10) for i in range(8)]
        loader, _ = _make_loader(records)
        first_total = sum(int(batch["loss_mask"].sum()) for batch in loader)
        assert first_total == 80
        assert loader.state_dict()["consumed"]
        # A new epoch re-slices with a fresh seed; the previous epoch's
        # bitmap must not silently empty the next epoch.
        loader.batch_sampler.set_epoch(1)
        second_total = sum(int(batch["loss_mask"].sum()) for batch in loader)
        assert second_total == 80
        loader.close()

    def test_generation_guard_marks_fatal_for_circuit_breaker(self):
        # on_missing=generate semantics: baseline retry budget, then a
        # structured SampleUnavailable whose fatal flag lets the trainer-side
        # BatchRecoveryCoordinator trip its distributed circuit breaker.
        from speculators.train.recovery import RECOVERY_METADATA_KEY  # noqa: PLC0415

        records = [_record(i, 10) for i in range(2)]
        source = _FakeSource(fail_generate={1})
        loader, _ = _make_loader(
            records,
            source=source,
            on_missing="generate",
            max_consecutive_failures=1,
        )
        batches = list(loader)
        loader.close()
        degraded = [b for b in batches if b.get("error_records", 0) > 0]
        assert degraded, "guard-exhausted sample must reach the batch"
        assert degraded[0]["error_records"] == 1
        metadata = degraded[0][RECOVERY_METADATA_KEY]
        assert metadata.failure_count == 1
        assert metadata.fatal, "consecutive-failure circuit breaker must trip"

    def test_generation_guard_uses_baseline_retry_budget(self):
        # Baseline default is generation_validation_retries=2 (3 attempts
        # total); the loader without a source guard matches that budget.
        records = [_record(i, 10) for i in range(2)]
        source = _FlakyThenOkSource(ok_after=2)  # succeeds on 3rd attempt
        loader, _ = _make_loader(records, source=source, on_missing="generate")
        total = sum(int(b["loss_mask"].sum()) for b in loader)
        loader.close()
        assert total == 20  # record 0 recovered on the final allowed attempt
        assert source.attempts == 4  # 3 attempts for record 0 + 1 for record 1

    def test_warn_policy_logs_but_degrades(self, caplog):
        records = [_record(i, 10) for i in range(2)]
        source = _FakeSource(fail_generate={1})
        loader, _ = _make_loader(records, source=source, on_missing="warn")
        batches = list(loader)
        loader.close()
        degraded = [b for b in batches if b.get("error_records", 0) > 0]
        assert degraded
        assert degraded[0]["error_records"] == 1
        assert any(
            "Streaming generation failed" in r.message and "skipping" in r.message
            for r in caplog.records
        )

    def test_skip_policy_is_quiet(self, caplog):
        records = [_record(i, 10) for i in range(2)]
        source = _FakeSource(fail_generate={1})
        loader, _ = _make_loader(records, source=source, on_missing="skip")
        batches = list(loader)
        loader.close()
        degraded = [b for b in batches if b.get("error_records", 0) > 0]
        assert degraded
        assert degraded[0]["error_records"] == 1
        assert not [
            r for r in caplog.records if "Streaming generation failed" in r.message
        ]


class _FlakyThenOkSource(_FakeSource):
    """Fails the first ``ok_after`` generation attempts, then succeeds."""

    def __init__(self, *, ok_after: int):
        super().__init__()
        self.ok_after = ok_after
        self.attempts = 0

    async def generate(self, record, extra_headers=None):
        self.attempts += 1
        if self.attempts <= self.ok_after:
            raise RuntimeError(f"transient failure #{self.attempts}")
        return await super().generate(record, extra_headers=extra_headers)


class TestSendBatchAffinity:
    """Send batches: maximal prefixes, X-Batch-Id grouping, in-flight gate."""

    def test_send_batches_are_maximal_prefixes(self):
        # Lengths: five 10s, a 40, two more 10s. Over the (shuffled) window
        # order every batch must be a maximal prefix: sum within budget,
        # and absorbing the next batch's first sample would exceed it.
        records = (
            [_record(i, 10) for i in range(5)]
            + [_record(5, 40)]
            + [_record(i, 10) for i in range(6, 8)]
        )
        loader, _ = _make_loader(records)
        window = loader._slice_indices(0).tolist()
        batches = loader._send_batches(window)
        loader.close()
        sums = [sum(records[i].seq_len for i in batch) for batch in batches]
        assert all(s <= 64 for s in sums)
        assert len(batches) > 1  # the 40 forces a cut somewhere
        for batch, nxt in zip(batches, batches[1:], strict=False):
            current = sum(records[i].seq_len for i in batch)
            assert current + records[nxt[0]].seq_len > 64
        # Coverage: batches partition the window without gaps or repeats.
        flat = [i for batch in batches for i in batch]
        assert sorted(flat) == sorted(window)

    def test_requests_share_batch_id_within_send_batch(self):
        records = [_record(i, 10) for i in range(8)]
        source = _FakeSource()
        loader, _ = _make_loader(records, source=source)
        list(loader)
        loader.close()
        window = loader._slice_indices(0).tolist()
        expected = loader._send_batches(window)
        # Every request carried the batch id of its send batch; ids are
        # shared within a send batch and distinct across batches.
        groups = [
            {source.headers[records[i].storage_index] for i in batch}
            for batch in expected
        ]
        assert all(len(g) == 1 for g in groups), (
            "requests of one send batch must share one X-Batch-Id"
        )
        assert len(groups) == len(expected) > 1
        assert len({next(iter(g)) for g in groups}) == len(expected)
        assert all(
            source.headers[records[i].storage_index] is not None
            for batch in expected
            for i in batch
        )

    def test_in_flight_send_batches_bounded(self):
        records = [_record(i, 10) for i in range(12)]
        # Uniform delays make requests of one send batch overlap, while the
        # gate must keep at most one send batch outstanding at a time.
        source = _FakeSource(delay_index=dict.fromkeys(range(12), 0.05))
        loader, _ = _make_loader(records, source=source, in_flight_batches=1)
        total = sum(int(b["loss_mask"].sum()) for b in loader)
        loader.close()
        assert total == 120
        assert source.max_inflight_batches <= 1, (
            "batch-level in-flight gate must bound outstanding send batches"
        )

    def test_stalled_consumer_stops_submission(self):
        # Pipeline-depth gate: when the trainer stops consuming, occupancy
        # (outstanding send batches + open packer bin + batches awaiting
        # fetch + assembled queue) pins at max_queued_batches and no further
        # requests are issued — no storage pile-up, no idle vLLM compute.
        import time  # noqa: PLC0415

        records = [_record(i, 10) for i in range(60)]
        source = _FakeSource(delay_index=dict.fromkeys(range(60), 0.02))
        loader, _ = _make_loader(
            records,
            source=source,
            max_queued_batches=3,
            in_flight_batches=2,
            yield_queue_depth=2,
        )
        it = iter(loader)
        first = int(next(it)["loss_mask"].sum())  # consume one, then stall
        time.sleep(0.4)
        issued_first = len(source.headers)
        time.sleep(0.4)
        issued_second = len(source.headers)
        assert issued_first < 60, "stall must not issue the whole slice"
        assert issued_second == issued_first, (
            "submission must stop while the trainer is not consuming"
        )
        total = first + sum(int(b["loss_mask"].sum()) for b in it)
        loader.close()
        assert total == 600
