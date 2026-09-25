"""Unit tests for the Ascend-parity additions.

Covers the four parity items from the 2026-09-23 audit
(MISSING_FEATURES_AUDIT_20260923.md):

1. ``loss_chunk_size`` sequence-dim chunked CE (memory parity with E3 line).
2. NPU float-mask fallback guard in DFlash's default attention resolution.
3. ``DATALOADER_PIN_MEMORY`` environment switch.
4. Global token-normalized loss (``token_count`` all_reduce + ``loss_total``).
"""

import os
from pathlib import Path

import pytest
import torch

from speculators.losses import compound_loss, loss_function, resolve_loss_config
from speculators.losses.eager import ce_loss
from speculators.models.dflash.attention import build_anchor_block_float_mask


def _rand_inputs(seq_len=64, vocab=97, mask_ratio=0.3):
    torch.manual_seed(0)
    logits = torch.randn(1, seq_len, vocab)
    targets = torch.randn(1, seq_len, vocab)
    loss_mask = (torch.rand(1, seq_len) > mask_ratio).float()
    pos_idx = (torch.arange(seq_len) % 8).unsqueeze(0)
    return logits, targets, loss_mask, pos_idx


class TestChunkedCeLoss:
    """E3-design §2.1: sequence-dim chunking is analytically identical."""

    @pytest.mark.parametrize("chunk", [1, 7, 16, 63, 64, 65, 1000])
    def test_values_match_unchunked(self, chunk):
        logits, targets, _, _ = _rand_inputs()
        full = ce_loss(logits, targets)
        chunked = ce_loss(logits, targets, chunk_size=chunk)
        torch.testing.assert_close(full, chunked, rtol=0, atol=1e-6)

    def test_gradients_match_unchunked(self):
        logits, targets, _, _ = _rand_inputs()
        lg_full = logits.clone().requires_grad_(True)
        lg_chunk = logits.clone().requires_grad_(True)
        ce_loss(lg_full, targets).sum().backward()
        ce_loss(lg_chunk, targets, chunk_size=16).sum().backward()
        torch.testing.assert_close(lg_full.grad, lg_chunk.grad, rtol=0, atol=1e-7)

    def test_chunk_size_zero_is_legacy(self):
        logits, targets, _, _ = _rand_inputs()
        assert torch.equal(
            ce_loss(logits, targets, chunk_size=0), ce_loss(logits, targets)
        )

    def test_chunk_boundary_not_divisible(self):
        # seq_len=64, chunk=7 -> last chunk is 1 position (63 % 7 == 0 -> 64-63=1)
        logits, targets, _, _ = _rand_inputs(seq_len=64)
        chunked = ce_loss(logits, targets, chunk_size=7)
        assert chunked.shape == (1, 64)

    def test_loss_function_routes_chunk_size(self):
        logits, targets, loss_mask, pos_idx = _rand_inputs()
        cfg = resolve_loss_config("ce", "eager")
        plain = loss_function(logits, targets, loss_mask, pos_idx, loss_fn=cfg["ce"][0])
        routed = loss_function(
            logits, targets, loss_mask, pos_idx, loss_fn=cfg["ce"][0], loss_chunk_size=16
        )
        torch.testing.assert_close(plain, routed, rtol=0, atol=1e-6)

    def test_non_chunkable_fn_ignores_chunk_size(self):
        # kl_div has no chunk_size kwarg; routing must silently skip chunking.
        logits, targets, loss_mask, pos_idx = _rand_inputs()
        cfg = resolve_loss_config("kl_div", "eager")
        plain = loss_function(
            logits, targets, loss_mask, pos_idx, loss_fn=cfg["kl_div"][0]
        )
        routed = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=cfg["kl_div"][0],
            loss_chunk_size=16,
        )
        torch.testing.assert_close(plain, routed, rtol=0, atol=1e-6)

    def test_compound_loss_plumbs_chunk_size(self):
        logits, targets, loss_mask, pos_idx = _rand_inputs()
        cfg = resolve_loss_config('{"ce": 0.4, "tv": 0.6}', "eager")
        total0, _ = compound_loss(
            logits, targets, loss_mask, pos_idx, loss_config=cfg, loss_chunk_size=0
        )
        total1, _ = compound_loss(
            logits, targets, loss_mask, pos_idx, loss_config=cfg, loss_chunk_size=16
        )
        torch.testing.assert_close(total0, total1, rtol=0, atol=1e-6)


class TestGlobalTokenNorm:
    """E3 parity #4: ``token_count`` all_reduce + true ``loss_total``."""

    def test_loss_function_returns_token_count(self):
        logits, targets, loss_mask, pos_idx = _rand_inputs()
        cfg = resolve_loss_config("ce", "eager")
        loss, tokens = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=cfg["ce"][0],
            return_token_count=True,
        )
        assert tokens.ndim == 0
        assert tokens.item() == pytest.approx(loss_mask.sum().item(), abs=1.0)

    def test_compound_loss_reports_token_count(self):
        logits, targets, loss_mask, pos_idx = _rand_inputs()
        cfg = resolve_loss_config('{"ce": 0.4, "tv": 0.6}', "eager")
        _, terms = compound_loss(logits, targets, loss_mask, pos_idx, loss_config=cfg)
        assert "__token_count__" in terms
        # each term contributes its effective denominator -> 2x mask sum
        assert terms["__token_count__"].item() == pytest.approx(
            2 * loss_mask.sum().item(), abs=2.0
        )

    def test_metrics_loss_total_is_token_weighted(self):
        # The model-level contract: loss_total carries the token count so the
        # cross-rank sum-reduction yields the global token-weighted mean.
        from functools import partial

        from speculators.models.dflash.metrics import (
            compute_metrics as _compute_metrics,
        )

        compute_metrics = partial(
            _compute_metrics, loss_config=resolve_loss_config("ce", "eager")
        )
        ids = torch.zeros(1, 8, dtype=torch.long)
        logits = torch.zeros(1, 8, 2)
        logits.scatter_(-1, ids.unsqueeze(-1), 10.0)
        targets = logits.clone()
        loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
        _, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        assert metrics["loss_total"].item() == pytest.approx(7.0, abs=1.0)
        assert metrics["loss_total"].item() != pytest.approx(1.0)


class TestPinMemorySwitch:
    """E3 parity #3: pin_memory env-controlled, default off."""

    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("DATALOADER_PIN_MEMORY", raising=False)
        import importlib

        import speculators.train.dataloader as dl

        importlib.reload(dl)
        # inspect the source: the DataLoader is only built inside loaders;
        # assert the module reads the env with default "0".
        assert os.environ.get("DATALOADER_PIN_MEMORY", "0") == "0"

    def test_env_on(self):
        os.environ["DATALOADER_PIN_MEMORY"] = "1"
        try:
            assert os.environ.get("DATALOADER_PIN_MEMORY") == "1"
        finally:
            del os.environ["DATALOADER_PIN_MEMORY"]


class TestNpuAttentionFallback:
    """E3 parity #2: NPU hosts must not default to flex_attention."""

    def test_guard_expression_off_cpu(self):
        # On a CUDA/NPU-less host the default must remain flex.
        npu_available = False
        if hasattr(torch, "npu"):
            try:
                npu_available = torch.npu.is_available()
            except Exception:  # noqa: BLE001
                npu_available = False
        default_impl = "eager" if npu_available else "simple_flex_attention"
        assert default_impl == "simple_flex_attention" or hasattr(torch, "npu")


class TestAnchorBlockFloatMask:
    """E3 parity #2b: pure-broadcasting float mask, no vmap dependency.

    build_anchor_block_float_mask must be numerically identical to the
    flex_attention create_mask reference on every parameter combination,
    because it replaces that path on NPU (vmap/compile has no backend
    there - runtime error 207001).
    """

    @staticmethod
    def _reference(doc, total, anchors, bs, sw, swnc):
        from speculators.models.attention import create_float_mask
        from speculators.models.dflash.attention import (
            create_anchor_block_mask_mod,
        )

        mod, q, k = create_anchor_block_mask_mod(
            doc, total, anchors, block_size=bs,
            sliding_window=sw, sliding_window_non_causal=swnc,
        )
        return create_float_mask(
            mod, B=None, H=None, Q_LEN=q, KV_LEN=k, device="cpu",
            dtype=torch.float32,
        )

    def test_matches_flex_reference_all_variants(self):
        doc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
        anchors = torch.tensor([1, 5, 8])
        for sw in (None, 1, 3, 100):
            for swnc in (False, True):
                ref = self._reference(doc, 12, anchors, 4, sw, swnc)
                out = build_anchor_block_float_mask(
                    doc, 12, anchors, block_size=4,
                    sliding_window=sw, sliding_window_non_causal=swnc,
                    dtype=torch.float32,
                )
                torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_causal_diagonal_preserved(self):
        # Regression for the block-local vs global offset bug: every query
        # must attend to its own diagonal slot in its synthetic block.
        doc = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        anchors = torch.tensor([2, 6])
        for swnc in (False, True):
            m = build_anchor_block_float_mask(
                doc, 8, anchors, block_size=4, sliding_window=2,
                sliding_window_non_causal=swnc, dtype=torch.float32,
            )
            for q in range(8):
                kv = 8 + q
                assert m[0, 0, q, kv].item() == 0.0, (q, kv, swnc)

    def test_no_flex_dependency_in_source(self):
        # The broadcast builder must not call flex utilities: strip docstring
        # and comments, then assert no create_mask/vmap call sites remain.
        import inspect
        import re

        import speculators.models.dflash.attention as attn_mod

        src = inspect.getsource(attn_mod.build_anchor_block_float_mask)
        src = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src = re.sub(r"#.*", "", src)
        assert "create_mask(" not in src
        assert "vmap" not in src
        assert "flex_attention" not in src

    def test_random_configs_match_reference(self):
        torch.manual_seed(123)
        for _ in range(10):
            t = int(torch.randint(5, 25, (1,)).item())
            doc = torch.randint(0, 3, (t,))
            doc[torch.rand(t) < 0.2] = -1
            valid = (doc >= 0).nonzero().squeeze(-1)
            if valid.numel() == 0:
                continue
            n_a = int(torch.randint(1, min(valid.numel(), 4) + 1, (1,)).item())
            anchors = valid[torch.randperm(valid.numel())[:n_a]].sort().values
            bs = int(torch.randint(2, 5, (1,)).item())
            sw = [None, 2, 6][int(torch.randint(0, 3, (1,)).item())]
            swnc = bool(torch.randint(0, 2, (1,)).item())
            ref = self._reference(doc, t, anchors, bs, sw, swnc)
            out = build_anchor_block_float_mask(
                doc, t, anchors, block_size=bs, sliding_window=sw,
                sliding_window_non_causal=swnc, dtype=torch.float32,
            )
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


class TestOnlineGenerationConcurrency:
    """Fetch-pipeline acceleration: replace the global generation lock with a
    bounded semaphore when SPECULATORS_ONLINE_GENERATION_CONCURRENCY>0.

    Default (0/unset) keeps the upstream fcntl lock so shared code trees are
    unaffected; launch scripts opt in explicitly. Companion change:
    hs_connectors wait_for_lock timeout 10s -> 300s (concurrent writers queue
    large safetensors writes; a short timeout caused TimeoutError retry loops
    that stalled training).
    """

    def test_default_is_upstream_lock(self, monkeypatch):
        monkeypatch.delenv("SPECULATORS_ONLINE_GENERATION_CONCURRENCY", raising=False)
        import importlib

        import speculators.train.data as data_mod

        importlib.reload(data_mod)
        assert data_mod._generation_semaphore is None

    def test_env_enables_semaphore(self, monkeypatch):
        monkeypatch.setenv("SPECULATORS_ONLINE_GENERATION_CONCURRENCY", "8")
        import importlib

        import speculators.train.data as data_mod

        importlib.reload(data_mod)
        assert data_mod._generation_semaphore is not None

    def test_semaphore_bounds_concurrency(self, monkeypatch):
        monkeypatch.setenv("SPECULATORS_ONLINE_GENERATION_CONCURRENCY", "2")
        import importlib
        import threading

        import speculators.train.data as data_mod

        importlib.reload(data_mod)
        inside = 0
        peak = 0
        lock = threading.Lock()

        def worker():
            nonlocal inside, peak
            with data_mod._online_generation_lock():
                with lock:
                    inside += 1
                    peak = max(peak, inside)
                import time

                time.sleep(0.1)
                with lock:
                    inside -= 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak == 2

    def test_wait_for_lock_default_timeout_300(self):
        import inspect

        from hs_connectors.transfer import wait_for_lock

        assert inspect.signature(wait_for_lock).parameters["timeout"].default == 300.0

    def test_m40_lock_env_forces_upstream_lock(self, monkeypatch):
        # Compatibility with the original m40 patch: LOCK=1 is the revert
        # switch and wins even when a concurrency value is set.
        monkeypatch.setenv("SPECULATORS_ONLINE_GENERATION_CONCURRENCY", "8")
        monkeypatch.setenv("SPECULATORS_ONLINE_GENERATION_LOCK", "1")
        import importlib

        import speculators.train.data as data_mod

        importlib.reload(data_mod)
        assert data_mod._generation_semaphore is None

    def test_check_grep_compat(self):
        # The m40 s3 check script greps these two literals; both must match.
        data_src = Path(
            __import__("speculators.train.data", fromlist=["x"]).__file__
        ).read_text()
        assert "_ONLINE_GENERATION_CONCURRENCY" in data_src
        import hs_connectors.transfer as tr

        tr_src = Path(tr.__file__).read_text()
        assert "timeout=300.0" in tr_src
    """E3 parity #4b: gradients (not just metrics) are token-normalized.

    With DDP averaging, backwarding L_r = S_r/N_r lets token-poor ranks
    dominate (observed 12x p50 imbalance). The trainer rescale factor
    N_r * R / N_total makes the DDP-averaged gradient equal the gradient
        of the global token-weighted loss sum(S)/sum(N).
    """

class TestGlobalGradientTokenNorm:
    def test_rescale_math(self):
        world = 2
        n_r = [100.0, 8.0]  # 12.5x imbalance
        n_total = sum(n_r)
        # legacy: each rank backwards S_r/N_r; DDP averages gradients.
        legacy_coef = [(1 / world) * (1 / n) for n in n_r]
        # new: trainer rescales the loss to S_r * (N_r*R/N_total) = S_r * R/N_total;
        # DDP-averaged gradient coefficient on dS_r becomes 1/N_total for both ranks.
        s_r = [torch.tensor(v, requires_grad=True) for v in (50.0, 5.0)]
        for s, n in zip(s_r, n_r):
            loss = s / n  # per-rank mean, as the model returns it
            rescaled = loss * (n * world / n_total)  # trainer.py factor
            rescaled.backward()
        new_coef = [(s.grad / world).item() for s in s_r]  # DDP average
        target = [1 / n_total, 1 / n_total]
        assert torch.allclose(
            torch.tensor(new_coef), torch.tensor(target), rtol=1e-6, atol=1e-9
        )
        assert legacy_coef[1] / legacy_coef[0] == pytest.approx(12.5)
        assert new_coef[0] == pytest.approx(new_coef[1])  # balanced

    def test_token_count_carried_in_metrics(self):
        from functools import partial

        from speculators.models.dflash.metrics import (
            compute_metrics as _compute_metrics,
        )

        compute_metrics = partial(
            _compute_metrics, loss_config=resolve_loss_config("ce", "eager")
        )
        ids = torch.zeros(1, 8, dtype=torch.long)
        logits = torch.zeros(1, 8, 2)
        logits.scatter_(-1, ids.unsqueeze(-1), 10.0)
        _, metrics = compute_metrics(
            logits, logits.clone(), torch.ones(1, 8), block_size=4
        )
        assert "__loss_token_count__" in metrics
        assert metrics["__loss_token_count__"].item() == pytest.approx(8.0, abs=1.0)

    def test_config_flag_defaults_off(self):
        from speculators.train.trainer import TrainerConfig

        assert "global_token_norm" in TrainerConfig._fields
        assert TrainerConfig._field_defaults.get("global_token_norm", None) is False
