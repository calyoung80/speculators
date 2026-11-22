"""Tests for the opt-in page-cache / read-only-view hidden-state optimizations."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from hs_connectors.transfer import _load_hs_file
from safetensors.torch import save_file

from hs_connectors import page_cache_drop, readonly_views


def _make_file(path: Path, size: int = 1024) -> Path:
    path.write_bytes(b"x" * size)
    return path


def _assert_mapped_read_only(path: Path) -> None:
    suffix = f" {path}\n"
    with Path("/proc/self/maps").open(encoding="utf-8") as maps:
        matching = [line for line in maps if line.endswith(suffix)]
    assert matching, f"{path} is not mmap'd"
    assert all(line.split()[1].startswith("r-") for line in matching)


def _sample() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "hidden_states_bf16": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "token_ids": torch.arange(3, dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
        "scalar": torch.tensor(3.5, dtype=torch.float32),
    }


def test_fadvise_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HS_FADVISE_DONTNEED", raising=False)
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda *args: calls.append(args), raising=False
    )

    assert page_cache_drop.drop_page_cache(_make_file(tmp_path / "hs.bin")) is False
    assert calls == []


def test_fadvise_issues_dontneed(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_FADVISE_DONTNEED", "1")
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda *args: calls.append(args), raising=False
    )

    target = _make_file(tmp_path / "hs.bin")
    assert page_cache_drop.drop_page_cache(target, min_bytes=0) is True
    assert len(calls) == 1
    assert calls[0][1:] == (0, 0, os.POSIX_FADV_DONTNEED)


def test_fadvise_swallows_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_FADVISE_DONTNEED", "1")

    def boom(*_args):
        raise OSError("injected")

    monkeypatch.setattr(os, "posix_fadvise", boom, raising=False)

    assert (
        page_cache_drop.drop_page_cache(_make_file(tmp_path / "hs.bin"), min_bytes=0)
        is False
    )


def test_fadvise_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_FADVISE_DONTNEED", "1")

    assert page_cache_drop.drop_page_cache(tmp_path / "missing.bin") is False


def test_fadvise_skips_small_files(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_FADVISE_DONTNEED", "1")
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda *args: calls.append(args), raising=False
    )

    assert page_cache_drop.drop_page_cache(_make_file(tmp_path / "small.bin")) is False
    assert calls == []


def test_fadvise_min_bytes_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_FADVISE_DONTNEED", "1")
    monkeypatch.setenv("HS_FADVISE_MIN_BYTES", "1")
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda *args: calls.append(args), raising=False
    )

    assert page_cache_drop.drop_page_cache(_make_file(tmp_path / "small.bin")) is True
    assert len(calls) == 1


def test_readonly_views_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HS_READONLY_VIEWS", raising=False)
    path = tmp_path / "sample.safetensors"
    save_file(_sample(), path)

    assert readonly_views.load_file_views(path) is None


def test_readonly_views_return_equal_tensors(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_READONLY_VIEWS", "1")
    expected = _sample()
    path = tmp_path / "sample.safetensors"
    save_file(expected, path)

    actual = readonly_views.load_file_views(path)

    assert actual is not None
    assert set(actual) == set(expected)
    for key, tensor in expected.items():
        assert actual[key].dtype == tensor.dtype
        assert actual[key].shape == tensor.shape
        torch.testing.assert_close(actual[key], tensor)
    _assert_mapped_read_only(path)


def test_readonly_views_return_none_for_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_READONLY_VIEWS", "1")
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(b"\xff" * 16 + b"\x00" * 64)

    assert readonly_views.load_file_views(path) is None


def test_readonly_views_release_mapping_and_drop_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_READONLY_VIEWS", "1")
    monkeypatch.setenv("HS_FADVISE_DONTNEED", "1")
    monkeypatch.setenv("HS_FADVISE_MIN_BYTES", "0")
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda *args: calls.append(args), raising=False
    )
    path = tmp_path / "sample.safetensors"
    save_file(_sample(), path)

    sample = readonly_views.load_file_views(path)
    assert sample is not None
    _assert_mapped_read_only(path)

    del sample
    gc.collect()

    with Path("/proc/self/maps").open(encoding="utf-8") as maps:
        assert not any(line.endswith(f" {path}\n") for line in maps)
    assert len(calls) == 1


def test_transfer_uses_views_and_drops_page_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_READONLY_VIEWS", "1")
    monkeypatch.setenv("HS_FADVISE_DONTNEED", "1")
    monkeypatch.setenv("HS_FADVISE_MIN_BYTES", "0")
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda *args: calls.append(args), raising=False
    )
    expected = _sample()
    path = tmp_path / "sample.safetensors"
    save_file(expected, path)

    actual = _load_hs_file(path)

    assert actual is not None
    for key, tensor in expected.items():
        torch.testing.assert_close(actual[key], tensor)
    assert len(calls) == 1


def test_transfer_falls_back_when_views_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("HS_READONLY_VIEWS", raising=False)
    monkeypatch.delenv("HS_FADVISE_DONTNEED", raising=False)
    expected = _sample()
    path = tmp_path / "sample.safetensors"
    save_file(expected, path)

    actual = _load_hs_file(path)

    assert actual is not None
    for key, tensor in expected.items():
        torch.testing.assert_close(actual[key], tensor)


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX")
def test_fadvise_supported_on_platform():
    assert page_cache_drop._FADV_DONTNEED is not None
