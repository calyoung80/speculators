"""Unit tests for MTP data preprocessing."""

import torch

from speculators.models.mtp.data import shift_batch_mtp


def _batch(**overrides):
    seq_len, hidden_size = 32, 128
    batch = {
        "input_ids": torch.randint(0, 1000, (seq_len,)),
        "verifier_last_hidden_states": torch.randn(seq_len, hidden_size),
        "loss_mask": torch.ones(seq_len),
        "lengths": torch.tensor([seq_len]),
        "position_ids": torch.arange(seq_len),
    }
    batch.update(overrides)
    return batch


def test_shift_batch_mtp():
    batch = _batch()

    result = shift_batch_mtp(batch)

    assert "verifier_last_hidden_states" not in result
    assert torch.equal(result["hidden_states"], batch["verifier_last_hidden_states"])
    assert result["hidden_states"].shape == (32, 128)

    for key in ("input_ids", "loss_mask", "lengths", "position_ids"):
        assert torch.equal(result[key], batch[key])


def test_shift_batch_mtp_lengths_optional_for_no_pack():
    batch = _batch()
    del batch["lengths"]

    result = shift_batch_mtp(batch)

    assert "lengths" not in result
    assert torch.equal(result["position_ids"], batch["position_ids"])


def test_shift_batch_mtp_drops_non_tensor_metadata():
    document_ids = torch.tensor([0] * 16 + [1] * 16)
    batch = _batch(document_ids=document_ids, hidden_state_stage="pre_norm")

    result = shift_batch_mtp(batch)

    # The packed collate regenerates document_ids from lengths, and the
    # capture stage is a run-level fact attached after collation -- carrying
    # either per-sample would corrupt (or crash) the dataloader.
    assert "document_ids" not in result
    assert "hidden_state_stage" not in result


def test_shift_batch_mtp_does_not_invent_optional_keys():
    result = shift_batch_mtp(_batch())

    assert "document_ids" not in result
    assert "hidden_state_stage" not in result
