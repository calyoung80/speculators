from speculators.train.data import BatchType

__all__ = ["shift_batch_mtp"]


def shift_batch_mtp(batch: BatchType) -> BatchType:
    """Rename verifier_last_hidden_states to hidden_states for MTP.

    No token-level shifting — the MTP forward pass handles alignment
    internally via per-step offset slicing of input_ids.

    ``lengths`` is optional in the NO_PACK main path.  Everything else the
    model does not consume (``document_ids``, capture-stage markers, ...) is
    dropped: the packed collate regenerates ``document_ids`` from ``lengths``,
    and the capture stage is a run-level fact that the trainer must attach to
    the collated batch, not a per-sample tensor.
    """
    shifted: BatchType = {
        "input_ids": batch["input_ids"],
        "hidden_states": batch["verifier_last_hidden_states"],
        "loss_mask": batch["loss_mask"],
        "position_ids": batch["position_ids"],
    }
    if "lengths" in batch:
        shifted["lengths"] = batch["lengths"]
    return shifted
