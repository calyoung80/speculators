import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
):
    """
    Build a flex-attention mask mod where each query block corresponds to one anchor.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries in block j:
        - may attend to base tokens in the same document with
          position < anchor_positions[j]
        - may attend to all tokens in their own synthetic block j
        - may not attend to other synthetic blocks or later base tokens

    Args:
        document_ids: [total_seq_len] maps each position to its doc index, pad -1
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block
        sliding_window: integer size of sliding window or None for full attn
        sliding_window_non_causal: Use non causal mask for sliding window attn

    Returns:
        mask_mod, q_len, kv_len
    """
    # Always use non_causal for full attn
    non_causal = sliding_window is None or sliding_window_non_causal

    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    # For each query position, which anchor does it belong to?
    # query q in [j*block_size, (j+1)*block_size) belongs to anchor_positions[j]
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may see base-sequence tokens in the same document before the anchor.
        """
        # absolute base position
        q_anchor = query_anchor_positions[q_idx]
        # doc id for this query block
        q_doc = document_ids[q_anchor]

        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)  # safe indexing
        kv_doc = document_ids[kv_base_pos]

        same_doc = (q_doc == kv_doc) & (q_doc != -1)
        before_anchor = kv_base_pos < q_anchor

        in_window = (
            (kv_base_pos >= q_anchor - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def same_block_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may attend to tokens in their own synthetic block.
        Non-causal unless non_causal=False,
        in which case only prior positions are attended.
        """
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        same = kv_is_block & (q_block == kv_block)
        if not non_causal:
            same = same & (kv_idx <= q_idx + total_seq_len)
        return same

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len


def build_anchor_block_float_mask(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build the anchor-block additive float mask with plain broadcasting.

    Semantically identical to ``create_float_mask`` applied to the mask mod
    from :func:`create_anchor_block_mask_mod`, but implemented without
    ``flex_attention.create_mask``: that entry point routes the mask mod
    through ``_vmap_for_bhqkv`` (vmap/compile chain), which has no working
    backend on Ascend NPU (runtime error 207001). This function only uses
    tensor comparisons and broadcasts, so it is safe on any device.

    Layout matches the flex path exactly:
        mask[0, 0, q, kv] with KV = [base(total_seq_len) | synthetic blocks]

    Args:
        document_ids: [total_seq_len] doc index per base position, pad -1.
        total_seq_len: padded packed sequence width.
        anchor_positions: [n_anchors] absolute base positions, one per block.
        block_size: query tokens per anchor block.
        sliding_window: sliding window size or None for full attention.
        sliding_window_non_causal: non-causal mask for sliding-window attention.
        dtype: output dtype (0 / -inf additive mask).

    Returns:
        Additive float mask of shape [1, 1, n_anchors*block_size,
        total_seq_len + n_anchors*block_size].
    """
    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()
    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    non_causal = sliding_window is None or sliding_window_non_causal

    # --- Q-side index tensors [q_len] ---
    q_idx = torch.arange(q_len, device=device)
    q_block = q_idx // block_size  # anchor index per query
    q_offset = q_idx % block_size  # position inside the block
    q_anchor = anchor_positions[q_block]  # anchor base position per query
    q_doc = document_ids[q_anchor]  # doc id per query

    # --- KV-side index tensors [kv_len] ---
    kv_idx = torch.arange(kv_len, device=device)
    kv_is_base = kv_idx < total_seq_len
    kv_base_pos = torch.where(
        kv_is_base, kv_idx, torch.zeros_like(kv_idx)
    )  # safe index for base lookups; synthetic entries masked out later
    kv_doc = document_ids[kv_base_pos]
    kv_block = torch.where(
        kv_is_base,
        torch.full_like(kv_idx, -1),  # base tokens never share a synthetic block
        (kv_idx - total_seq_len) // block_size,
    )

    # --- base-prefix visibility [q_len, total_seq_len] ---
    same_doc = (q_doc.unsqueeze(1) == kv_doc[:total_seq_len].unsqueeze(0)) & (
        q_doc.unsqueeze(1) != -1
    )
    before_anchor = (
        torch.arange(total_seq_len, device=device).unsqueeze(0) < q_anchor.unsqueeze(1)
    )
    base_visible = same_doc & before_anchor
    if sliding_window is not None:
        in_window = (
            torch.arange(total_seq_len, device=device).unsqueeze(0)
            >= (q_anchor - sliding_window).unsqueeze(1)
        )
        base_visible = base_visible & in_window

    # --- same-synthetic-block visibility [q_len, q_len] ---
    same_block = q_block.unsqueeze(1) == kv_block[total_seq_len:].unsqueeze(0)
    if not non_causal:
        # Flex semantics: kv_idx <= q_idx + total_seq_len, i.e. the
        # synthetic-side *global* offset (block*bs + offset) must not exceed
        # the query's *global* index. Comparing block-local offsets instead
        # would wrongly mask the diagonal.
        kv_syn_global = torch.arange(q_len, device=device)
        same_block = same_block & (kv_syn_global.unsqueeze(0) <= q_idx.unsqueeze(1))

    # --- combine into [q_len, kv_len] ---
    visible = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
    visible[:, :total_seq_len] = base_visible
    visible[:, total_seq_len:] = same_block

    float_mask = torch.zeros(
        (1, 1, q_len, kv_len), dtype=dtype, device=device
    )
    float_mask.masked_fill_(~visible, float("-inf"))
    return float_mask
