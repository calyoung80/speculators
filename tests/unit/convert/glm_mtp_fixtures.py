"""Synthetic GLM-MoE-DSA native MTP checkpoints for converter/stitch tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "glm_mtp"
TINY_CONFIG: dict = json.loads((FIXTURE_DIR / "tiny_config.json").read_text())
NUM_LAYERS: int = TINY_CONFIG["num_hidden_layers"]
MTP_PREFIX = f"model.layers.{NUM_LAYERS}."

HIDDEN = TINY_CONFIG["hidden_size"]
VOCAB = TINY_CONFIG["vocab_size"]
EXPERTS = TINY_CONFIG["n_routed_experts"]
INTER = TINY_CONFIG["moe_intermediate_size"]
Q_LORA = TINY_CONFIG["q_lora_rank"]
KV_LORA = TINY_CONFIG["kv_lora_rank"]
ROPE = TINY_CONFIG["qk_rope_head_dim"]
NOPE = TINY_CONFIG["qk_nope_head_dim"]
V_HEAD = TINY_CONFIG["v_head_dim"]
HEADS = TINY_CONFIG["num_attention_heads"]
QK = NOPE + ROPE
INDEX_DIM = TINY_CONFIG["index_head_dim"]
INDEX_HEADS = TINY_CONFIG["index_n_heads"]


def _rand(shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(dtype)


def build_native_mtp(
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    *,
    include_head_embed: bool = True,
    include_quant_scale: bool = False,
    include_rot: bool = False,
) -> dict[str, torch.Tensor]:
    """Build a per-expert native MTP head under ``model.layers.<N>.``."""
    weights: dict[str, torch.Tensor] = {}

    def add(name: str, shape: tuple[int, ...]) -> None:
        weights[MTP_PREFIX + name] = _rand(shape, seed + len(weights), dtype)

    add("enorm.weight", (HIDDEN,))
    add("hnorm.weight", (HIDDEN,))
    add("eh_proj.weight", (HIDDEN, 2 * HIDDEN))
    add("shared_head.norm.weight", (HIDDEN,))
    add("input_layernorm.weight", (HIDDEN,))
    add("post_attention_layernorm.weight", (HIDDEN,))

    add("self_attn.q_a_proj.weight", (Q_LORA, HIDDEN))
    add("self_attn.q_a_layernorm.weight", (Q_LORA,))
    add("self_attn.q_b_proj.weight", (HEADS * QK, Q_LORA))
    add("self_attn.kv_a_proj_with_mqa.weight", (KV_LORA + ROPE, HIDDEN))
    add("self_attn.kv_a_layernorm.weight", (KV_LORA,))
    add("self_attn.kv_b_proj.weight", (HEADS * (NOPE + V_HEAD), KV_LORA))
    add("self_attn.o_proj.weight", (HIDDEN, HEADS * V_HEAD))

    add("self_attn.indexer.wq_b.weight", (INDEX_HEADS * INDEX_DIM, Q_LORA))
    add("self_attn.indexer.wk.weight", (INDEX_DIM, HIDDEN))
    add("self_attn.indexer.k_norm.weight", (INDEX_DIM,))
    add("self_attn.indexer.k_norm.bias", (INDEX_DIM,))
    add("self_attn.indexer.weights_proj.weight", (INDEX_HEADS, HIDDEN))

    add("mlp.gate.weight", (EXPERTS, HIDDEN))
    # The router buffer is fp32 even in bf16 checkpoints.
    weights[MTP_PREFIX + "mlp.gate.e_score_correction_bias"] = _rand(
        (EXPERTS,), seed + len(weights), torch.float32
    )
    add("mlp.shared_experts.gate_proj.weight", (INTER, HIDDEN))
    add("mlp.shared_experts.up_proj.weight", (INTER, HIDDEN))
    add("mlp.shared_experts.down_proj.weight", (HIDDEN, INTER))
    for expert in range(EXPERTS):
        add(f"mlp.experts.{expert}.gate_proj.weight", (INTER, HIDDEN))
        add(f"mlp.experts.{expert}.up_proj.weight", (INTER, HIDDEN))
        add(f"mlp.experts.{expert}.down_proj.weight", (HIDDEN, INTER))

    if include_head_embed:
        add("shared_head.head.weight", (VOCAB, HIDDEN))
        add("embed_tokens.weight", (VOCAB, HIDDEN))
    if include_rot:
        add("rot.weight", (HIDDEN, HIDDEN))
    if include_quant_scale:
        weights[MTP_PREFIX + "mlp.experts.0.gate_proj.weight_scale_inv"] = _rand(
            (1,), seed + 1000, torch.float32
        )
    return weights


def build_non_mtp_weights(seed: int = 500) -> dict[str, torch.Tensor]:
    """Non-MTP tensors that stitch must leave byte-identical."""
    return {
        "model.embed_tokens.weight": _rand((VOCAB, HIDDEN), seed, torch.float32),
        "lm_head.weight": _rand((VOCAB, HIDDEN), seed + 1, torch.float32),
        "model.norm.weight": _rand((HIDDEN,), seed + 2, torch.float32),
        "model.layers.0.mlp.gate_proj.weight": _rand(
            (INTER, HIDDEN), seed + 3, torch.float32
        ),
    }


def write_checkpoint(
    directory: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    index_name: str | None = None,
    shards: int = 1,
    config: dict | None = None,
) -> Path:
    """Write a single-file or sharded safetensors checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(config if config is not None else TINY_CONFIG)
    )

    if shards <= 1:
        save_file(state_dict, str(directory / "model.safetensors"))
        return directory

    keys = sorted(state_dict)
    chunk_size = (len(keys) + shards - 1) // shards
    weight_map: dict[str, str] = {}
    for chunk_index in range(shards):
        chunk = keys[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
        if not chunk:
            continue
        filename = f"model-{chunk_index + 1:05d}-of-{shards:05d}.safetensors"
        save_file(
            {key: state_dict[key] for key in chunk},
            str(directory / filename),
        )
        for key in chunk:
            weight_map[key] = filename

    index = {"metadata": {}, "weight_map": weight_map}
    (directory / (index_name or "model.safetensors.index.json")).write_text(
        json.dumps(index, indent=2)
    )
    return directory


def expected_fused_experts(
    native: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (gate_up_proj, down_proj) as the converter must fuse them."""
    gate = torch.stack(
        [
            native[f"{MTP_PREFIX}mlp.experts.{i}.gate_proj.weight"]
            for i in range(EXPERTS)
        ]
    )
    up = torch.stack(
        [native[f"{MTP_PREFIX}mlp.experts.{i}.up_proj.weight"] for i in range(EXPERTS)]
    )
    down = torch.stack(
        [
            native[f"{MTP_PREFIX}mlp.experts.{i}.down_proj.weight"]
            for i in range(EXPERTS)
        ]
    )
    return torch.cat([gate, up], dim=1), down
