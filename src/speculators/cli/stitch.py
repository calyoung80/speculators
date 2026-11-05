"""
Stitch finetuned MTP weights back into a verifier checkpoint.

Takes a finetuned MTP speculator checkpoint (speculators format) and merges
the trained weights back into the original verifier checkpoint, producing a
self-contained checkpoint directory deployable on vLLM.

Frozen weights (embed_tokens, lm_head) are skipped -- only the MTP layer
weights are replaced.

GLM-MoE-DSA checkpoints are detected from the finetuned config and mapped
back with the GLM inverse layout (per-expert experts,
enorm/hnorm/eh_proj/shared_head.norm).  Both ``model.safetensors.index.json``
and ``quant_model_weights.safetensors.index.json`` verifiers are supported
and the original index filename is preserved.  Quantized verifiers are
protected: float MTP weights are never written over quant metadata siblings,
and every written tensor is cast back to the verifier's per-key dtype (so a
BF16 native round-trip stays bitwise identical).

Usage::

    speculators stitch-mtp ./finetuned-mtp ./Qwen3-Next-80B-A3B-Instruct

    # verifier can be a HuggingFace model ID:
    speculators stitch-mtp ./finetuned-mtp Qwen/Qwen3-Next-80B-A3B-Instruct

    # custom output path (defaults to {verifier-name}-stitched):
    speculators stitch-mtp ./finetuned-mtp ./verifier --output-path ./out
"""

import json
import re
import shutil
from pathlib import Path
from typing import Annotated

import torch
import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from safetensors import safe_open
from safetensors.torch import save_file

from speculators.convert.mtp import MTP_EXACT_REMAP, MTP_PREFIX_REMAP
from speculators.convert.mtp.glm_layout import (
    GLM_MODEL_TYPE,
    GlmMtpPolicy,
    build_glm_layout,
    derive_glm_policy,
    detect_glm_native_prefix,
    verify_glm_layer_index,
    verify_glm_policy,
)
from speculators.convert.mtp.layout import NativeLayout
from speculators.utils.loading import SAFETENSORS_INDEX_NAMES, list_checkpoint_keys

console = Console()

INVERSE_MTP_EXACT_REMAP = {v: k for k, v in MTP_EXACT_REMAP.items()}
INVERSE_MTP_PREFIX_REMAP = [(dst, src) for src, dst in MTP_PREFIX_REMAP]

_FROZEN_KEYS = {"embed_tokens.weight", "lm_head.weight"}

_QUANT_SIBLING_SUFFIXES = (
    "_scale_inv",
    "_scale",
    "_zero_point",
    "_offset",
    "_quant_bias",
)

_FUSED_GATE_UP_PATTERN = re.compile(r"^(.+\.experts)\.gate_up_proj$")
_FUSED_DOWN_PATTERN = re.compile(r"^(.+\.experts)\.down_proj$")


def _spinner() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )


def _bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} shards"),
        TimeElapsedColumn(),
        console=console,
    )


def _remap_key(key: str) -> str:
    """Map a speculators-format weight key back to its native HF name."""
    if key in INVERSE_MTP_EXACT_REMAP:
        return INVERSE_MTP_EXACT_REMAP[key]
    for src, dst in INVERSE_MTP_PREFIX_REMAP:
        if key.startswith(src):
            return dst + key[len(src) :]
    return key


def _inverse_remap_key(key: str, layout: NativeLayout | None) -> str:
    if layout is not None:
        return layout.inverse_remap_key(key)
    return _remap_key(key)


def _read_finetuned_layer_config(checkpoint_dir: Path) -> dict:
    """Read the transformer layer config of a finetuned speculators checkpoint."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        config = json.load(f)
    return config.get("transformer_layer_config") or config


def _resolve_glm_layout(
    finetuned_checkpoint: Path,
    verifier_keys: set[str],
    verifier_path: Path,
) -> tuple[GlmMtpPolicy | None, NativeLayout | None]:
    layer_config = _read_finetuned_layer_config(finetuned_checkpoint)
    if layer_config.get("model_type") != GLM_MODEL_TYPE:
        return None, None
    prefix = detect_glm_native_prefix(verifier_keys)
    # Same wrong-verifier guard as the converter: the native MTP head must
    # sit right after the verifier stack the finetuned model was built from.
    verify_glm_layer_index(prefix, layer_config)
    # Derive the per-asset policy from the verifier package itself (same
    # source of truth as conversion) so export remaps head/embed/rot
    # consistently with the training checkpoint.
    policy = derive_glm_policy(verifier_keys, prefix, verifier_path)
    verify_glm_policy(policy, verifier_keys, prefix)
    return policy, build_glm_layout(prefix, policy)


def _check_quant_siblings(
    native_keys: set[str], present_keys: set[str], *, location: str
) -> None:
    for key in native_keys:
        for suffix in _QUANT_SIBLING_SUFFIXES:
            if key + suffix in present_keys:
                raise NotImplementedError(
                    f"Cannot stitch float MTP weight '{key}' over quantized "
                    f"metadata at {location}: sibling '{key + suffix}' exists. "
                    "Decide the bf16-vs-requant export policy first (plan "
                    "section 19.2)."
                )


def _filter_frozen_keys(
    weights: dict[str, torch.Tensor],
    frozen: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    frozen = _FROZEN_KEYS if frozen is None else frozen
    return {k: v for k, v in weights.items() if k not in frozen}


def _unfuse_moe_experts(  # noqa: C901
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Unfuse packed 3D expert tensors back to per-expert format.

    Inverse of ``MTPConverter._fuse_moe_experts``.
    """
    result: dict[str, torch.Tensor] = {}
    gate_up_keys: dict[str, torch.Tensor] = {}
    down_keys: dict[str, torch.Tensor] = {}

    for key, tensor in weights.items():
        m_gu = _FUSED_GATE_UP_PATTERN.match(key)
        m_d = _FUSED_DOWN_PATTERN.match(key)
        if m_gu is not None:
            gate_up_keys[m_gu.group(1)] = tensor
        elif m_d is not None:
            down_keys[m_d.group(1)] = tensor
        else:
            result[key] = tensor

    if not gate_up_keys:
        if down_keys:
            raise ValueError(
                f"Found down_proj without matching gate_up_proj at: {sorted(down_keys)}"
            )
        return weights

    for prefix, gate_up in gate_up_keys.items():
        if prefix not in down_keys:
            raise ValueError(f"Found gate_up_proj at '{prefix}' but missing down_proj")
        down = down_keys[prefix]

        num_experts = gate_up.shape[0]
        if down.shape[0] != num_experts:
            raise ValueError(
                f"Expert count mismatch at '{prefix}': "
                f"gate_up_proj has {num_experts} but down_proj has {down.shape[0]}"
            )
        half = gate_up.shape[1] // 2

        for i in range(num_experts):
            result[f"{prefix}.{i}.gate_proj.weight"] = gate_up[i, :half].contiguous()
            result[f"{prefix}.{i}.up_proj.weight"] = gate_up[i, half:].contiguous()
            result[f"{prefix}.{i}.down_proj.weight"] = down[i].contiguous()

        console.print(f"  Unfused [cyan]{num_experts}[/] experts at [dim]{prefix}[/]")

    orphaned = set(down_keys) - set(gate_up_keys)
    if orphaned:
        raise ValueError(
            f"Found down_proj without matching gate_up_proj at: {sorted(orphaned)}"
        )

    return result


def _resolve_verifier_path(verifier_path: Path) -> Path:
    """Return a local directory, downloading from HF if needed."""
    if verifier_path.exists():
        return verifier_path

    model_id = str(verifier_path)
    console.print(
        f"Verifier path [cyan]{model_id}[/] not found locally, "
        "downloading from HuggingFace..."
    )
    with _spinner() as progress:
        progress.add_task(f"Downloading {model_id}", total=None)
        local_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.index.json",
            ],
        )
    console.print(f"  Downloaded to [dim]{local_path}[/]")
    return Path(local_path)


def _load_finetuned_weights(
    checkpoint_dir: Path,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        shards = set(weight_map.values())
        with _bar() as progress:
            task = progress.add_task("Loading finetuned weights", total=len(shards))
            for shard in shards:
                with safe_open(str(checkpoint_dir / shard), framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        weights[key] = f.get_tensor(key)
                progress.advance(task)
        return weights

    single = checkpoint_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                weights[key] = f.get_tensor(key)
        return weights

    raise FileNotFoundError(f"No safetensors found at {checkpoint_dir}")


def _stitch_sharded(
    output_dir: Path,
    native_weights: dict[str, torch.Tensor],
    index_name: str,
) -> None:
    index_path = output_dir / index_name
    with index_path.open() as f:
        index_data = json.load(f)
    weight_map: dict[str, str] = index_data["weight_map"]

    missing = sorted(set(native_weights) - set(weight_map))
    if missing:
        raise ValueError(
            "Finetuned keys not found in the verifier weight map: "
            f"{missing}. The finetuned checkpoint may not match the verifier."
        )
    _check_quant_siblings(
        set(native_weights), set(weight_map), location=str(index_path)
    )

    shard_to_new: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in native_weights.items():
        shard = weight_map[key]
        shard_to_new.setdefault(shard, {})[key] = tensor

    with _bar() as progress:
        task = progress.add_task("Stitching shards", total=len(shard_to_new))
        for shard_filename, new_weights in shard_to_new.items():
            shard_path = output_dir / shard_filename
            existing: dict[str, torch.Tensor] = {}
            metadata = None
            with safe_open(str(shard_path), framework="pt") as f:
                metadata = f.metadata()
                for k in f.keys():  # noqa: SIM118
                    existing[k] = f.get_tensor(k)
            for key, tensor in new_weights.items():
                if key in existing:
                    if existing[key].shape != tensor.shape:
                        raise ValueError(
                            f"Shape mismatch for '{key}': verifier has "
                            f"{existing[key].shape} but finetuned has "
                            f"{tensor.shape}"
                        )
                    # Export in the verifier's per-key dtype (bf16 native
                    # weights, fp32 router buffers, ...).
                    new_weights[key] = tensor.to(existing[key].dtype)
            existing.update(new_weights)
            save_file(existing, str(shard_path), metadata=metadata)
            progress.advance(task)


def _stitch_single(
    safetensors_path: Path,
    native_weights: dict[str, torch.Tensor],
) -> None:
    existing: dict[str, torch.Tensor] = {}
    metadata = None
    with safe_open(str(safetensors_path), framework="pt") as f:
        metadata = f.metadata()
        present = set(f.keys())
        _check_quant_siblings(
            set(native_weights), present, location=str(safetensors_path)
        )
        for k in present:
            existing[k] = f.get_tensor(k)
    for key in list(native_weights):
        tensor = native_weights[key]
        if key in existing:
            if existing[key].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': verifier has "
                    f"{existing[key].shape} but finetuned has {tensor.shape}"
                )
            native_weights[key] = tensor.to(existing[key].dtype)
    existing.update(native_weights)
    save_file(existing, str(safetensors_path), metadata=metadata)


def stitch(
    finetuned_checkpoint: Path,
    verifier_path: Path,
    output_path: Path,
) -> Path:
    """Stitch finetuned MTP weights back into a verifier checkpoint."""
    console.print(
        Panel(
            f"[bold]Finetuned:[/] {finetuned_checkpoint}\n"
            f"[bold]Verifier:[/]  {verifier_path}\n"
            f"[bold]Output:[/]    {output_path}",
            title="[bold green]MTP Stitch[/]",
            border_style="green",
        )
    )

    verifier_path = _resolve_verifier_path(verifier_path)
    verifier_keys = set(list_checkpoint_keys(verifier_path))
    glm_policy, layout = _resolve_glm_layout(
        finetuned_checkpoint, verifier_keys, verifier_path
    )

    weights = _load_finetuned_weights(finetuned_checkpoint)
    console.print(f"  Loaded [cyan]{len(weights)}[/] finetuned weight tensors")

    # Own-head/embed checkpoints carry these weights trained values; they are
    # written back into the native package instead of being skipped.
    frozen = set(_FROZEN_KEYS)
    if glm_policy is not None:
        if glm_policy.own_mtp_head:
            frozen.discard("lm_head.weight")
        if glm_policy.own_mtp_embed:
            frozen.discard("embed_tokens.weight")
    frozen_count = sum(1 for k in weights if k in frozen)
    weights = _filter_frozen_keys(weights, frozen)
    if frozen_count:
        console.print(
            f"  Skipped [yellow]{frozen_count}[/] frozen key(s) (embed_tokens, lm_head)"
        )

    weights = _unfuse_moe_experts(weights)
    native_weights = {_inverse_remap_key(k, layout): v for k, v in weights.items()}
    # Training-side verifier reference tensors (verifier_lm_head.weight, ...)
    # travel with speculators checkpoints for the training loss but are not
    # part of the native MTP layer; stitch never writes them back.
    reference_keys = sorted(k for k in native_weights if k.startswith("verifier_"))
    if reference_keys:
        console.print(
            f"  Skipped [yellow]{len(reference_keys)}[/] verifier reference "
            f"key(s): {reference_keys[:3]}"
        )
        native_weights = {
            k: v for k, v in native_weights.items() if not k.startswith("verifier_")
        }
    console.print(f"  Remapped [cyan]{len(native_weights)}[/] keys to native format")

    # Fail before copying the verifier so a rejected run leaves no output.
    _check_quant_siblings(
        set(native_weights), verifier_keys, location=str(verifier_path)
    )

    if output_path.exists():
        console.print(
            f"[yellow]Warning:[/] Output directory [dim]{output_path}[/] already "
            "exists; stale files from a previous run may remain."
        )
    with _spinner() as progress:
        progress.add_task("Copying verifier checkpoint", total=None)
        shutil.copytree(verifier_path, output_path, dirs_exist_ok=True)

    index_path = next(
        (
            output_path / name
            for name in SAFETENSORS_INDEX_NAMES
            if (output_path / name).exists()
        ),
        None,
    )
    if index_path is not None:
        _stitch_sharded(output_path, native_weights, index_path.name)
    else:
        single = output_path / "model.safetensors"
        if single.exists():
            _stitch_single(single, native_weights)
        else:
            raise FileNotFoundError(f"No safetensors checkpoint found at {output_path}")

    console.print(
        Panel(
            f"[bold]{output_path}[/]",
            title="[bold green]Stitched checkpoint saved[/]",
            border_style="green",
        )
    )
    return output_path


def stitch_command(
    finetuned_checkpoint: Annotated[
        Path,
        typer.Argument(
            help="Path to the finetuned MTP speculator checkpoint.",
        ),
    ],
    verifier_path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Path to the verifier checkpoint, or a HuggingFace "
                "model ID (e.g. Qwen/Qwen3-Next-80B-A3B-Instruct)."
            ),
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Output directory for the stitched checkpoint. "
                "Defaults to {verifier-name}-stitched."
            ),
        ),
    ] = None,
) -> None:
    """Stitch finetuned MTP weights back into a verifier checkpoint."""
    if output_path is None:
        output_path = Path.cwd() / f"{verifier_path.name}-stitched"

    stitch(
        finetuned_checkpoint=finetuned_checkpoint,
        verifier_path=verifier_path,
        output_path=output_path,
    )
