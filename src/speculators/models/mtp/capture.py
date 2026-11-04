"""Teacher hidden-state capture manifests for the GLM MTP pipeline.

The teacher capture contract is frozen in the GLM MTP plan (sections 10,
17.11 and 21.6).  This module keeps the CPU-checkable half of it:

* validate a capture manifest (layer semantics, stage, file hashes);
* prove that the training side applies the frozen verifier norm exactly
  once (``pre_norm``) or not at all (``post_norm``);
* compare two captures with the bitwise-first determinism gate.

Nothing here imports vLLM or touches an accelerator, so the rules can be
tested before the real teacher is available.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "CAPTURE_STAGES",
    "CaptureManifest",
    "compare_capture_files",
    "compare_capture_tensors",
    "reference_post_norm",
    "sha256_file",
]

CAPTURE_STAGES = ("pre_norm", "post_norm")
_CLOSE_TOLERANCE = 1e-12

_REQUIRED_FIELDS = (
    "hidden_state_stage",
    "verifier_source_layer_ordinal",
    "verifier_source_layer_index_zero_based",
    "verifier_source_tensor_semantics",
    "capture_tensor_semantics",
    "vllm_target_layer_id",
    "native_mtp_layer_index",
    "native_mtp_key_prefix",
    "native_mtp_tensor_semantics",
    "source_norm",
    "hidden_size",
    "dtype",
    "input_bytes_sha256",
    "position_ids_sha256",
    "capture_files",
)

_PREFIX_PATTERN = re.compile(r"^model\.layers\.(\d+)\.$")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_post_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference ``model.norm`` output for a captured pre-norm tensor.

    Mirrors ``GlmMoeDsaRMSNorm``: fp32 reduction, then the gain multiplies the
    tensor after casting back to the input dtype.
    """
    input_dtype = hidden_states.dtype
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    normalized = hidden_fp32 * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


class CaptureManifest:
    """Validated view over one ``capture_manifest.json``.

    :param data: parsed manifest mapping.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = dict(data)
        self.base_dir: Path | None = None
        self.validate()

    @classmethod
    def from_path(cls, path: str | Path) -> CaptureManifest:
        """Load and validate a manifest from disk.

        Relative ``capture_files`` paths are later resolved against the
        manifest's own directory.
        """
        manifest_path = Path(path)
        with manifest_path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Capture manifest {path} must contain a JSON object.")
        manifest = cls(data)
        manifest.base_dir = manifest_path.resolve().parent
        return manifest

    @property
    def hidden_state_stage(self) -> str:
        return self.data["hidden_state_stage"]

    @property
    def uses_verifier_norm(self) -> bool:
        """Whether training must apply the frozen verifier norm."""
        return self.hidden_state_stage == "pre_norm"

    def validate(  # noqa: C901
        self,
        *,
        config: Any | None = None,
        checkpoint_keys: Sequence[str] | None = None,
        verify_files: bool = False,
    ) -> None:
        """Validate the manifest, optionally against a model config/checkpoint."""
        missing = [name for name in _REQUIRED_FIELDS if name not in self.data]
        if missing:
            raise ValueError(f"Capture manifest is missing required fields: {missing}")

        stage = self.data["hidden_state_stage"]
        if stage not in CAPTURE_STAGES:
            raise ValueError(
                f"Unknown hidden_state_stage {stage!r}; expected one of "
                f"{CAPTURE_STAGES}."
            )

        ordinal = self.data["verifier_source_layer_ordinal"]
        zero_based = self.data["verifier_source_layer_index_zero_based"]
        if not isinstance(ordinal, int) or ordinal < 1:
            raise ValueError(
                "verifier_source_layer_ordinal must be a positive integer, got "
                f"{ordinal!r}."
            )
        if zero_based != ordinal - 1:
            raise ValueError(
                "verifier_source_layer_index_zero_based must equal "
                "verifier_source_layer_ordinal - 1 "
                f"({zero_based} != {ordinal - 1})."
            )

        prefix = self.data["native_mtp_key_prefix"]
        match = _PREFIX_PATTERN.match(prefix)
        if match is None:
            raise ValueError(
                "native_mtp_key_prefix must look like 'model.layers.<N>.', got "
                f"{prefix!r}."
            )
        if int(match.group(1)) != self.data["native_mtp_layer_index"]:
            raise ValueError(
                "native_mtp_key_prefix and native_mtp_layer_index disagree: "
                f"{prefix!r} vs {self.data['native_mtp_layer_index']}."
            )
        if self.data["native_mtp_tensor_semantics"] != "native_mtp_head":
            raise ValueError(
                "native_mtp_tensor_semantics must be 'native_mtp_head', got "
                f"{self.data['native_mtp_tensor_semantics']!r}."
            )

        self._validate_stage_semantics(stage)

        hidden_size = self.data["hidden_size"]
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive int, got {hidden_size!r}."
            )

        files = self.data["capture_files"]
        if not isinstance(files, list) or not files:
            raise ValueError("capture_files must be a non-empty list.")
        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"capture_files entry must be an object, got {entry!r}."
                )
            if not entry.get("path") or not entry.get("sha256"):
                raise ValueError(
                    f"capture_files entries need both 'path' and 'sha256': {entry!r}."
                )
            if verify_files:
                path = self._resolve_capture_path(entry["path"])
                if not path.exists():
                    raise FileNotFoundError(f"Capture file not found: {path}")
                if sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"Capture file hash mismatch: {path}")

        model_norm = self.data.get("model_norm")
        if stage == "post_norm":
            if (
                not isinstance(model_norm, Mapping)
                or "eps" not in model_norm
                or "weight_sha256" not in model_norm
            ):
                raise ValueError(
                    "post_norm captures must record the source model.norm "
                    "epsilon and weight hash under 'model_norm' "
                    "('eps', 'weight_sha256'); the post-norm parity gate "
                    "extracts and hashes model.norm.weight (plan section 10.3)."
                )
        elif not isinstance(model_norm, Mapping) or "weight_sha256" not in model_norm:
            raise ValueError(
                "pre_norm captures must record the frozen model.norm "
                "weight hash under 'model_norm.weight_sha256'."
            )

        self._validate_indexer_type()
        self._validate_against_config(config)
        self._validate_against_keys(checkpoint_keys)

    def training_hidden_states(
        self,
        hidden_states: torch.Tensor,
        verifier_norm: Any,
    ) -> torch.Tensor:
        """Apply the stage rule: norm pre-norm captures exactly once.

        ``post_norm`` captures are returned unchanged, so a manifest cannot
        induce a double normalization.  ``verifier_norm`` must be the frozen
        verifier final norm (``model.norm``).
        """
        if self.uses_verifier_norm:
            return verifier_norm(hidden_states)
        return hidden_states

    def _resolve_capture_path(self, raw_path: str) -> Path:
        """Resolve a capture file path against the manifest directory.

        Absolute paths are used as-is; a manifest built from a dict (no
        ``base_dir``) resolves relative paths against the working directory.
        """
        path = Path(raw_path)
        if path.is_absolute() or self.base_dir is None:
            return path
        return self.base_dir / path

    def _validate_stage_semantics(self, stage: str) -> None:
        source_semantics = self.data["verifier_source_tensor_semantics"]
        if source_semantics != "decoder_output_before_model_norm":
            raise ValueError(
                "verifier_source_tensor_semantics must be "
                "'decoder_output_before_model_norm', got "
                f"{source_semantics!r}."
            )
        if stage == "post_norm":
            if self.data["capture_tensor_semantics"] != "model_norm_output":
                raise ValueError(
                    "post_norm captures need capture_tensor_semantics="
                    "'model_norm_output', got "
                    f"{self.data['capture_tensor_semantics']!r}."
                )
            if self.data["source_norm"] != "model.norm":
                raise ValueError(
                    "post_norm captures need source_norm='model.norm', got "
                    f"{self.data['source_norm']!r}."
                )
        elif self.data["capture_tensor_semantics"] != (
            "decoder_output_before_model_norm"
        ):
            raise ValueError(
                "pre_norm captures keep capture_tensor_semantics="
                "'decoder_output_before_model_norm', got "
                f"{self.data['capture_tensor_semantics']!r}."
            )

    def _validate_indexer_type(self) -> None:
        indexer_type = self.data.get("verifier_source_indexer_type")
        if indexer_type is not None and indexer_type not in ("full", "shared"):
            raise ValueError(
                "verifier_source_indexer_type must be 'full' or 'shared', got "
                f"{indexer_type!r}."
            )

    def _validate_against_config(  # noqa: C901
        self, config: Any | None
    ) -> None:
        if config is None:
            return

        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is not None:
            ordinal = self.data["verifier_source_layer_ordinal"]
            if ordinal != num_layers:
                raise ValueError(
                    "Capture layer does not match the verifier depth: "
                    f"verifier_source_layer_ordinal={ordinal}, "
                    f"num_hidden_layers={num_layers}."
                )
            if self.data["native_mtp_layer_index"] != num_layers:
                raise ValueError(
                    "native MTP layer index must equal num_hidden_layers: "
                    f"{self.data['native_mtp_layer_index']} != {num_layers}."
                )

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None and self.data["hidden_size"] != hidden_size:
            raise ValueError(
                f"Capture hidden_size={self.data['hidden_size']} does not match "
                f"config hidden_size={hidden_size}."
            )

        eps = getattr(config, "rms_norm_eps", None)
        model_norm = self.data.get("model_norm")
        if (
            eps is not None
            and isinstance(model_norm, Mapping)
            and "eps" in model_norm
            and not _close(model_norm["eps"], eps)
        ):
            raise ValueError(
                f"model_norm.eps={model_norm['eps']} does not match "
                f"config.rms_norm_eps={eps}."
            )

        indexer_type = self.data.get("verifier_source_indexer_type")
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_type is not None and indexer_types:
            source_index = self.data["verifier_source_layer_index_zero_based"]
            if source_index >= len(indexer_types):
                raise ValueError(
                    "verifier_source_layer_index_zero_based is outside "
                    f"config.indexer_types ({source_index} >= {len(indexer_types)})."
                )
            if indexer_types[source_index] != indexer_type:
                raise ValueError(
                    "Capture indexer type does not match the verifier config: "
                    f"{indexer_type!r} != {indexer_types[source_index]!r}."
                )

        index_share = self.data.get("index_share_for_mtp_iteration")
        config_index_share = getattr(config, "index_share_for_mtp_iteration", None)
        if (
            index_share is not None
            and config_index_share is not None
            and bool(index_share) != bool(config_index_share)
        ):
            raise ValueError(
                "index_share_for_mtp_iteration disagrees with the verifier "
                f"config: {index_share} != {config_index_share}."
            )

    def _validate_against_keys(self, keys: Sequence[str] | None) -> None:
        if keys is None:
            return
        prefix = self.data["native_mtp_key_prefix"]
        if not any(key.startswith(prefix) for key in keys):
            raise ValueError(
                f"No checkpoint key starts with native_mtp_key_prefix {prefix!r}."
            )


def compare_capture_tensors(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    *,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> dict[str, Any]:
    """Compare two captured tensor sets with the plan's determinism gate."""
    if set(first) != set(second):
        raise ValueError(
            "Capture tensor sets differ: "
            f"only in first={sorted(set(first) - set(second))}, "
            f"only in second={sorted(set(second) - set(first))}."
        )
    if not first:
        raise ValueError("No tensors to compare.")

    bitwise = True
    numel = 0
    mismatched = 0
    max_abs = 0.0
    sum_abs = 0.0
    max_rel = 0.0
    within_tolerance = True

    for name in sorted(first):
        a, b = first[name], second[name]
        if a.shape != b.shape:
            raise ValueError(f"Capture tensor {name!r} shape {a.shape} != {b.shape}.")
        if a.dtype != b.dtype:
            raise ValueError(f"Capture tensor {name!r} dtype {a.dtype} != {b.dtype}.")
        if not torch.equal(a, b):
            bitwise = False
        a_fp32 = a.detach().float()
        b_fp32 = b.detach().float()
        diff = (a_fp32 - b_fp32).abs()
        denom = b_fp32.abs().clamp_min(torch.finfo(torch.float32).tiny)
        element_ok = diff <= atol + rtol * b_fp32.abs()
        within_tolerance = within_tolerance and bool(element_ok.all())
        numel += diff.numel()
        mismatched += int((~element_ok).sum())
        if diff.numel():
            max_abs = max(max_abs, float(diff.max()))
            sum_abs += float(diff.sum())
            max_rel = max(max_rel, float((diff / denom).max()))

    if bitwise:
        determinism = "bitwise"
    elif within_tolerance:
        determinism = "tolerance"
    else:
        determinism = "fail"

    return {
        "determinism": determinism,
        "bitwise": bitwise,
        "within_tolerance": within_tolerance,
        "tensor_count": len(first),
        "numel": numel,
        "mismatch_count": mismatched,
        "mismatch_fraction": (mismatched / numel) if numel else 0.0,
        "max_abs": max_abs,
        "mean_abs": (sum_abs / numel) if numel else 0.0,
        "max_rel": max_rel,
        "atol": atol,
        "rtol": rtol,
    }


def compare_capture_files(
    first: Sequence[str | Path],
    second: Sequence[str | Path],
    *,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> dict[str, Any]:
    """Compare capture files, starting from per-file SHA-256 equality.

    Bitwise-equal files win without loading tensors.  Otherwise the tensors
    are loaded and compared with the tolerance gate from section 17.11.
    """
    first_paths = [Path(path) for path in first]
    second_paths = [Path(path) for path in second]
    first_names = [path.name for path in first_paths]
    second_names = [path.name for path in second_paths]
    if first_names != second_names:
        raise ValueError(f"Capture file lists differ: {first_names} != {second_names}.")

    first_hashes = [sha256_file(path) for path in first_paths]
    second_hashes = [sha256_file(path) for path in second_paths]
    result: dict[str, Any] = {
        "files": [
            {"name": name, "sha256_first": a, "sha256_second": b}
            for name, a, b in zip(first_names, first_hashes, second_hashes, strict=True)
        ]
    }
    if first_hashes == second_hashes:
        result.update(
            {
                "determinism": "bitwise",
                "bitwise": True,
                "within_tolerance": True,
            }
        )
        return result

    from safetensors.torch import load_file

    first_tensors: dict[str, torch.Tensor] = {}
    second_tensors: dict[str, torch.Tensor] = {}
    for label, paths, sink in (
        ("first", first_paths, first_tensors),
        ("second", second_paths, second_tensors),
    ):
        for path in paths:
            loaded = load_file(str(path))
            duplicates = sorted(set(loaded) & set(sink))
            if duplicates:
                raise ValueError(
                    f"Tensor names repeat across {label} capture files: "
                    f"{duplicates}. Sharded capture files must use "
                    "distinct tensor names."
                )
            sink.update(loaded)

    result.update(
        compare_capture_tensors(first_tensors, second_tensors, atol=atol, rtol=rtol)
    )
    return result


def _close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= _CLOSE_TOLERANCE
    except (TypeError, ValueError):
        return left == right
