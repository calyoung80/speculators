"""Teacher hidden-state capture manifests for the GLM MTP pipeline.

The teacher capture contract is frozen in the GLM MTP plan (sections 10,
17.11 and 21.6).  This module keeps the CPU-checkable half of it:

* build/validate a capture manifest (layer semantics, stage, hashes);
* prove that the training side applies the frozen verifier norm exactly
  once (``pre_norm``) or not at all (``post_norm``);
* compare two captures with the bitwise-first determinism gate;
* execute the section 10.3 pre-norm/post-norm parity gate against an
  extracted ``model.norm.weight``.

Nothing here imports vLLM or touches an accelerator, so the rules can be
tested before the real teacher is available.  A capture that spans multiple
prefill chunks is only comparable when it records ``chunk_semantics``.
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
    "TENSOR_SHA256_SCHEME",
    "CaptureManifest",
    "build_capture_manifest",
    "compare_capture_files",
    "compare_capture_tensors",
    "compare_post_norm_files",
    "compare_post_norm_tensors",
    "load_capture_norm",
    "load_capture_tensors",
    "reference_post_norm",
    "run_post_norm_gate",
    "sha256_file",
    "sha256_tensor",
    "write_capture_manifest",
]

CAPTURE_STAGES = ("pre_norm", "post_norm")
# Canonical tensor hash scheme for ``model_norm.weight_sha256``.  Changing this
# changes every capture manifest, so it must stay frozen once real teacher
# manifests exist (plan sections 10.3/17.11).
TENSOR_SHA256_SCHEME = "glm-mtp/tensor-raw-v1"
_CLOSE_TOLERANCE = 1e-12

_REQUIRED_FIELDS = (
    "hidden_state_stage",
    "verifier_source_layer_ordinal",
    "verifier_source_layer_index_zero_based",
    "verifier_source_tensor_semantics",
    "verifier_source_indexer_type",
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

# Fields that two captures must share to be comparable at all: same request
# bytes/positions, same source layer and vLLM target, same native MTP head.
_CAPTURE_INVARIANT_FIELDS = (
    "verifier_source_layer_ordinal",
    "verifier_source_layer_index_zero_based",
    "verifier_source_tensor_semantics",
    "verifier_source_indexer_type",
    "vllm_target_layer_id",
    "native_mtp_layer_index",
    "native_mtp_key_prefix",
    "native_mtp_tensor_semantics",
    "source_norm",
    "hidden_size",
    "dtype",
    "input_bytes_sha256",
    "position_ids_sha256",
    "index_share_for_mtp_iteration",
    "chunked_prefill",
    "chunk_semantics",
)

_DEFAULT_HIDDEN_STATE_KEYS = ("hidden_states",)
# Provenance fields identifying one serving instance (plan section 10.3: the
# two captures must come from the same server).  Optional: enforced only when
# at least one manifest records them.
_PROVENANCE_FIELDS = ("container_image", "vllm_commit", "vllm_ascend_commit")
# Logical keys that must stay bitwise aligned between the two captures.  FP8
# scale tensors are deliberately absent: they differ with the captured values.
_DEFAULT_ALIGNMENT_KEYS = (
    "input_ids",
    "token_ids",
    "position_ids",
    "lengths",
    "document_ids",
)

_PREFIX_PATTERN = re.compile(r"^model\.layers\.(\d+)\.$")
# Lowercase hex only: ``sha256_file``/``sha256_tensor`` emit lowercase, so
# accepting anything else would contradict the file-verification comparison.
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Return a canonical SHA-256 digest for one tensor.

    The digest covers the dtype, the shape and the contiguous payload, so a
    capture manifest can pin ``model.norm.weight`` without depending on the
    checkpoint file layout or on how the tensor was sliced.  Payload bytes are
    the tensor's raw machine-order storage; bfloat16 (and other dtypes numpy
    cannot export) is hashed through a ``uint8`` view.  The canonical scheme is
    :data:`TENSOR_SHA256_SCHEME`; every producer must use it or manifests from
    different tools will not recognise the same norm weight.
    """
    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    try:
        payload = value.numpy().tobytes()
    except TypeError:
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(payload)
    return digest.hexdigest()


def _as_path_list(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Normalize a single path or a sequence of paths into a list."""
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a 64-character lowercase hexadecimal SHA-256, "
            f"got {value!r}."
        )


def _require_index(value: Any, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")


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
    :param base_dir: directory used to resolve relative ``capture_files``
        paths.  Defaults to ``None`` (resolve against the working directory).
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
        verify_files: bool = False,
    ) -> None:
        self.data = dict(data)
        self.base_dir: Path | None = (
            Path(base_dir).resolve() if base_dir is not None else None
        )
        self.validate(verify_files=verify_files)

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
        return cls(data, base_dir=manifest_path.resolve().parent)

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
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
            raise ValueError(
                "verifier_source_layer_ordinal must be a positive integer, got "
                f"{ordinal!r}."
            )
        _require_index(
            self.data["verifier_source_layer_index_zero_based"],
            "verifier_source_layer_index_zero_based",
        )
        if zero_based != ordinal - 1:
            raise ValueError(
                "verifier_source_layer_index_zero_based must equal "
                "verifier_source_layer_ordinal - 1 "
                f"({zero_based} != {ordinal - 1})."
            )

        _require_index(self.data["vllm_target_layer_id"], "vllm_target_layer_id")
        _require_index(self.data["native_mtp_layer_index"], "native_mtp_layer_index")
        _require_sha256(self.data["input_bytes_sha256"], "input_bytes_sha256")
        _require_sha256(self.data["position_ids_sha256"], "position_ids_sha256")

        chunked = self.data.get("chunked_prefill")
        if chunked is not None and not isinstance(chunked, bool):
            raise ValueError(
                f"chunked_prefill must be a bool when present, got {chunked!r}."
            )
        chunk_semantics = self.data.get("chunk_semantics")
        if chunked and (
            not isinstance(chunk_semantics, Mapping) or not chunk_semantics
        ):
            raise ValueError(
                "A chunked-prefill capture must record how chunks are "
                "stitched (non-empty 'chunk_semantics'): plan section 10.3 "
                "only allows positional comparison for single-chunk requests."
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
            _require_sha256(entry["sha256"], "capture_files[].sha256")
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
        _require_sha256(model_norm["weight_sha256"], "model_norm.weight_sha256")
        if "eps" in model_norm and (
            not isinstance(model_norm["eps"], int | float)
            or isinstance(model_norm["eps"], bool)
        ):
            raise ValueError(
                f"model_norm.eps must be a number, got {model_norm['eps']!r}."
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

    def resolve_capture_files(self) -> list[Path]:
        """Return ``capture_files`` paths resolved against the manifest dir."""
        return [
            self._resolve_capture_path(entry["path"])
            for entry in self.data["capture_files"]
        ]

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
        if indexer_type not in ("full", "shared"):
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


_STAGE_TENSOR_SEMANTICS = {
    "pre_norm": "decoder_output_before_model_norm",
    "post_norm": "model_norm_output",
}


def build_capture_manifest(
    *,
    hidden_state_stage: str,
    verifier_source_layer_ordinal: int,
    native_mtp_layer_index: int,
    native_mtp_key_prefix: str,
    vllm_target_layer_id: int,
    hidden_size: int,
    dtype: str | torch.dtype,
    input_bytes_sha256: str,
    position_ids_sha256: str,
    capture_files: str | Path | Sequence[str | Path],
    model_norm_weight: torch.Tensor,
    verifier_source_indexer_type: str,
    model_norm_eps: float | None = None,
    chunked_prefill: bool | None = None,
    chunk_semantics: Mapping[str, Any] | None = None,
    source_norm: str = "model.norm",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a capture manifest from producer-side facts.

    This is the writer half of the contract: the vLLM hook (or an offline
    capture tool) hashes its capture files and the frozen ``model.norm.weight``
    here, and derives every field whose value is a pure function of the stage.
    The result is validated by :class:`CaptureManifest` before it is written, so
    a producer cannot emit a manifest the training side would reject.

    :param hidden_state_stage: ``"pre_norm"`` or ``"post_norm"``.
    :param verifier_source_layer_ordinal: 1-based verifier layer ordinal
        (``num_hidden_layers`` for the last decoder layer).
    :param native_mtp_layer_index: index of the native MTP head in the
        checkpoint (``model.layers.<N>.``), which is *not* the capture layer.
    :param native_mtp_key_prefix: native checkpoint prefix, e.g.
        ``"model.layers.78."``.
    :param vllm_target_layer_id: the ``--target-layer-ids`` value the teacher
        was launched with.
    :param hidden_size: capture hidden size.
    :param dtype: capture dtype as a string or ``torch.dtype``.
    :param input_bytes_sha256: SHA-256 of the exact request input bytes.
    :param position_ids_sha256: SHA-256 of the position ids used for capture.
    :param capture_files: capture files to hash and list.  Paths are recorded
        as given, so they must be absolute or relative to the directory the
        manifest will be written to.
    :param model_norm_weight: the frozen verifier ``model.norm.weight``.
    :param verifier_source_indexer_type: source-layer indexer mode
        (``"full"``/``"shared"``); required by the frozen capture contract.
    :param model_norm_eps: ``model.norm`` epsilon (required for ``post_norm``).
    :param chunked_prefill: whether the request spans multiple prefill chunks.
        ``True`` requires ``chunk_semantics`` so the gate never compares
        position-by-position across an undefined stitch (plan section 10.3).
    :param chunk_semantics: recorded chunk stitch semantics (positions, KV
        page/block, padding) when ``chunked_prefill`` is true.
    :param source_norm: norm that produced the capture; ``"model.norm"``.
    :param extra: additional provenance fields (container/vLLM commits, ...).
        Keys may not shadow derived fields.
    :return: the manifest mapping, ready for :func:`write_capture_manifest`.
    """
    if hidden_state_stage not in CAPTURE_STAGES:
        raise ValueError(
            f"Unknown hidden_state_stage {hidden_state_stage!r}; expected one "
            f"of {CAPTURE_STAGES}."
        )
    if hidden_state_stage == "post_norm" and model_norm_eps is None:
        raise ValueError(
            "post_norm captures must record model.norm epsilon "
            "(model_norm_eps); the parity gate needs it to rebuild the "
            "reference output."
        )
    capture_paths = _as_path_list(capture_files)
    if not capture_paths:
        raise ValueError("capture_files must be a non-empty sequence.")

    capture_tensor_semantics = _STAGE_TENSOR_SEMANTICS[hidden_state_stage]
    data: dict[str, Any] = {
        "hidden_state_stage": hidden_state_stage,
        "verifier_source_layer_ordinal": verifier_source_layer_ordinal,
        "verifier_source_layer_index_zero_based": verifier_source_layer_ordinal - 1,
        "verifier_source_tensor_semantics": "decoder_output_before_model_norm",
        "capture_tensor_semantics": capture_tensor_semantics,
        "vllm_target_layer_id": vllm_target_layer_id,
        "native_mtp_layer_index": native_mtp_layer_index,
        "native_mtp_key_prefix": native_mtp_key_prefix,
        "native_mtp_tensor_semantics": "native_mtp_head",
        "source_norm": source_norm,
        "hidden_size": hidden_size,
        "dtype": (
            str(dtype).removeprefix("torch.")
            if isinstance(dtype, torch.dtype)
            else str(dtype)
        ),
        "input_bytes_sha256": input_bytes_sha256,
        "position_ids_sha256": position_ids_sha256,
        "capture_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in capture_paths
        ],
        "model_norm": {
            "weight_sha256": sha256_tensor(model_norm_weight),
        },
        "verifier_source_indexer_type": verifier_source_indexer_type,
    }
    if model_norm_eps is not None:
        data["model_norm"]["eps"] = model_norm_eps
    if chunked_prefill is not None:
        data["chunked_prefill"] = chunked_prefill
    if chunk_semantics is not None:
        data["chunk_semantics"] = dict(chunk_semantics)

    if extra:
        overlap = sorted(set(extra) & set(data))
        if overlap:
            raise ValueError(
                f"extra fields may not override derived capture fields: {overlap}."
            )
        data.update(extra)

    CaptureManifest(data)
    return data


def write_capture_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> CaptureManifest:
    """Validate and atomically write one ``capture_manifest.json``.

    Relative ``capture_files`` entries are resolved against the manifest's
    directory.  With ``verify_files=True`` every capture file must exist and
    match its recorded SHA-256, so training never starts on a partial or
    modified capture (plan section 17.11).
    """
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    validated = CaptureManifest(
        manifest,
        base_dir=manifest_path.parent,
        verify_files=verify_files,
    )

    _write_json_atomic(manifest_path, validated.data)
    return validated


def load_capture_tensors(
    paths: str | Path | Sequence[str | Path],
    *,
    label: str = "capture",
) -> dict[str, torch.Tensor]:
    """Load capture safetensors files into one tensor mapping.

    Tensor names must be unique across shards; a repeated name means the caller
    concatenated the same shard twice and would silently compare against a
    truncated capture.
    """
    from safetensors.torch import load_file

    tensors: dict[str, torch.Tensor] = {}
    for path in _as_path_list(paths):
        loaded = load_file(str(path))
        duplicates = sorted(set(loaded) & set(tensors))
        if duplicates:
            raise ValueError(
                f"Tensor names repeat across {label} capture files: "
                f"{duplicates}. Sharded capture files must use distinct tensor "
                "names."
            )
        tensors.update(loaded)
    return tensors


def _token_count(tensors: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for tensor in tensors.values():
        count = 1
        for size in tensor.shape[:-1]:
            count *= int(size)
        total += count
    return total


def _check_alignment(
    pre_norm: Mapping[str, torch.Tensor],
    post_norm: Mapping[str, torch.Tensor],
    alignment_keys: Sequence[str],
) -> dict[str, Any]:
    """Check that non-hidden tensors (tokens/positions) are identical."""
    present = [key for key in alignment_keys if key in pre_norm]
    mismatched: list[str] = []
    for key in present:
        first = pre_norm[key]
        second = post_norm.get(key)
        if (
            second is None
            or first.shape != second.shape
            or first.dtype != second.dtype
            or not torch.equal(first, second)
        ):
            mismatched.append(key)
    return {
        "keys": present,
        "aligned": not mismatched,
        "mismatched_keys": mismatched,
    }


def compare_post_norm_tensors(
    pre_norm: Mapping[str, torch.Tensor],
    post_norm: Mapping[str, torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    *,
    keys: Sequence[str] = _DEFAULT_HIDDEN_STATE_KEYS,
    alignment_keys: Sequence[str] = _DEFAULT_ALIGNMENT_KEYS,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    expected_weight_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the section 10.3 gate: ``model.norm(pre_norm)`` vs ``post_norm``.

    Only ``keys`` (``hidden_states`` by default) are normalized and value
    compared.  Real capture files also carry ``token_ids``/``input_ids``/
    ``position_ids``/FP8 scales; the token-side keys must match bitwise
    (``alignment_keys``) and the remaining tensors are ignored, never fed
    through the norm.  FP8 captures must be dequantized first: this gate is
    defined for the BF16 capture contract.
    """
    if set(pre_norm) != set(post_norm):
        raise ValueError(
            "Capture tensor sets differ: "
            f"only in pre_norm={sorted(set(pre_norm) - set(post_norm))}, "
            f"only in post_norm={sorted(set(post_norm) - set(pre_norm))}."
        )
    if not pre_norm:
        raise ValueError("No tensors to compare.")
    if "hidden_states_scales" in pre_norm:
        raise NotImplementedError(
            "FP8 captures (hidden_states + hidden_states_scales) must be "
            "dequantized before the post-norm parity gate; the gate expects "
            "the BF16 capture contract."
        )

    missing = [key for key in keys if key not in pre_norm]
    if missing:
        raise ValueError(
            f"Capture files are missing hidden-state keys {missing}; pass "
            "'keys' for a non-default capture schema."
        )

    reference = {
        name: reference_post_norm(pre_norm[name], weight, eps) for name in keys
    }
    result = compare_capture_tensors(
        reference,
        {name: post_norm[name] for name in keys},
        atol=atol,
        rtol=rtol,
    )
    alignment = _check_alignment(pre_norm, post_norm, alignment_keys)
    weight_sha256 = sha256_tensor(weight)
    weight_hash_match = (
        None
        if expected_weight_sha256 is None
        else weight_sha256 == expected_weight_sha256
    )
    passed = (
        result["within_tolerance"]
        and alignment["aligned"]
        and weight_hash_match is not False
    )
    result.update(
        {
            "gate": "pass" if passed else "fail",
            "token_count": _token_count({name: post_norm[name] for name in keys}),
            "compared_keys": list(keys),
            "token_alignment": alignment,
            "ignored_keys": sorted(set(pre_norm) - set(keys) - set(alignment_keys)),
            "eps": eps,
            "weight_sha256": weight_sha256,
            "expected_weight_sha256": expected_weight_sha256,
            "weight_hash_match": weight_hash_match,
        }
    )
    return result


def compare_post_norm_files(
    pre_norm: str | Path | Sequence[str | Path],
    post_norm: str | Path | Sequence[str | Path],
    weight: torch.Tensor,
    eps: float,
    *,
    keys: Sequence[str] = _DEFAULT_HIDDEN_STATE_KEYS,
    alignment_keys: Sequence[str] = _DEFAULT_ALIGNMENT_KEYS,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    expected_weight_sha256: str | None = None,
) -> dict[str, Any]:
    """File-backed :func:`compare_post_norm_tensors` with per-file hashes."""
    pre_paths = _as_path_list(pre_norm)
    post_paths = _as_path_list(post_norm)
    result = compare_post_norm_tensors(
        load_capture_tensors(pre_paths, label="pre_norm"),
        load_capture_tensors(post_paths, label="post_norm"),
        weight,
        eps,
        keys=keys,
        alignment_keys=alignment_keys,
        atol=atol,
        rtol=rtol,
        expected_weight_sha256=expected_weight_sha256,
    )
    result["files"] = {
        "pre_norm": [
            {"path": str(path), "sha256": sha256_file(path)} for path in pre_paths
        ],
        "post_norm": [
            {"path": str(path), "sha256": sha256_file(path)} for path in post_paths
        ],
    }
    return result


def load_capture_norm(
    source: str | Path | torch.Tensor,
    *,
    weight_key: str = "model.norm.weight",
    bias_key: str = "model.norm.bias",
) -> torch.Tensor:
    """Load the frozen verifier norm weight for the parity gate.

    Accepts the already-materialized tensor, a single local safetensors file,
    or a local native/quantized model directory (``model.safetensors.index.json``
    or ``quant_model_weights.safetensors.index.json``).  No hub downloads: a
    path that does not exist is a hard error instead of a network round trip
    with a confusing failure.  A non-zero ``model.norm.bias`` is rejected: the
    RMSNorm reference has no bias term and silently ignoring it would bias the
    gate (plan section 9.2).
    """
    if isinstance(source, torch.Tensor):
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"model.norm source {source} does not exist; the parity gate "
            "accepts a local safetensors file, a local model directory, or an "
            "already-loaded tensor."
        )
    if path.is_file():
        from safetensors.torch import load_file

        tensors = load_file(str(path))
    else:
        from speculators.utils.loading import load_model_layers

        tensors = load_model_layers([weight_key, bias_key], str(path))

    if weight_key not in tensors:
        raise ValueError(f"No {weight_key!r} found in {source}.")
    bias = tensors.get(bias_key)
    if bias is not None and bool(torch.count_nonzero(bias)):
        raise ValueError(
            f"{bias_key!r} is non-zero; the post-norm reference gate only "
            "supports the converter's all-zero-bias policy."
        )
    return tensors[weight_key]


def run_post_norm_gate(  # noqa: C901
    pre_norm_manifest: str | Path | CaptureManifest,
    post_norm_manifest: str | Path | CaptureManifest,
    *,
    norm_source: str | Path | torch.Tensor,
    keys: Sequence[str] = _DEFAULT_HIDDEN_STATE_KEYS,
    alignment_keys: Sequence[str] = _DEFAULT_ALIGNMENT_KEYS,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the section 10.3 pre-norm -> post-norm acceptance gate.

    This is the executor around :func:`compare_post_norm_files`: it validates
    both manifests and their capture files, proves that both captures describe
    the same request/source layer/vLLM target, loads and hashes
    ``model.norm.weight``, runs the tolerance gate, and optionally writes a
    JSON report.  The comparison is only meaningful for a single prefill chunk;
    a manifest declaring ``chunked_prefill`` must carry ``chunk_semantics`` and
    both manifests must agree on it.
    """
    pre_manifest = _coerce_manifest(pre_norm_manifest)
    post_manifest = _coerce_manifest(post_norm_manifest)
    if pre_manifest.hidden_state_stage != "pre_norm":
        raise ValueError(
            "run_post_norm_gate expects a pre_norm capture manifest, got "
            f"{pre_manifest.hidden_state_stage!r}."
        )
    if post_manifest.hidden_state_stage != "post_norm":
        raise ValueError(
            "run_post_norm_gate expects a post_norm capture manifest, got "
            f"{post_manifest.hidden_state_stage!r}."
        )

    mismatches: dict[str, Any] = {}
    for field in _CAPTURE_INVARIANT_FIELDS:
        first = pre_manifest.data.get(field)
        second = post_manifest.data.get(field)
        if first != second:
            mismatches[field] = {"pre_norm": first, "post_norm": second}
    for field in _PROVENANCE_FIELDS:
        first = pre_manifest.data.get(field)
        second = post_manifest.data.get(field)
        if (first is not None or second is not None) and first != second:
            mismatches[field] = {"pre_norm": first, "post_norm": second}
    pre_eps = pre_manifest.data["model_norm"].get("eps")
    post_eps = post_manifest.data["model_norm"].get("eps")
    if pre_eps is not None and post_eps is not None and not _close(pre_eps, post_eps):
        mismatches["model_norm.eps"] = {
            "pre_norm": pre_eps,
            "post_norm": post_eps,
        }
    pre_weight_sha = pre_manifest.data["model_norm"]["weight_sha256"]
    post_weight_sha = post_manifest.data["model_norm"]["weight_sha256"]
    if pre_weight_sha != post_weight_sha:
        mismatches["model_norm.weight_sha256"] = {
            "pre_norm": pre_weight_sha,
            "post_norm": post_weight_sha,
        }
    if mismatches:
        raise ValueError(
            "The two captures do not describe the same request/source layer: "
            f"{mismatches}."
        )

    weight = load_capture_norm(norm_source)
    weight_sha256 = sha256_tensor(weight)
    if weight_sha256 != pre_weight_sha:
        raise ValueError(
            "Loaded model.norm.weight does not match the capture manifest: "
            f"{weight_sha256} != {pre_weight_sha}."
        )

    eps = post_eps if post_eps is not None else pre_eps
    if eps is None:
        raise ValueError(
            "Neither capture manifest records model_norm.eps; cannot rebuild "
            "the reference norm."
        )

    result = compare_post_norm_files(
        pre_manifest.resolve_capture_files(),
        post_manifest.resolve_capture_files(),
        weight,
        eps,
        keys=keys,
        alignment_keys=alignment_keys,
        atol=atol,
        rtol=rtol,
        expected_weight_sha256=pre_weight_sha,
    )
    result["request"] = {
        "input_bytes_sha256": pre_manifest.data["input_bytes_sha256"],
        "position_ids_sha256": pre_manifest.data["position_ids_sha256"],
        "vllm_target_layer_id": pre_manifest.data["vllm_target_layer_id"],
        "verifier_source_layer_ordinal": pre_manifest.data[
            "verifier_source_layer_ordinal"
        ],
        "native_mtp_key_prefix": pre_manifest.data["native_mtp_key_prefix"],
    }
    result["chunked_prefill"] = bool(pre_manifest.data.get("chunked_prefill", False))
    result["chunk_semantics"] = pre_manifest.data.get("chunk_semantics")

    if report_path is not None:
        _write_json_atomic(report_path, result)
    return result


def _coerce_manifest(value: str | Path | CaptureManifest) -> CaptureManifest:
    manifest = (
        value
        if isinstance(value, CaptureManifest)
        else CaptureManifest.from_path(value)
    )
    manifest.validate(verify_files=True)
    return manifest


def _write_json_atomic(path: str | Path, data: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f"{output.name}.tmp")
    temp_path.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(output)


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

    result.update(
        compare_capture_tensors(
            load_capture_tensors(first_paths, label="first"),
            load_capture_tensors(second_paths, label="second"),
            atol=atol,
            rtol=rtol,
        )
    )
    return result


def _close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= _CLOSE_TOLERANCE
    except (TypeError, ValueError):
        return left == right
