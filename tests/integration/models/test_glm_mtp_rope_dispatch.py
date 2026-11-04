"""NPU dispatched-rope parity gate for GLM DSA (plan section 22.3).

Compares, on Ascend NPU:

- dispatched path: vllm-ascend ``AscendRotaryEmbedding.forward_oot``
  (``torch.ops.vllm.npu_rotary_embedding``) -- what the served GLM-5.3
  verifier actually executes;
- native path: upstream ``RotaryEmbedding.forward_native``/``forward_static``
  (interleave pairing for GLM, ``is_neox_style=False``);
- absolute grounding: fp32 CPU reference of the same math.

Gates (plan section 22.3): Q/K shape and dtype must match; per-head norm
ratio dispatched/native within ``1e-2``; no global ``~0.73`` scale; partial
rotary tail must pass through unchanged; BF16/FP32 both covered.

GLM-5.3 main path: ``head_size = qk_nope(192) + qk_rope(64) = 256``,
``rotary_dim = 64``, ``is_neox_style=False`` (interleave), ``theta=8e6``.
``rotary_dim=96/192`` are non-blocking generalization regressions.

Item 8 (rot.weight present/absent fixtures) and item 10 (greedy proposal
token parity on the real teacher) are covered elsewhere: the former by the
CPU unit tests around ``GlmMTPLayer`` (own-weights branch), the latter by
the e2e 910C run with a native-forced rope server.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")
pytest.importorskip("torch_npu")

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding

if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU not available", allow_module_level=True)

DEVICE = "npu:0"
# Plan section 22.4: max_position_embeddings=1048576 in the real config; the
# cos-sin cache math only depends on positions < max, so the test uses a
# small cache and positions well within it (identical values).
MAX_POSITION_EMBEDDINGS = 8192
BASE = 8_000_000.0  # GLM-5.3 rope_parameters.rope_theta

GLM_HEAD_SIZE = 256  # qk_nope 192 + qk_rope 64
GLM_ROTARY_DIM = 64
GLM_IS_NEOX = False  # interleave pairing


@pytest.fixture(autouse=True)
def _vllm_config_ctx():
    """vllm custom ops and rope modules read the current vllm config at
    construction AND call time; wrap the whole test in a config context.
    The dispatched rope kernel also needs triton device properties, which
    the vllm worker normally initializes at startup."""
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    with set_current_vllm_config(VllmConfig()):
        yield


def _build_modules(
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
    dtype: torch.dtype,
):
    """(dispatched, native) rope modules with identical parameters."""
    dispatched = AscendRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=BASE,
        is_neox_style=is_neox_style,
        dtype=dtype,
    ).to(DEVICE)
    native = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=BASE,
        is_neox_style=is_neox_style,
        dtype=dtype,
    ).to(DEVICE)
    return dispatched, native


def _make_qk(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    q = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=DEVICE)
    k = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=DEVICE)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    return q, k, positions


def _run_dispatched(mod, positions, q, k):
    # forward_oot -> torch.ops.vllm.npu_rotary_embedding; feed clones in case
    # the op mutates its inputs.
    out_q, out_k = mod.forward_oot(positions, q.clone(), k.clone())
    return out_q, out_k


def _run_native(mod, positions, q, k):
    return mod.forward_native(positions, q.clone(), k.clone())


def _run_fp32_reference(positions, q, k, head_size, rotary_dim, is_neox_style):
    """Absolute grounding: upstream forward_static on fp32 CPU tensors."""
    cache_mod = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        base=BASE,
        is_neox_style=is_neox_style,
        dtype=torch.float32,
    )
    cos_sin = cache_mod.cos_sin_cache
    return RotaryEmbedding.forward_static(
        positions.cpu(),
        q.detach().float().cpu(),
        k.detach().float().cpu(),
        head_size,
        rotary_dim,
        cos_sin,
        is_neox_style,
    )


def _assert_rope_parity(
    out_q,
    out_k,
    ref_q,
    ref_k,
    native_q,
    native_k,
    in_q,
    in_k,
    rotary_dim: int,
    dtype: torch.dtype,
) -> None:
    # 1) shape / dtype
    for out, inp, name in ((out_q, in_q, "q"), (out_k, in_k, "k")):
        assert out.shape == inp.shape, (name, out.shape, inp.shape)
        assert out.dtype == dtype, (name, out.dtype)

    # 2) partial rotary tail passes through unchanged (bitwise)
    torch.testing.assert_close(
        out_q[..., rotary_dim:], in_q[..., rotary_dim:], rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_k[..., rotary_dim:], in_k[..., rotary_dim:], rtol=0, atol=0
    )

    # 3) elementwise dispatched vs native (bf16 kernels: fixed tolerance)
    tol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    torch.testing.assert_close(out_q, native_q, rtol=tol[0], atol=tol[1])
    torch.testing.assert_close(out_k, native_k, rtol=tol[0], atol=tol[1])

    # 4) fp32 reference grounding (looser for bf16 round-trip; ref is CPU)
    torch.testing.assert_close(
        out_q.float().cpu(), ref_q, rtol=tol[0] * 2, atol=tol[1] * 2
    )
    torch.testing.assert_close(
        out_k.float().cpu(), ref_k, rtol=tol[0] * 2, atol=tol[1] * 2
    )

    # 5) per-head norm ratio dispatched/native within 1e-2, no global scale
    for out, native, name in ((out_q, native_q, "q"), (out_k, native_k, "k")):
        n_out = out.float().pow(2).sum(-1).sqrt()
        n_nat = native.float().pow(2).sum(-1).sqrt()
        ratio = n_out / (n_nat + 1e-12)
        assert (ratio - 1).abs().max() <= 1e-2, (name, "per-head norm ratio", ratio)
        mean_ratio = ratio.mean().item()
        assert 0.95 <= mean_ratio <= 1.05, (
            name,
            "global scale deviation (0.73 symptom)",
            mean_ratio,
        )


@pytest.mark.sanity
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dispatched_native_parity_glm_main(dtype):
    """GLM-5.3 main path: head 256 / rotary 64 / interleave (blocking gate)."""
    num_tokens, num_heads = 33, 8
    dispatched, native = _build_modules(
        GLM_HEAD_SIZE, GLM_ROTARY_DIM, GLM_IS_NEOX, dtype
    )
    q, k, positions = _make_qk(num_tokens, num_heads, GLM_HEAD_SIZE, dtype)

    out_q, out_k = _run_dispatched(dispatched, positions, q, k)
    nat_q, nat_k = _run_native(native, positions, q, k)
    ref_q, ref_k = _run_fp32_reference(
        positions, q, k, GLM_HEAD_SIZE, GLM_ROTARY_DIM, GLM_IS_NEOX
    )

    _assert_rope_parity(
        out_q, out_k, ref_q, ref_k, nat_q, nat_k, q, k, GLM_ROTARY_DIM, dtype
    )


@pytest.mark.sanity
def test_dispatched_neox_pairing_does_not_cross():
    """is_neox_style True/False both match native and differ from each other."""
    num_tokens, num_heads = 17, 4
    for is_neox in (True, False):
        dispatched, native = _build_modules(
            GLM_HEAD_SIZE, GLM_ROTARY_DIM, is_neox, torch.bfloat16
        )
        q, k, positions = _make_qk(
            num_tokens, num_heads, GLM_HEAD_SIZE, dtype=torch.bfloat16, seed=1
        )

        out_q, out_k = _run_dispatched(dispatched, positions, q, k)
        nat_q, nat_k = _run_native(native, positions, q, k)

        torch.testing.assert_close(out_q, nat_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_k, nat_k, rtol=2e-2, atol=2e-2)
        if is_neox:
            neox_q = out_q
        else:
            # interleave vs neox reorder the rotated pairs: outputs must differ
            diff = (out_q.float() - neox_q.float()).abs().max()
            assert diff > 1e-2, "neox/interleave outputs unexpectedly identical"


@pytest.mark.regression
@pytest.mark.parametrize("rotary_dim", [96, 192])
def test_dispatched_native_parity_generalization(rotary_dim):
    """Non-power-of-two / non-fixed-tile rotary dims (plan: record, non-blocking)."""
    head_size = rotary_dim  # rope-only heads for the generalization case
    num_tokens, num_heads = 17, 4
    dispatched, native = _build_modules(
        head_size, rotary_dim, GLM_IS_NEOX, torch.bfloat16
    )
    q, k, positions = _make_qk(num_tokens, num_heads, head_size, torch.bfloat16)

    out_q, out_k = _run_dispatched(dispatched, positions, q, k)
    nat_q, nat_k = _run_native(native, positions, q, k)
    ref_q, ref_k = _run_fp32_reference(
        positions, q, k, head_size, rotary_dim, GLM_IS_NEOX
    )

    _assert_rope_parity(
        out_q, out_k, ref_q, ref_k, nat_q, nat_k, q, k, rotary_dim, torch.bfloat16
    )
