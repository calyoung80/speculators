# sitecustomize.py - auto-imported by Python at startup
# Patches topk_softmax/topk_sigmoid for Ascend NPU (no _moe_C C++ extension)
# Forces global patch loading for vllm-ascend
try:
    import os
    import sys
    print(f"[sitecustomize] loading, PID={os.getpid()}, VLLM_PLUGINS={os.environ.get('VLLM_PLUGINS','NOT_SET')}, PYTHONPATH={os.environ.get('PYTHONPATH','')[:80]}", flush=True)
    _cann_py = "/usr/local/Ascend/cann-9.1.0/python/site-packages"
    if _cann_py not in sys.path and os.path.isdir(_cann_py):
        sys.path.insert(0, _cann_py)

    # Initialize NPU device early so triton get_arch() works in subprocesses
    try:
        import torch_npu  # noqa
        if torch.npu.is_available():
            torch.npu.set_device(0)
    except Exception:
        pass

    if os.environ.get("VLLM_PLUGINS") == "ascend":
        # Force load global patches (patch_fused_moe etc.)
        import vllm_ascend.patch.platform  # noqa
        # Also apply _ensure_global_patch
        import vllm_ascend
        vllm_ascend._ensure_global_patch()

        # Patch topk_softmax/topk_sigmoid for Ascend (no _moe_C C++ extension)
        import torch
        import torch.nn.functional as F
        import vllm._custom_ops as ops

        def _topk_softmax_pt(topk_weights, topk_ids, token_expert_indices,
                             gating_output, renormalize=False,
                             e_score_correction_bias=None):
            if e_score_correction_bias is not None:
                gating_output = gating_output.float()
                gating_output = gating_output - e_score_correction_bias.unsqueeze(0)
            probs = F.softmax(gating_output.float(), dim=-1)
            k = topk_weights.size(1)
            weights, indices = probs.topk(k, dim=-1)
            if renormalize:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            topk_weights.copy_(weights.to(topk_weights.dtype))
            topk_ids.copy_(indices.to(topk_ids.dtype))
            token_expert_indices.copy_(indices.to(token_expert_indices.dtype))

        def _topk_sigmoid_pt(topk_weights, topk_ids, token_expert_indices,
                              gating_output, renormalize=False,
                              e_score_correction_bias=None):
            if e_score_correction_bias is not None:
                gating_output = gating_output.float()
                gating_output = gating_output - e_score_correction_bias.unsqueeze(0)
            scores = torch.sigmoid(gating_output.float())
            k = topk_weights.size(1)
            weights, indices = scores.topk(k, dim=-1)
            if renormalize:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            topk_weights.copy_(weights.to(topk_weights.dtype))
            topk_ids.copy_(indices.to(topk_ids.dtype))
            token_expert_indices.copy_(indices.to(token_expert_indices.dtype))

        ops.topk_softmax = _topk_softmax_pt
        ops.topk_sigmoid = _topk_sigmoid_pt
        print(f"[sitecustomize] patches applied OK, PID={os.getpid()}", flush=True)
except Exception as e:
    print(f"[sitecustomize] ERROR: {e}", flush=True)
