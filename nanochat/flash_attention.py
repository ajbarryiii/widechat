"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Backend priority (auto mode):
1) Flash Attention 3 on Hopper (sm90)
2) PyTorch SDPA fallback everywhere else

Exports `flash_attn` namespace matching the Flash Attention interface.
"""
import importlib
import os

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 kernels
# =============================================================================
def _resolve_kernel_interface(kernel):
    """Return an object exposing flash_attn_func/flash_attn_with_kvcache."""
    candidates = [
        getattr(kernel, "flash_attn_interface", None),
        getattr(kernel, "interface", None),
        kernel,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "flash_attn_func") and hasattr(candidate, "flash_attn_with_kvcache"):
            return candidate
    return None


class _FlashAttnFunctionAdapter:
    """Adapter for modules that only expose flash_attn_func."""

    def __init__(self, flash_attn_func_impl, flash_attn_with_kvcache_impl=None):
        self._flash_attn_func_impl = flash_attn_func_impl
        self._flash_attn_with_kvcache_impl = flash_attn_with_kvcache_impl

    def flash_attn_func(self, q, k, v, causal=False, window_size=(-1, -1)):
        return self._flash_attn_func_impl(q, k, v, causal=causal, window_size=window_size)

    def flash_attn_with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        cache_seqlens=None,
        causal=False,
        window_size=(-1, -1),
    ):
        if self._flash_attn_with_kvcache_impl is not None:
            return self._flash_attn_with_kvcache_impl(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                causal=causal,
                window_size=window_size,
            )
        return globals()["_sdpa_flash_attn_with_kvcache_impl"](
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            window_size=window_size,
        )


def _resolve_function_only_interface(module):
    flash_attn_func_impl = getattr(module, "flash_attn_func", None)
    if flash_attn_func_impl is None:
        return None

    flash_attn_with_kvcache_impl = getattr(module, "flash_attn_with_kvcache", None)
    if flash_attn_with_kvcache_impl is None:
        try:
            fallback_module = importlib.import_module("flash_attn.flash_attn_interface")
            flash_attn_with_kvcache_impl = getattr(fallback_module, "flash_attn_with_kvcache", None)
        except Exception:
            flash_attn_with_kvcache_impl = None

    return _FlashAttnFunctionAdapter(
        flash_attn_func_impl=flash_attn_func_impl,
        flash_attn_with_kvcache_impl=flash_attn_with_kvcache_impl,
    )


def _resolve_module_interface(module):
    interface = _resolve_kernel_interface(module)
    if interface is not None:
        return interface
    return _resolve_function_only_interface(module)


_fa3_probe = "not_attempted"

_DEFAULT_FA3_REPO_ID = "varunneal/flash-attention-3"
_FA3_REPO_ID_ENV = "NANOCHAT_FA3_REPO_ID"


def _resolve_repo_id(env_var: str, default_repo_id: str) -> str:
    repo_id = os.environ.get(env_var, "").strip()
    return repo_id if repo_id else default_repo_id


def _load_flash_attention_3():
    """Try to load Flash Attention 3 (Hopper GPU, sm90)."""
    global _fa3_probe
    if not torch.cuda.is_available():
        _fa3_probe = "cuda_unavailable"
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
        if major != 9:
            _fa3_probe = f"unsupported_cc_sm{major}{minor}"
            return None

        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        kernel = get_kernel(_resolve_repo_id(_FA3_REPO_ID_ENV, _DEFAULT_FA3_REPO_ID))
        interface = _resolve_module_interface(kernel)
        if interface is None:
            _fa3_probe = "invalid_kernel_interface"
            return None
        _fa3_probe = "available"
        return interface
    except Exception as exc:
        _fa3_probe = f"load_error:{type(exc).__name__}"
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
HAS_FLASH_ATTN = HAS_FA3

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _backend_name():
    """Return selected backend name after override resolution."""
    assert _override_impl in (None, 'fa3', 'sdpa')
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'sdpa':
        return 'sdpa'
    if HAS_FA3:
        return 'fa3'
    return 'sdpa'


def backend_diagnostics():
    """Structured backend selection diagnostics for logs/artifacts."""
    backend_name = _backend_name()
    mode = _override_impl if _override_impl is not None else "auto"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        capability = f"{major}.{minor}"
    else:
        capability = "n/a"
    return {
        "selected_backend": backend_name,
        "mode": mode,
        "has_fa3": HAS_FA3,
        "cuda_available": torch.cuda.is_available(),
        "cuda_cc": capability,
        "fa3_probe": _fa3_probe,
    }


def backend_status_message():
    """Return an explicit one-line backend status message for logs."""
    diagnostics = backend_diagnostics()
    return (
        "Flash Attention backend selection: "
        f"selected={diagnostics['selected_backend']}, mode={diagnostics['mode']}, "
        f"has_fa3={diagnostics['has_fa3']}, "
        f"cuda_available={diagnostics['cuda_available']}, cuda_cc={diagnostics['cuda_cc']}, "
        f"fa3_probe={diagnostics['fa3_probe']}"
    )


def _use_fa3():
    """Backward-compatibility helper used by older tests."""
    return _backend_name() == 'fa3'


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def _sdpa_flash_attn_with_kvcache_impl(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, window_size=(-1, -1)):
    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    assert cache_seqlens is not None, "cache_seqlens is required for SDPA fallback"
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    backend_name = _backend_name()
    if backend_name == 'fa3':
        assert _fa3 is not None
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    backend_name = _backend_name()
    if backend_name == 'fa3':
        assert _fa3 is not None
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    return _sdpa_flash_attn_with_kvcache_impl(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        window_size=window_size,
    )


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
