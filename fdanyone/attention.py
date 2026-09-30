# fdanyone/attention.py
from __future__ import annotations

import importlib
import logging
import re
from functools import cache
from importlib import metadata
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from fdanyone.errors import ConfigurationError

LOGGER = logging.getLogger(__name__)
DEFAULT_ATTENTION_BACKEND = "sage2pp"
# "sageattention" is an alias kept for old requests; it resolves to the tested 2++ path.
ATTENTION_BACKENDS = ("auto", DEFAULT_ATTENTION_BACKEND, "sdpa", "sageattention")
SAGE2PP_CAPABILITIES = frozenset({(8, 9), (12, 0), (12, 1)})
SAGE2PP_KERNEL_OPTIONS = {
    "tensor_layout": "HND",
    "is_causal": False,
    "qk_quant_gran": "per_warp",
    "pv_accum_dtype": "fp32+fp16",
    "smooth_k": True,
    "smooth_v": False,
    "return_lse": False,
}


def normalize_attention_backend(backend: str) -> str:
    """Reject unknown backends and resolve the SageAttention alias to the tested 2++ path."""

    if backend not in ATTENTION_BACKENDS:
        raise ConfigurationError(f"Unknown attention backend: {backend!r}.")
    return DEFAULT_ATTENTION_BACKEND if backend == "sageattention" else backend


@cache
def _sage2pp(index: int) -> ModuleType:
    """Require an SM89/SM120/SM121 device and the 2++ API with its FP8 CUDA extension, never the 1.x dispatcher."""

    import torch

    major, minor = torch.cuda.get_device_capability(index)
    try:
        version = metadata.version("sageattention")
        release = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
        # Never import another major API (1.x dispatcher, 3.x); 2.2 is the first release with the 2++ FP8 kernel.
        compatible = release is not None and (2, 2) <= tuple(map(int, release.groups())) < (3, 0)
        core = importlib.import_module("sageattention.core") if compatible else None
    except (ImportError, OSError, RuntimeError) as exc:
        raise ConfigurationError(f"SageAttention 2++ unavailable ({exc}); use --attention_backend sdpa.") from exc
    if core is None or (major, minor) not in SAGE2PP_CAPABILITIES:
        raise ConfigurationError(f"No SageAttention 2++ for sageattention {version} on SM{major}{minor}; use sdpa.")
    # sageattention.core imports even when its FP8 CUDA extension fails to load (e.g. a Windows DLL error).
    if not getattr(core, "SM89_ENABLED", False):
        raise ConfigurationError("SageAttention 2++ FP8 CUDA extension failed to load; use --attention_backend sdpa.")
    return core


def get_attention_backend(backend: str = DEFAULT_ATTENTION_BACKEND, *, device=None) -> str:
    """Resolve on the selected CUDA device before expensive model preparation."""

    backend = normalize_attention_backend(backend)
    if backend == "sdpa":
        return backend
    import torch

    try:
        index = torch.device("cuda" if device is None else device).index
        _sage2pp(torch.cuda.current_device() if index is None else index)
    except ConfigurationError as exc:
        if backend != "auto":
            raise
        LOGGER.warning("SageAttention 2++ unavailable; auto selected SDPA: %s", exc)
        return "sdpa"
    return "sage2pp"


def attention_hnd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    """Apply unmasked, noncausal attention to HND tensors with a backend from get_attention_backend."""

    if backend == "sdpa":
        import torch.nn.functional as F

        return F.scaled_dot_product_attention(q, k, v)
    # Pass the strided HND views as-is; the CUDA quantizers read them without a multi-GB contiguous copy.
    return _sage2pp(q.device.index).sageattn_qk_int8_pv_fp8_cuda(q, k, v, **SAGE2PP_KERNEL_OPTIONS)
