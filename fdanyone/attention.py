"""Attention backend configuration and lazily imported inference kernels."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import importlib
from importlib import metadata
import logging
import re
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from fdanyone.errors import ConfigurationError

DEFAULT_ATTENTION_BACKEND = "sage2pp"
ATTENTION_BACKEND_PRIORITY = (DEFAULT_ATTENTION_BACKEND, "sdpa")
ATTENTION_BACKEND_ALIASES = {"sageattention": DEFAULT_ATTENTION_BACKEND}
ATTENTION_BACKENDS = ("auto", *ATTENTION_BACKEND_PRIORITY, *ATTENTION_BACKEND_ALIASES)


def validate_attention_backend(backend: str) -> None:
    if not isinstance(backend, str) or backend not in ATTENTION_BACKENDS:
        raise ConfigurationError(f"Unknown attention backend: {backend!r}.")


def normalize_attention_backend(backend: str) -> str:
    """Preserve old requests while resolving SageAttention to the tested 2++ path."""

    validate_attention_backend(backend)
    return ATTENTION_BACKEND_ALIASES.get(backend, backend)


def resolve_attention_backend(backend: str, availability: Mapping[str, bool]) -> str:
    """Honor explicit selection, otherwise use the installed-backend priority."""

    backend = normalize_attention_backend(backend)
    if backend == "auto":
        for candidate in ATTENTION_BACKEND_PRIORITY:
            if availability.get(candidate, False):
                return candidate
        raise ConfigurationError("No attention backend is available.")
    if not availability.get(backend, False):
        raise ConfigurationError(f"Attention backend unavailable: {backend}; use sdpa.")
    return backend


LOGGER = logging.getLogger(__name__)
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
_INSTALL_HELP = (
    "Install SageAttention 2.2.0 for this environment's PyTorch/CUDA and GPU "
    "(see README.md#installation), "
    "or explicitly select --attention_backend sdpa."
)


@lru_cache(maxsize=1)
def _sage2pp_core() -> ModuleType:
    """Require the 2++ API and its FP8 CUDA extension, never the 1.x dispatcher."""

    try:
        version = metadata.version("sageattention")
        release = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
        if release is None or not ((2, 2) <= tuple(map(int, release.groups())) < (3, 0)):
            raise ConfigurationError(f"SageAttention 2++ requires sageattention>=2.2,<3; found {version}.")
        core = importlib.import_module("sageattention.core")
        if not callable(getattr(core, "sageattn_qk_int8_pv_fp8_cuda", None)):
            raise ConfigurationError("The installed SageAttention has no 2++ INT8/FP8 CUDA API.")
        if not getattr(core, "SM89_ENABLED", False):
            raise ConfigurationError("The SageAttention 2++ FP8 CUDA extension could not be loaded.")
        return core
    except (ImportError, OSError, RuntimeError) as exc:
        raise ConfigurationError(f"SageAttention 2++ is unavailable: {exc} {_INSTALL_HELP}") from exc


@lru_cache(maxsize=None)
def _require_sage2pp_device(index: int) -> None:
    import torch

    capability = torch.cuda.get_device_capability(index)
    if capability not in SAGE2PP_CAPABILITIES:
        raise ConfigurationError(
            f"The SageAttention 2++ INT8/FP8 backend requires SM89, SM120 or SM121; "
            f"cuda:{index} is SM{capability[0]}{capability[1]}. Use --attention_backend sdpa."
        )
    _sage2pp_core()


def get_attention_backend(backend: str = DEFAULT_ATTENTION_BACKEND, *, device=None) -> str:
    """Resolve on the selected CUDA device before expensive model preparation."""

    backend = normalize_attention_backend(backend)
    if backend == "sdpa":
        return backend
    import torch

    try:
        if not torch.cuda.is_available():
            raise ConfigurationError("SageAttention 2++ requires a CUDA GPU; use --attention_backend sdpa.")
        selected = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        if selected.type != "cuda":
            raise ConfigurationError("SageAttention 2++ requires CUDA tensors; use --attention_backend sdpa.")
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        _require_sage2pp_device(index)
    except ConfigurationError as exc:
        if backend != "auto":
            raise
        LOGGER.warning("SageAttention 2++ unavailable; auto selected SDPA: %s", exc)
        return "sdpa"
    return "sage2pp"


def attention_hnd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    """Apply unmasked, noncausal attention to independently projected HND tensors."""

    import torch
    import torch.nn.functional as F

    backend = normalize_attention_backend(backend)
    if backend == "auto":
        backend = get_attention_backend(backend, device=q.device)
    if backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    if not (q.ndim == k.ndim == v.ndim == 4):
        raise ConfigurationError("SageAttention 2++ expects rank-four HND tensors.")
    if not (q.is_cuda and q.device == k.device == v.device):
        raise ConfigurationError("SageAttention 2++ requires Q/K/V on the same CUDA device.")
    if not (q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16, torch.bfloat16)):
        raise ConfigurationError("SageAttention 2++ requires matching FP16 or BF16 Q/K/V.")
    if not (q.shape[:2] == k.shape[:2] == v.shape[:2] and k.shape[2] == v.shape[2]):
        raise ConfigurationError("SageAttention 2++ requires equal batch/head counts and matching K/V lengths.")
    if not (q.shape[-1] == k.shape[-1] == v.shape[-1] == 128):
        raise ConfigurationError("This 4DAnyone SageAttention 2++ path requires head dimension 128.")
    if min(q.shape[2], k.shape[2]) == 0 or any(value.stride(-1) != 1 for value in (q, k, v)):
        raise ConfigurationError("SageAttention 2++ requires nonempty sequences and last-dimension stride 1.")
    if torch.is_grad_enabled() and any(value.requires_grad for value in (q, k, v)):
        raise ConfigurationError("SageAttention 2++ is an inference backend; use SDPA for autograd.")
    _require_sage2pp_device(q.device.index)
    # Preserve the NHD-storage HND views. The CUDA quantizers consume these strides
    # directly, so an extra multi-gigabyte contiguous copy would be wasteful.
    return _sage2pp_core().sageattn_qk_int8_pv_fp8_cuda(q, k, v, **SAGE2PP_KERNEL_OPTIONS)
