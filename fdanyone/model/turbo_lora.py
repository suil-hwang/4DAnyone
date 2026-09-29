"""Apply the pinned Wan2.2 Turbo difference-LoRA to a resident DiT."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from fdanyone.errors import AssetError

if TYPE_CHECKING:
    from torch import Tensor, nn

_PREFIX = "diffusion_model."
_DOWN = ".lora_down.weight"
_UP = ".lora_up.weight"
_BIAS_DIFF = ".diff_b"
_TENSOR_DIFF = ".diff"
_FUSED_METADATA_KEYS = (
    "4danyone.lora_sha256",
    "4danyone.turbo_lora_sha256",
    "4danyone.turbo_lora",
)


def _stem(key: str, suffix: str) -> str:
    stem = key[: -len(suffix)]
    return stem.removeprefix(_PREFIX)


def _direct_target(state_dict: Mapping[str, Tensor], stem: str) -> str:
    target, = (key for key in (stem, f"{stem}.weight") if key in state_dict)
    return target


def validate_turbo_base_metadata(base_metadata: Mapping[str, str] | None) -> None:
    """Reject checkpoints that explicitly identify an earlier Turbo fusion."""

    fused_markers = sorted(set(base_metadata or {}).intersection(_FUSED_METADATA_KEYS))
    if fused_markers:
        raise AssetError(f"Turbo is already fused: {fused_markers}")


def _fuse_low_rank_delta(state_dict: Mapping[str, Tensor], adapter: Mapping[str, Tensor], down_key: str) -> None:
    import torch

    up_key = f"{down_key[: -len(_DOWN)]}{_UP}"
    target = f"{_stem(down_key, _DOWN)}.weight"
    base = state_dict[target]
    down = adapter[down_key]
    up = adapter[up_key]

    merged = base.detach().to(dtype=torch.float32, copy=True)
    merged.addmm_(up.float(), down.float())
    base.copy_(merged)


def _fuse_direct_delta(state_dict: Mapping[str, Tensor], adapter: Mapping[str, Tensor], key: str) -> None:
    import torch

    if key.endswith(_BIAS_DIFF):
        target = f"{_stem(key, _BIAS_DIFF)}.bias"
    elif key.endswith(_TENSOR_DIFF):
        target = _direct_target(state_dict, _stem(key, _TENSOR_DIFF))
    else:
        raise ValueError(f"Unknown Turbo tensor: {key}")

    base = state_dict[target]
    delta = adapter[key]
    if tuple(base.shape) != tuple(delta.shape):
        raise ValueError(f"Turbo shape mismatch for {target}: {tuple(delta.shape)} != {tuple(base.shape)}")

    merged = base.detach().to(dtype=torch.float32, copy=True)
    merged.add_(delta.float())
    base.copy_(merged)


def fuse_turbo_lora(model: nn.Module, adapter_path: str | Path) -> None:
    """Fuse the fixed-strength Wan2.2 5B Turbo LoRA into one resident DiT.

    The complete adapter remains in its serialized FP16 dtype on the target
    device.  Each target uses one temporary FP32 base tensor, converts only its
    current adapter components to FP32, and copies the merged result back into
    the original base storage and dtype.
    """

    state_dict = model.state_dict(keep_vars=True)

    path = Path(adapter_path).expanduser().resolve()

    import torch
    from safetensors.torch import load_file

    device = next(iter(state_dict.values())).device
    adapter = load_file(str(path), device=str(device))
    with torch.no_grad():
        for key in sorted(adapter):
            if key.endswith(_UP):
                down_key = f"{key[: -len(_UP)]}{_DOWN}"
                if down_key not in adapter:
                    raise ValueError(f"Missing Turbo pair: {down_key}")
                continue
            if key.endswith(_DOWN):
                _fuse_low_rank_delta(state_dict, adapter, key)
            else:
                _fuse_direct_delta(state_dict, adapter, key)
