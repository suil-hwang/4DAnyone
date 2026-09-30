# fdanyone/device.py
from __future__ import annotations

import os
import subprocess
from collections.abc import MutableMapping, Sequence

from fdanyone.errors import ConfigurationError

CUDA_ALLOCATOR_CONF = "PYTORCH_CUDA_ALLOC_CONF"
CUDA_MAX_SPLIT_SIZE_MB = 4096
LOW_MEMORY_GPU_MAX_BYTES = 24 * 1024**3


def has_low_memory_gpu(
    gpu_ids: Sequence[int] | None,
    environment: MutableMapping[str, str] | None = None,
) -> bool:
    """Return whether any selected GPU has at most 24 GiB, asking nvidia-smi before PyTorch."""

    environment = os.environ if environment is None else environment
    raw = environment.get("CUDA_VISIBLE_DEVICES", environment.get("NVIDIA_VISIBLE_DEVICES"))
    # CUDA-visible nvidia-smi IDs (None = all); nvidia-smi rejects hidden forms (-1, none, void), giving False.
    if raw is None or raw.strip().lower() == "all":
        identifiers = None
    else:
        identifiers = tuple(item.strip() for item in raw.split(",") if item.strip())
    if gpu_ids is not None:
        try:
            identifiers = tuple(str(gpu_id) if identifiers is None else identifiers[gpu_id] for gpu_id in gpu_ids)
        except (IndexError, TypeError):
            # select_cuda_devices rejects invalid IDs before any GPU work.
            return False
    command = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    if identifiers is not None:
        command.insert(1, f"--id={','.join(identifiers)}")
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        totals_mib = [int(line) for line in completed.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return False
    return any(total_mib * 1024**2 <= LOW_MEMORY_GPU_MAX_BYTES for total_mib in totals_mib)


def configure_inference_cuda_allocator(
    environment: MutableMapping[str, str] | None = None,
    *,
    use_expandable_segments: bool = False,
) -> None:
    """Configure the long-lived DiT allocator before the first PyTorch import."""

    environment = os.environ if environment is None else environment
    current = environment.get(CUDA_ALLOCATOR_CONF, "").strip()
    options = tuple(option.strip() for option in current.split(",") if option.strip())
    keys = {option.partition(":")[0] for option in options}
    backend = next((option.partition(":")[2] for option in options if option.partition(":")[0] == "backend"), None)
    if backend not in {None, "native"}:
        return
    additions: list[str] = []
    if use_expandable_segments and "expandable_segments" not in keys:
        additions.append("expandable_segments:True")
    if "max_split_size_mb" not in keys:
        additions.append(f"max_split_size_mb:{CUDA_MAX_SPLIT_SIZE_MB}")
    if additions:
        environment[CUDA_ALLOCATOR_CONF] = ",".join((*options, *additions))


def select_cuda_device(device: str) -> tuple[str, int]:
    """Validate, select, and normalize one CUDA device."""

    import torch

    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(requested)
    index = torch.cuda.current_device()
    return f"cuda:{index}", index


def validate_gpu_ids(gpu_ids: Sequence[int] | None, available: int) -> tuple[int, ...]:
    """Validate logical indices without initializing CUDA or selecting a device."""
    selected = tuple(range(available) if gpu_ids is None else gpu_ids)
    valid = all(type(gpu_id) is int and 0 <= gpu_id < available for gpu_id in selected)
    if not (selected and valid and len(set(selected)) == len(selected)):
        raise ConfigurationError(f"Select at least one distinct GPU ID from {list(range(available))}.")
    return selected


def select_cuda_devices(gpu_ids: Sequence[int] | None = None) -> tuple[str, ...]:
    """Select an ordered set of CUDA-visible devices for one inference run."""
    import torch

    selected = validate_gpu_ids(gpu_ids, torch.cuda.device_count())
    torch.cuda.set_device(selected[0])
    return tuple(f"cuda:{gpu_id}" for gpu_id in selected)
