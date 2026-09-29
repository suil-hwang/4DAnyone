"""CUDA device selection shared by pipeline and isolated workers."""

from __future__ import annotations

import os
import subprocess
from collections.abc import MutableMapping, Sequence

from fdanyone.errors import ConfigurationError

CUDA_ALLOCATOR_CONF = "PYTORCH_CUDA_ALLOC_CONF"
CUDA_MAX_SPLIT_SIZE_MB = 4096
LOW_MEMORY_GPU_MAX_BYTES = 24 * 1024**3


def _visible_gpu_identifiers(environment: MutableMapping[str, str]) -> tuple[str, ...] | None:
    """Return CUDA-visible NVIDIA identifiers without initializing CUDA."""

    raw = environment.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        raw = environment.get("NVIDIA_VISIBLE_DEVICES")
    if raw is None or raw.strip().lower() == "all":
        return None
    if raw.strip().lower() in {"", "-1", "none", "void"}:
        return ()
    return tuple(identifier.strip() for identifier in raw.split(",") if identifier.strip())


def _selected_gpu_identifiers(
    gpu_ids: Sequence[int] | None,
    environment: MutableMapping[str, str],
) -> tuple[str, ...] | None:
    """Map CUDA-visible indices to identifiers accepted by ``nvidia-smi``."""

    visible = _visible_gpu_identifiers(environment)
    if gpu_ids is None:
        return visible
    if (
        isinstance(gpu_ids, (str, bytes))
        or not isinstance(gpu_ids, Sequence)
        or not gpu_ids
        or any(isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids)
    ):
        return ()
    if visible is None:
        return tuple(str(gpu_id) for gpu_id in gpu_ids)
    if any(gpu_id >= len(visible) for gpu_id in gpu_ids):
        return ()
    return tuple(visible[gpu_id] for gpu_id in gpu_ids)


def _query_total_memory_bytes(identifiers: tuple[str, ...] | None) -> tuple[int, ...]:
    """Query selected GPU capacities through the driver utility, before PyTorch."""

    if identifiers == ():
        return ()
    command = ["nvidia-smi"]
    if identifiers is not None:
        command.append(f"--id={','.join(identifiers)}")
    command.extend(("--query-gpu=memory.total", "--format=csv,noheader,nounits"))
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        totals_mib = tuple(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return ()
    return tuple(total_mib * 1024**2 for total_mib in totals_mib)


def has_low_memory_gpu(
    gpu_ids: Sequence[int] | None,
    environment: MutableMapping[str, str] | None = None,
) -> bool:
    """Return whether any selected GPU has at most 24 GiB of device memory."""

    environment = os.environ if environment is None else environment
    identifiers = _selected_gpu_identifiers(gpu_ids, environment)
    totals = _query_total_memory_bytes(identifiers)
    return any(total <= LOW_MEMORY_GPU_MAX_BYTES for total in totals)


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
    if not selected:
        raise ConfigurationError("Select at least one available CUDA device.")
    if any(
        isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or not 0 <= gpu_id < available
        for gpu_id in selected
    ):
        raise ConfigurationError(f"Invalid gpu_ids: {selected}; available IDs: 0..{available - 1}.")
    if len(set(selected)) != len(selected):
        raise ConfigurationError(f"Duplicate gpu_ids: {selected}.")
    return selected


def select_cuda_devices(gpu_ids: Sequence[int] | None = None) -> tuple[str, ...]:
    """Select an ordered set of CUDA-visible devices for one inference run."""
    import torch

    selected = validate_gpu_ids(gpu_ids, torch.cuda.device_count() if torch.cuda.is_available() else 0)
    torch.cuda.set_device(selected[0])
    return tuple(f"cuda:{gpu_id}" for gpu_id in selected)
