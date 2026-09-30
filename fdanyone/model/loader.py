"""Direct, registry-free loading of the frozen Wan/SpaTem inference stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

from fdanyone.config import DenoisingProfile
from fdanyone.model.dit import MODEL_DIM, NUM_HEADS, FourDAnyoneDiT, precompute_freqs_cis_3d
from fdanyone.model.pose_encoder import PoseEncoder
from fdanyone.model.turbo_lora import fuse_turbo_lora, validate_turbo_base_metadata

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

POSE_ENCODER_PREFIX = "pose_encoder."


@dataclass
class Denoiser:
    """A loaded DiT, its denoising trajectory, and optional Turbo fusion."""

    model: nn.Module
    timesteps: Tensor
    sigma_deltas: Tensor
    dtype: torch.dtype
    turbo_lora_path: Path | None = None
    turbo_lora_applied: bool = field(default=False, init=False)

    def prepare_on_device(self, device: str | torch.device) -> None:
        """Move the DiT to a device and apply its configured Turbo delta once."""

        self.model.to(device=device, dtype=self.dtype)
        if self.turbo_lora_path is not None and not self.turbo_lora_applied:
            fuse_turbo_lora(self.model, self.turbo_lora_path)
            self.turbo_lora_applied = True


def _load_checkpoint(
    path: Path,
    *,
    include_prefix: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
):
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = checkpoint.keys()
        keys = (
            key
            for key in checkpoint_keys
            if (include_prefix is None or key.startswith(include_prefix))
            and not any(key.startswith(prefix) for prefix in exclude_prefixes)
        )
        return (
            {key: checkpoint.get_tensor(key) for key in keys},
            dict(checkpoint.metadata() or {}),
        )


def _load_dit(checkpoint_path: Path, attention_backend: str):
    with torch.device("meta"):
        dit = FourDAnyoneDiT(attention_backend=attention_backend)
    state_dict, metadata = _load_checkpoint(checkpoint_path, exclude_prefixes=(POSE_ENCODER_PREFIX,))
    dit.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    # ``freqs`` is a derived, non-persistent tensor and therefore is not in the
    # state dict populated above.
    dit.freqs = precompute_freqs_cis_3d(MODEL_DIM // NUM_HEADS)
    return dit.eval().requires_grad_(False), metadata


def load_pose_encoder(checkpoint_path: str | Path, device: str):
    """Load only the small pose encoder partition from the DiT checkpoint."""

    checkpoint, _ = _load_checkpoint(Path(checkpoint_path), include_prefix=POSE_ENCODER_PREFIX)
    state_dict = {key.removeprefix(POSE_ENCODER_PREFIX): value for key, value in checkpoint.items()}
    del checkpoint
    with torch.device("meta"):
        pose_encoder = PoseEncoder(out_dim=MODEL_DIM, in_channels=3)
    pose_encoder.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    return pose_encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def _load_vae(path: Path, dtype):
    from diffsynth.models.wan_video_vae import WanVideoVAE38

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = WanVideoVAE38.state_dict_converter().from_civitai(state_dict)
    # The external VAE also constructs normalization tensors outside its state dict.
    vae = WanVideoVAE38()
    vae.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    return vae.to(dtype=dtype).eval().requires_grad_(False)


def load_vae(path: str | Path):
    """Load the frozen Wan VAE as an independent generation stage."""

    return _load_vae(Path(path), torch.bfloat16)


def load_denoiser(
    *,
    checkpoint_path: str | Path,
    turbo_lora_path: str | Path | None,
    profile: DenoisingProfile,
    attention_backend: str,
) -> Denoiser:
    """Load one DiT and configure its denoising trajectory."""

    dtype = torch.bfloat16
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    turbo_path = None if turbo_lora_path is None else Path(turbo_lora_path).expanduser().resolve()
    model, metadata = _load_dit(checkpoint, attention_backend)
    if turbo_path is not None:
        validate_turbo_base_metadata(metadata)
    # Include the zero endpoint; indexed deltas can be reused across stages and groups.
    sigmas = torch.linspace(profile.denoising_strength, 0.0, profile.num_inference_steps + 1)
    shift = profile.scheduler_shift
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return Denoiser(
        model=model,
        timesteps=sigmas[:-1] * 1000,
        sigma_deltas=sigmas[1:] - sigmas[:-1],
        dtype=dtype,
        turbo_lora_path=turbo_path,
    )
