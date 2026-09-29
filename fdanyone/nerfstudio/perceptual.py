"""VGG-19 perceptual loss for Nerfstudio reconstruction."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

# The reference objective reads these layers from the MatConvNet VGG-19
# checkpoint and returns features after conv{1,2,3,4,5}_2. The final two VGG
# convolutions and classifier are not evaluated by its forward pass.
VGG19_CONV_CHANNELS = (
    (3, 64),
    (64, 64),
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 256),
    (256, 256),
    (256, 256),
    (256, 512),
    (512, 512),
    (512, 512),
    (512, 512),
    (512, 512),
    (512, 512),
)
VGG19_FEATURE_LAYERS = (2, 4, 6, 10, 14)
VGG19_POOL_AFTER = frozenset((2, 4, 8, 12))
PERCEPTUAL_FEATURE_DIVISORS = (2.6, 4.8, 3.7, 5.6, 0.15)
PERCEPTUAL_COMPUTE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _autocast_context(compute_dtype: torch.dtype, device: torch.device) -> AbstractContextManager:
    if compute_dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=compute_dtype)


class MatConvNetVGG19Features(nn.Module):
    """Frozen MatConvNet VGG-19 features used by the perceptual objective.

    The weights are non-persistent buffers: they move with the Nerfstudio model
    but remain in the separately downloaded asset instead of being duplicated
    into every Splatfacto checkpoint.
    """

    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        path = Path(weights_path).expanduser().resolve()
        state = load_file(str(path), device="cpu")
        for index in range(1, len(VGG19_CONV_CHANNELS) + 1):
            self.register_buffer(
                f"conv{index}_weight",
                state[f"conv{index}.weight"].float().contiguous(),
                persistent=False,
            )
            self.register_buffer(
                f"conv{index}_bias",
                state[f"conv{index}.bias"].float().contiguous(),
                persistent=False,
            )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = []
        value = image.float().contiguous()
        for index in range(1, len(VGG19_CONV_CHANNELS) + 1):
            weight = getattr(self, f"conv{index}_weight")
            bias = getattr(self, f"conv{index}_bias")
            value = F.relu(F.conv2d(value, weight, bias, padding=1), inplace=False)
            if index in VGG19_FEATURE_LAYERS:
                features.append(value)
            if index in VGG19_POOL_AFTER:
                value = F.avg_pool2d(value, kernel_size=2, stride=2)
        return tuple(features)


def perceptual_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_features: tuple[torch.Tensor, ...],
    target_features: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Apply the reference pixel/feature weighting to scaled images."""

    terms = [torch.mean(torch.abs(target - prediction))]
    terms.extend(
        torch.mean(torch.abs(target_features[index] - prediction_features[index])) / divisor
        for index, divisor in enumerate(PERCEPTUAL_FEATURE_DIVISORS)
    )
    return torch.stack(terms).sum() / 255.0


class VGG19PerceptualLoss(nn.Module):
    """The reconstruction objective's VGG-19 loss, not standard LPIPS."""

    def __init__(self, weights_path: str | Path, compute_dtype: str = "float32") -> None:
        super().__init__()
        self.compute_dtype = PERCEPTUAL_COMPUTE_DTYPES[compute_dtype]
        self.features = MatConvNetVGG19Features(weights_path)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((123.6800, 116.7790, 103.9390), dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_scaled = prediction.float() * 255.0 - self.imagenet_mean
        target_scaled = target.float() * 255.0 - self.imagenet_mean
        with _autocast_context(self.compute_dtype, prediction.device):
            with torch.no_grad():
                target_features = self.features(target_scaled)
            prediction_features = self.features(prediction_scaled)
        return perceptual_distance(
            prediction_scaled,
            target_scaled,
            prediction_features,
            target_features,
        )
