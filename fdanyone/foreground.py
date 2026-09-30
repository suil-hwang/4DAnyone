# fdanyone/foreground.py
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForImageSegmentation

from fdanyone.config import FOREGROUND


def predict_foreground_masks(
    frames: tuple[np.ndarray, ...],
    model_path: str | Path,
    device: str,
    *,
    batch_size: int = FOREGROUND.batch_size,
) -> np.ndarray:
    """Return full-raster 8-bit foreground masks for the canonical clip."""

    model = AutoModelForImageSegmentation.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = model.eval().half().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(FOREGROUND.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    masks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            images = [Image.fromarray(frame, mode="RGB") for frame in frames[start : start + batch_size]]
            inputs = torch.stack([transform(image) for image in images])
            predictions = model(inputs.to(device=device, dtype=torch.float16))[-1].sigmoid().cpu()
            for image, prediction in zip(images, predictions, strict=True):
                mask = to_pil_image(prediction).resize(image.size).convert("L")
                masks.append(np.array(mask))
    # Both callers run more GPU work next, and a failure ends their process.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(masks)
