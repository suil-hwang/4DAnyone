"""Export synchronized 4DAnyone frames as directly trainable Nerfstudio data."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import av
import numpy as np
from PIL import Image

from fdanyone.assets import resolve_foreground_model
from fdanyone.config import INFERENCE
from fdanyone.device import select_cuda_device
from fdanyone.download import ensure_foreground_model
from fdanyone.errors import ConfigurationError, FourDAnyoneError
from fdanyone.foreground import predict_foreground_masks
from fdanyone.io import AtomicResultDirectory, write_json
from fdanyone.nerfstudio.cameras import camera_to_nerfstudio
from fdanyone.nerfstudio.visual_hull import (
    NERFSTUDIO_POINT_CLOUD,
    build_sparse_point_cloud,
    write_sparse_point_cloud,
)
from fdanyone.output import read_output_metadata
from fdanyone.result_videos import read_target_videos

# Nerfstudio casts mask pixels directly to bool, so soft BiRefNet predictions
# must cross a real decision boundary before serialization.
NERFSTUDIO_MASK_THRESHOLD = 128


def _read_cameras(result: Path) -> dict:
    path = result / "cameras.json"
    return json.loads(path.read_text())


def _extract_frame(video_path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(video_path), mode="r") as container:
        streams = container.streams.video
        return next(
            frame.to_ndarray(format="rgb24")
            for index, frame in enumerate(container.decode(streams[0]))
            if index == frame_index
        )


def _validate_raster(image: np.ndarray, camera: dict) -> None:
    camera_id = int(camera["camera_id"])
    expected = (int(camera["image_height"]), int(camera["image_width"]), 3)
    if image.dtype != np.uint8 or image.shape != expected:
        raise FourDAnyoneError(f"Camera {camera_id}: expected uint8 {expected}, got {image.dtype} {image.shape}")


def _write_images(images: tuple[np.ndarray, ...], binary_masks: np.ndarray, root: Path) -> None:
    """Write RGBA inputs so Splatfacto supervises the background through alpha.

    A separate Nerfstudio ``mask_path`` excludes masked pixels from the loss. It
    therefore does not constrain Gaussians outside the silhouette. Keeping the
    foreground mask as alpha instead lets Splatfacto composite a fresh training
    background and penalize stray geometry everywhere in the image.
    """

    root.mkdir()
    for camera_id, image in enumerate(images):
        alpha = np.multiply(binary_masks[camera_id], 255, dtype=np.uint8)[..., None]
        rgba = np.concatenate([image, alpha], axis=2)
        Image.fromarray(rgba, mode="RGBA").save(root / f"{camera_id:02d}.png", format="PNG")


def _write_masks(masks: np.ndarray, root: Path) -> np.ndarray:
    root.mkdir()
    binary_masks = masks >= NERFSTUDIO_MASK_THRESHOLD
    for camera_id in range(len(masks)):
        binary = binary_masks[camera_id]
        Image.fromarray(np.multiply(binary, 255, dtype=np.uint8)).save(root / f"{camera_id:02d}.png", format="PNG")
    return binary_masks


def _transforms(cameras: list[dict]) -> dict:
    frames = []
    for camera in cameras:
        camera_id = int(camera["camera_id"])
        intrinsic = np.asarray(camera.get("K"), dtype=np.float64)
        width = int(camera["image_width"])
        height = int(camera["image_height"])
        frames.append(
            {
                "file_path": f"images/{camera_id:02d}.png",
                "fl_x": float(intrinsic[0, 0]),
                "fl_y": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "h": height,
                "w": width,
                "transform_matrix": camera_to_nerfstudio(camera.get("camera_to_world")),
            }
        )
    return {
        "camera_model": "OPENCV",
        "ply_file_path": NERFSTUDIO_POINT_CLOUD,
        "frames": frames,
    }


def export_nerfstudio(
    data_dir: str,
    output_dir: str | None = None,
    frame_index: int = 0,
    model_dir: str = "models",
    device: str = "cuda:0",
) -> dict:
    """Export one masked multi-view timestamp and visual hull for Nerfstudio.

    Args:
        data_dir: Completed 4DAnyone output directory for one clip.
        output_dir: Dataset directory. Defaults to data/ns_data/<clip>/frame_NNN.
        frame_index: Synchronized frame index from 0 through 120.
        model_dir: Model root containing (or receiving) the pinned BiRefNet files.
        device: CUDA device used for foreground segmentation.
    """

    if isinstance(frame_index, bool) or not isinstance(frame_index, Integral) or not 0 <= frame_index < INFERENCE.num_frames:
        raise ConfigurationError(
            f"frame_index must be an integer from 0 through {INFERENCE.num_frames - 1}: {frame_index!r}."
        )
    frame_index = int(frame_index)

    result = Path(data_dir).expanduser().resolve()
    read_output_metadata(result)
    cameras = _read_cameras(result)["cameras"]
    videos = read_target_videos(result, cameras)
    transforms = _transforms(cameras)

    if output_dir is None:
        destination = Path("data/ns_data") / result.name / f"frame_{frame_index:03d}"
    else:
        destination = Path(output_dir)
    atomic = AtomicResultDirectory(destination)
    destination = atomic.destination

    with atomic as work:
        images = tuple(_extract_frame(video, frame_index) for video in videos)
        for image, camera in zip(images, cameras, strict=True):
            _validate_raster(image, camera)

        device, _ = select_cuda_device(device)
        ensure_foreground_model(model_dir)
        foreground_model = resolve_foreground_model(model_dir)
        masks = predict_foreground_masks(images, foreground_model, device)
        binary_masks = _write_masks(masks, work / "masks")
        _write_images(images, binary_masks, work / "images")
        points, colors = build_sparse_point_cloud(images, binary_masks, cameras, device)
        write_sparse_point_cloud(work / NERFSTUDIO_POINT_CLOUD, points, colors)
        write_json(work / "transforms.json", transforms, sort_keys=False)

    return {
        "output_dir": str(destination),
        "frame_index": frame_index,
        "num_images": len(cameras),
        "num_masks": len(cameras),
        "num_points": len(points),
    }
