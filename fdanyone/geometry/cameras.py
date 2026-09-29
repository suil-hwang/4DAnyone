# fdanyone/geometry/cameras.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from fdanyone.config import CAMERA, FRAMING, CameraConfig

WORLD_FRAME = {
    "name": "canonical_human_world",
    "handedness": "right",
    "axes": {
        "x": "right in the source-facing ring camera (the subject's anatomical left)",
        "y": "up",
        "z": "front; the subject initially faces +z and the source-facing camera lies on the +z side",
    },
    "origin": "initial root projected to the ground plane",
    "units": "meters",
}

CAMERA_FRAME = {
    "name": "opencv_camera",
    "handedness": "right",
    "axes": {"x": "image right", "y": "image down", "z": "forward from camera into the scene"},
    "matrix_convention": "column vectors: x_camera = world_to_camera @ x_world_homogeneous",
    "intrinsics_convention": "pixels with origin at the top-left",
}


@dataclass(frozen=True)
class Camera:
    """Immutable OpenCV calibration and pose for one sampled view."""

    camera_id: int
    layer_index: int
    yaw_degrees: float
    azimuth_degrees: float
    pitch_degrees: float
    position: tuple[float, float, float]
    K: tuple[tuple[float, float, float], ...]
    world_to_camera: tuple[tuple[float, float, float, float], ...]
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    image_width: int
    image_height: int

    def to_dict(self) -> dict:
        return asdict(self)


def reference_intrinsics(
    image_height: int,
    image_width: int,
    max_render_height: int = 1280,
    *,
    focal_normalized: float = FRAMING.reference_focal_normalized,
) -> np.ndarray:
    """Reproduce the renderer's downscale-then-rescale intrinsic construction."""

    divisor = max(2, ceil(image_height / max_render_height))
    render_height = image_height // divisor
    render_width = image_width // divisor
    scale = image_height / render_height
    focal = focal_normalized * render_height
    intrinsic = np.array(
        [[focal, 0.0, render_width / 2.0], [0.0, focal, render_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsic[:2] *= scale
    return intrinsic


def look_at_opencv(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return column-vector world-to-camera (R, t) with OpenCV axes for positions [..., 3]."""

    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down, axis=-1, keepdims=True)
    rotation = np.stack([right, down, forward], axis=-2)
    return rotation, -(rotation @ position[..., None])[..., 0]


def project_points(points_world: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixels, signed camera depths, and a positive-depth, in-frame mask."""

    points = np.asarray(points_world, dtype=np.float64)
    w2c = np.asarray(camera.world_to_camera)
    K = np.asarray(camera.K)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    points_camera = points_h @ w2c[:3].T
    homogeneous = points_camera @ K.T
    xy = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-8)
    valid = (
        (points_camera[:, 2] > 0.0)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < camera.image_width)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < camera.image_height)
    )
    return xy.astype(np.float32), points_camera[:, 2].astype(np.float32), valid


def camera_ring(
    *,
    center: np.ndarray,
    front_direction: np.ndarray,
    K: np.ndarray,
    image_height: int,
    image_width: int,
    radius: float = FRAMING.reference_radius,
    target_height: float = FRAMING.reference_target_height,
    spec: CameraConfig = CAMERA,
    start_yaw_degrees: float = 0.0,
    yaw_span_degrees: float = 360.0,
    layer_index: int = 0,
    camera_id_offset: int = 0,
) -> tuple[Camera, ...]:
    """Sample a uniform yaw ring at a fixed pitch and horizontal XZ radius."""

    center = np.asarray(center, dtype=np.float64)
    front_direction = np.asarray(front_direction, dtype=np.float64)
    front_azimuth = np.arctan2(front_direction[2], front_direction[0])
    # Relative yaw zero faces the person's front; start_yaw chooses where the
    # first camera lies and IDs then advance uniformly through the span.
    azimuth_start = front_azimuth + np.pi + np.deg2rad(start_yaw_degrees)
    target = center.copy()
    target[1] = target_height
    intrinsic = tuple(map(tuple, np.asarray(K, dtype=np.float64).tolist()))

    fractions = np.arange(spec.count) / spec.count
    azimuths = azimuth_start + fractions * np.deg2rad(yaw_span_degrees)
    # Cameras sit `radius` out horizontally from the target and `radius * tan(pitch)` above it.
    slope = np.tan(np.deg2rad(spec.pitch_degrees))
    positions = target + radius * np.column_stack([np.cos(azimuths), np.full(spec.count, slope), np.sin(azimuths)])
    w2c = np.zeros((spec.count, 4, 4))
    w2c[:, :3, :3], w2c[:, :3, 3] = look_at_opencv(positions, target)
    w2c[:, 3, 3] = 1.0
    c2w = np.linalg.inv(w2c)
    return tuple(
        Camera(
            camera_id=camera_id_offset + view_index,
            layer_index=layer_index,
            yaw_degrees=float(start_yaw_degrees + fractions[view_index] * yaw_span_degrees),
            azimuth_degrees=float(np.rad2deg(azimuths[view_index]) % 360.0),
            pitch_degrees=spec.pitch_degrees,
            position=tuple(positions[view_index].tolist()),
            K=intrinsic,
            world_to_camera=tuple(map(tuple, w2c[view_index].tolist())),
            camera_to_world=tuple(map(tuple, c2w[view_index].tolist())),
            image_width=image_width,
            image_height=image_height,
        )
        for view_index in range(spec.count)
    )


def camera_grid(
    *,
    center: np.ndarray,
    front_direction: np.ndarray,
    K: np.ndarray,
    image_height: int,
    image_width: int,
    views_per_layer: int,
    layer_pitches: tuple[int, ...],
    start_yaw: int,
    yaw_span: int,
    radius: float = FRAMING.reference_radius,
    target_height: float = FRAMING.reference_target_height,
) -> tuple[Camera, ...]:
    """Create a layer-major target camera grid."""

    return tuple(
        camera
        for layer_index, pitch in enumerate(layer_pitches)
        for camera in camera_ring(
            center=center,
            front_direction=front_direction,
            K=K,
            image_height=image_height,
            image_width=image_width,
            radius=radius,
            target_height=target_height,
            spec=CameraConfig(count=views_per_layer, pitch_degrees=float(pitch)),
            start_yaw_degrees=float(start_yaw),
            yaw_span_degrees=float(yaw_span),
            layer_index=layer_index,
            camera_id_offset=layer_index * views_per_layer,
        )
    )
