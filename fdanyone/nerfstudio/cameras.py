"""Camera validation and coordinate conversion for Nerfstudio exports."""

from __future__ import annotations

import numpy as np

from fdanyone.errors import FourDAnyoneError

# 4DAnyone uses a right-handed Y-up world. Nerfstudio uses a right-handed
# Z-up world, so rotate +Y onto +Z while keeping +X fixed.
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# OpenCV camera axes are +X right, +Y down, +Z forward. Nerfstudio follows
# OpenGL/Blender: +X right, +Y up, +Z back.
_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def camera_to_nerfstudio(camera_to_world: object) -> list[list[float]]:
    """Convert an OpenCV/Y-up camera-to-world matrix to OpenGL/Z-up."""

    matrix = np.asarray(camera_to_world, dtype=np.float64)
    converted = _Y_UP_TO_Z_UP @ matrix @ _OPENCV_TO_OPENGL
    return converted.tolist()


def camera_geometry(cameras: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-to-world and projection matrices for a validated rig."""

    intrinsics = []
    camera_to_worlds = []
    for camera in cameras:
        intrinsic = np.asarray(camera.get("K"), dtype=np.float64)
        camera_to_world = np.asarray(camera.get("camera_to_world"), dtype=np.float64)
        intrinsics.append(intrinsic)
        camera_to_worlds.append(camera_to_world)

    intrinsics_array = np.stack(intrinsics)
    camera_to_worlds_array = np.stack(camera_to_worlds)
    world_to_cameras = np.linalg.inv(camera_to_worlds_array)
    projections = intrinsics_array @ world_to_cameras[:, :3]
    return camera_to_worlds_array, projections


def visual_hull_center(camera_to_worlds: np.ndarray) -> np.ndarray:
    """Return the common look-at target of the generated camera rig."""

    centers = camera_to_worlds[:, :3, 3]
    directions = camera_to_worlds[:, :3, 2]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    system = projectors.sum(axis=0)
    if np.linalg.matrix_rank(system) < 3:
        raise FourDAnyoneError("Camera rig has no bounded visual-hull center.")
    target = np.linalg.solve(system, np.einsum("bij,bj->i", projectors, centers))
    return target


def points_to_nerfstudio(points: np.ndarray) -> np.ndarray:
    """Convert 4DAnyone Y-up XYZ points into Nerfstudio Z-up coordinates."""

    rotation = _Y_UP_TO_Z_UP[:3, :3].astype(np.float32)
    return points @ rotation.T
