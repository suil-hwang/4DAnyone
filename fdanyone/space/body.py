"""Prepare reusable SMPL-X geometry on the CPU in an isolated process."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from fdanyone.assets import SMPLX_MODEL, resolve_regressor
from fdanyone.errors import FourDAnyoneError


def body_cache_identity(motion_dir, model_dir, gvhmr_root, regressor_path=None):
    """Use the same geometry inputs for body, recording and overlay caches."""
    def identity(path):
        if path is None or not path.is_file():
            return None
        stat = path.stat()
        return str(path.resolve()), stat.st_size, stat.st_mtime_ns

    files = [motion_dir / name for name in ("motion.json", "motion.safetensors")]
    if model_dir is not None:
        files.extend([model_dir / SMPLX_MODEL, resolve_regressor(regressor_path, model_dir)])
    if gvhmr_root is not None:
        files.extend(gvhmr_root / name for name in (
            "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
            "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
            "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",
            "hmr4d/utils/body_model/__init__.py",
            "hmr4d/utils/body_model/body_model_smplx.py",
            "hmr4d/utils/geo_transform.py",
            "hmr4d/utils/smplx_utils.py",
        ))
    return [2, str(gvhmr_root.resolve()) if gvhmr_root else None, [identity(path) for path in files]]


def source_transforms(points_incam: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """Fit the rigid source-camera pose between two versions of the same body."""

    source = np.asarray(points_incam, dtype=np.float64)
    target = np.asarray(points_world, dtype=np.float64)
    source_mean = source.mean(axis=1)
    target_mean = target.mean(axis=1)
    covariance = np.swapaxes(source - source_mean[:, None], 1, 2) @ (target - target_mean[:, None])
    left, _, right = np.linalg.svd(covariance)
    correction = np.broadcast_to(np.eye(3), covariance.shape).copy()
    correction[:, 2, 2] = np.linalg.det(left @ right)
    row_rotation = left @ correction @ right
    transforms = np.broadcast_to(np.eye(4), (len(source), 4, 4)).copy()
    transforms[:, :3, :3] = np.swapaxes(row_rotation, 1, 2)
    transforms[:, :3, 3] = target_mean - np.einsum("fi,fij->fj", source_mean, row_rotation)
    return transforms


def prepare_body(request: dict) -> None:
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry

    motion = MotionResult.load(request["motion_dir"])
    geometry = _body_geometry(
        motion,
        resolve_regressor(request.get("regressor_path"), request["model_dir"]),
        Path(request["gvhmr_root"]),
        "cpu",
        include_mesh=True,
    )
    destination = Path(request["destination"])
    temporary = destination.with_name(f".{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(
            temporary,
            vertices=geometry.mesh_world,
            faces=geometry.mesh_faces,
            keypoints=geometry.keypoints_world,
            source_camera_to_world=source_transforms(geometry.keypoints_incam, geometry.keypoints_world),
            intrinsics=motion.K_fullimg.numpy(),
            source_indices=np.asarray(motion.source_frame_indices),
            source_size=np.asarray([motion.image_width, motion.image_height]),
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_body(
    motion_dir: Path, model_dir: Path, gvhmr_root: Path, cache_dir: Path, check_cancelled,
    *, regressor_path: Path | None = None,
) -> dict:
    regressor_path = resolve_regressor(regressor_path, model_dir)
    files = [
        motion_dir / "motion.json",
        motion_dir / "motion.safetensors",
        model_dir / SMPLX_MODEL,
        regressor_path,
    ]
    metadata = json.loads((motion_dir / "motion.json").read_text())
    if metadata.get("tensor_file") != "motion.safetensors" or any(
        not path.resolve().is_relative_to(motion_dir.resolve()) for path in files[:2]
    ):
        raise FourDAnyoneError("Motion cache requires local motion.json and motion.safetensors.")
    for path in files:
        path.stat()
    identity = body_cache_identity(motion_dir, model_dir, gvhmr_root, regressor_path)
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    directory = cache_dir / "body" / key
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "body.npz"
    if not destination.is_file():
        from fdanyone.space.jobs import worker_process

        request = directory / "request.json"
        request.write_text(
            json.dumps(
                {
                    "motion_dir": str(motion_dir),
                    "model_dir": str(model_dir),
                    "regressor_path": str(regressor_path),
                    "gvhmr_root": str(gvhmr_root),
                    "destination": str(destination),
                }
            )
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        environment["PYTHONIOENCODING"] = "utf-8"
        environment.setdefault("OMP_NUM_THREADS", "8")
        with (directory / "body.log").open("w", encoding="utf-8") as log:
            with worker_process(
                [sys.executable, "-m", "fdanyone.space.body", str(request)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            ) as process:
                while process.poll() is None:
                    check_cancelled()
                    time.sleep(0.1)
                if process.returncode:
                    raise FourDAnyoneError(f"Body preview failed: {directory / 'body.log'}")
    with np.load(destination, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


if __name__ == "__main__":
    prepare_body(json.loads(Path(sys.argv[1]).read_text()))
