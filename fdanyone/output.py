# fdanyone/output.py
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fdanyone.config import INFERENCE
from fdanyone.device import CUDA_ALLOCATOR_CONF
from fdanyone.errors import ConfigurationError, FourDAnyoneError
from fdanyone.io import (
    lock_output,
    remove_tree,
    resolve_output_path,
    sha256_file,
    write_json,
)

if TYPE_CHECKING:
    from fdanyone.model.inference import GeneratedViews
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import Conditioning
    from fdanyone.video import ClipInfo

LOGGER = logging.getLogger("fdanyone")
REQUEST_FILE = ".4danyone-request.json"
REQUEST_VERSION = 2
_GENERATED = ("cameras.json", "skeletons", "videos", "metadata.json")
_DIRECTORIES = {"gvhmr", "skeletons", "videos", ".inference"}
_REQUEST_TEMPORARY = re.compile(rf"\.{re.escape(REQUEST_FILE)}\.[0-9a-f]{{32}}\.tmp")


class OutputDirectory:
    """Stage output under an OS lock, retaining motion and unfinished work on failure."""

    def __init__(self, destination: str | Path):
        self.destination = resolve_output_path(destination)
        self.motion_dir = self.destination / "gvhmr"
        self.working = self.destination / ".inference"
        self._owner = self.working / ".4danyone"
        self._publishing = self.working / ".publishing"

    def validate_available(self) -> None:
        """Protect completed results and unrelated files before preparing a retry."""

        if not os.path.lexists(self.destination):
            return
        if self.destination.is_symlink() or not self.destination.is_dir():
            raise ConfigurationError(f"Output path is occupied: {self.destination}.")
        if os.path.lexists(self.destination / "metadata.json"):
            raise ConfigurationError(f"Completed output already exists: {self.destination}.")
        allowed = {"gvhmr", ".inference", REQUEST_FILE}
        allowed.update(path.name for path in self.destination.iterdir() if _REQUEST_TEMPORARY.fullmatch(path.name))
        read_run_request(self.destination)
        if self._owner.is_file() and self._publishing.is_file():
            allowed.update(_GENERATED[:-1])
        self._check_entries(self.destination, allowed)
        if self.working.exists():
            allowed = {*_GENERATED, "gvhmr", ".4danyone", ".publishing"} if self._owner.is_file() else set()
            self._check_entries(self.working, allowed)

    @staticmethod
    def _check_entries(directory: Path, allowed: set[str]) -> None:
        for path in directory.iterdir():
            expected_type = path.is_dir() if path.name in _DIRECTORIES else path.is_file()
            if path.name not in allowed or path.is_symlink() or not expected_type:
                raise ConfigurationError(f"Unrecognized output entry: {path}.")

    def save_motion(self, motion: MotionResult) -> None:
        """Publish the complete motion pair once, independently of generation."""

        if os.path.lexists(self.motion_dir):
            raise ConfigurationError(f"Motion output already exists: {self.motion_dir}.")
        staged = motion.save(self.working / "gvhmr")
        staged.rename(self.motion_dir)

    @contextmanager
    def stage(self):
        self.validate_available()
        self.destination.mkdir(parents=True, exist_ok=True)
        with lock_output(self.destination):
            self.validate_available()

            # Validation rejects symlinks and directories; cleanup runs under the
            # output lock and owns only UUID temporaries from write_json's protocol.
            for path in self.destination.iterdir():
                if _REQUEST_TEMPORARY.fullmatch(path.name):
                    path.unlink()
            # Only a marked, interrupted publication owns artifacts outside staging.
            if self._publishing.is_file():
                for name in _GENERATED[:-1]:
                    path = self.destination / name
                    if name in _DIRECTORIES:
                        remove_tree(path)
                    else:
                        path.unlink(missing_ok=True)
            self.working.mkdir(exist_ok=True)
            self._owner.touch(exist_ok=True)
            # Ownership survives an interruption in cleanup itself.
            for path in self.working.iterdir():
                if path == self._owner:
                    continue
                if path.is_dir():
                    remove_tree(path)
                else:
                    path.unlink()

            yield self.working

            self._check_entries(self.working, {*_GENERATED, ".4danyone"})
            if {path.name for path in self.working.iterdir()} != {*_GENERATED, ".4danyone"}:
                raise ConfigurationError("Cannot publish incomplete output.")
            read_output_metadata(self.working)
            self._check_entries(self.destination, {"gvhmr", ".inference", REQUEST_FILE})
            self._publishing.touch(exist_ok=False)
            for name in _GENERATED:
                (self.working / name).rename(self.destination / name)

            try:
                remove_tree(self.working)
            except OSError as exc:
                LOGGER.warning("Output is complete, but staging cleanup failed at %s: %s", self.working, exc)


def read_run_request(directory: str | Path) -> dict | None:
    """Read a request in the current format without changing its arguments."""

    path = Path(directory) / REQUEST_FILE
    if not path.exists() and not path.is_symlink():
        return None
    value = json.loads(path.read_text())
    if value["version"] != REQUEST_VERSION:
        raise ConfigurationError(f"Unsupported request version {value['version']}: {path}.")
    return value


def save_run_request(directory: str | Path, options: dict, **fields) -> dict:
    """Call while reserving the output; never place orchestration files in staging."""

    directory = Path(directory)
    previous = read_run_request(directory) or {}
    source = Path(options["video_path"]).resolve()
    stat = source.stat()
    identity = {"path": str(source), "filename": source.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    old_source = previous.get("source", {})
    identity["sha256"] = (
        old_source["sha256"]
        if old_source.get("sha256") and all(old_source.get(key) == value for key, value in identity.items())
        else sha256_file(source)
    )
    options = dict(options)
    for key in ("model_dir", "gvhmr_root", "checkpoint_path", "mhr70_regressor_path"):
        if options.get(key):
            options[key] = str(Path(options[key]).expanduser().resolve())
    value = {
        **previous,
        "version": REQUEST_VERSION,
        "created_at": previous.get("created_at", datetime.now(timezone.utc).isoformat()),
        "options": {**options, "video_path": str(source), "output_dir": str(directory.resolve())},
        "source": identity,
        **fields,
    }
    write_json(directory / REQUEST_FILE, value)
    return value


def target_video_path(camera_id: int) -> Path:
    """Return the relative publication path for a target camera."""

    return Path("videos") / f"{camera_id:02d}.mp4"


def write_output(
    *,
    clip: ClipInfo,
    conditioning: Conditioning,
    generated: GeneratedViews,
    destination: str | Path,
    motion: MotionResult,
    model_identity: dict,
    pipeline_started: float,
) -> dict:
    """Write target, skeleton, camera, and metadata artifacts into staging."""

    root = Path(destination).expanduser().resolve()
    attention_backend = generated.attention_backend
    view_plan = generated.view_plan
    fps = f"{clip.fps_num}/{clip.fps_den}"
    skeletons_root = root / "skeletons"
    (root / "videos").mkdir(parents=True, exist_ok=False)
    skeletons_root.mkdir(exist_ok=False)
    view_outputs = zip(
        range(view_plan.num_target_views),
        generated.target_videos,
        conditioning.target_skeletons,
        strict=True,
    )
    for camera_id, video_path, skeleton in view_outputs:
        shutil.copy2(video_path, root / target_video_path(camera_id))
        shutil.copy2(skeleton.path, skeletons_root / f"{camera_id:02d}.mp4")

    camera_payload = json.loads((conditioning.root / "cameras.json").read_text())
    conditioning_metadata = json.loads((conditioning.root / "metadata.json").read_text())
    total_elapsed = time.monotonic() - pipeline_started
    generation_metadata = {
        "seed": generated.seed,
        "view_plan": {
            **view_plan.to_dict(),
            "num_layers": view_plan.num_layers,
            "num_target_views": view_plan.num_target_views,
            "num_groups": view_plan.num_groups,
        },
        "attention_backend": attention_backend,
        "inference_steps": generated.denoising_profile.num_inference_steps,
        "denoising_profile": generated.denoising_profile.to_dict(),
        "elapsed_seconds": generated.elapsed_seconds,
        "stage_peak_vram_bytes": generated.stage_peak_vram_bytes,
        "total_elapsed_seconds": total_elapsed,
        "peak_vram_allocated_bytes": generated.peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": generated.peak_vram_reserved_bytes,
    }
    if generated.parallelism is not None:
        generation_metadata["parallelism"] = generated.parallelism

    import torch

    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "allocator_config": os.environ.get(CUDA_ALLOCATOR_CONF),
    }
    if cuda["available"]:
        torch_device = torch.device(generated.device)
        properties = torch.cuda.get_device_properties(torch_device)
        cuda.update(
            {
                "device": generated.device,
                "device_name": torch.cuda.get_device_name(torch_device),
                "device_capability": list(torch.cuda.get_device_capability(torch_device)),
                "device_total_memory_bytes": properties.total_memory,
            }
        )

    metadata = {
        "input": {
            "filename": clip.source_path.name,
            "fps": fps,
            "start_time_seconds": float(clip.start_time),
            "num_frames": clip.num_frames,
            "width": clip.width,
            "height": clip.height,
        },
        "motion": {
            "method": "GVHMR",
            "revision": motion.gvhmr_revision,
        },
        "preprocessing": {
            "source_crop_policy": conditioning_metadata["source_crop_policy"],
            "foreground_model": conditioning_metadata["foreground_model"],
            "framing": conditioning_metadata["framing"],
            "skeleton_draw_scale": conditioning_metadata["skeleton_draw_scale"],
        },
        "model": dict(model_identity),
        "generation": generation_metadata,
        "output": {
            "target_views": view_plan.num_target_views,
            "frames_per_video": INFERENCE.num_frames,
            "width": INFERENCE.width,
            "height": INFERENCE.height,
            "fps": fps,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": cuda,
        },
    }
    camera_rig = {
        "camera_model": "OPENCV",
        "world_frame": camera_payload["world_frame"],
        "camera_frame": camera_payload["camera_frame"],
        "front_camera_ids": camera_payload["front_camera_ids"],
        "framing": camera_payload["framing"],
        "cameras": [],
    }
    for camera in camera_payload["cameras"]:
        camera_id = int(camera["camera_id"])
        camera_rig["cameras"].append(
            {
                "camera_id": camera_id,
                "layer_index": int(camera["layer_index"]),
                "pitch": int(camera["pitch_degrees"]),
                "yaw": float(camera["yaw_degrees"]),
                "K": camera["K"],
                "camera_to_world": camera["camera_to_world"],
                "image_width": int(camera["image_width"]),
                "image_height": int(camera["image_height"]),
                "video": target_video_path(camera_id).as_posix(),
                "skeleton_video": f"skeletons/{camera_id:02d}.mp4",
            }
        )

    # Staging is private. Output publication, not each JSON write, owns the
    # commit boundary and interruption recovery.
    for name, payload in (
        ("cameras.json", camera_rig),
        ("metadata.json", metadata),
    ):
        (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "attention_backend": attention_backend,
        "num_target_videos": view_plan.num_target_views,
        "fps": fps,
        "peak_vram_allocated_bytes": generated.peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": generated.peak_vram_reserved_bytes,
        "total_pipeline_elapsed_seconds": total_elapsed,
    }


def read_output_metadata(directory: str | Path) -> dict:
    """Require the final commit marker before any reader consumes the output."""

    return json.loads((Path(directory) / "metadata.json").read_text())


def read_target_videos(root: Path, cameras: list[dict]) -> tuple[Path, ...]:
    """Read the canonical camera-indexed videos, contained within the result."""

    if not cameras or [camera.get("camera_id") for camera in cameras] != list(range(len(cameras))):
        raise FourDAnyoneError("Camera IDs must be consecutive and ordered.")
    layout = tuple(target_video_path(index) for index in range(len(cameras)))
    if any(
        not isinstance(camera.get("video"), str) or camera["video"].replace("\\", "/") != path.as_posix()
        for camera, path in zip(cameras, layout, strict=True)
    ):
        raise FourDAnyoneError("Video paths must match videos/<camera_id>.mp4.")

    root = root.resolve()
    paths = tuple(root / relative for relative in layout)
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise FourDAnyoneError(f"Missing or external result file: {path.relative_to(root)}")
    return paths
