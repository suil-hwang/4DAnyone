"""Recover static-camera motion with the pinned GVHMR inference pipeline."""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import chdir, closing, contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

from fdanyone.errors import AssetError, VideoContractError
from fdanyone.motion.result import SMPL_PARAMETER_NAMES, MotionResult
from fdanyone.video import CanonicalClip

GVHMR_ASSETS = (
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
    "inputs/checkpoints/yolo/yolov8x.pt",
    "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
)


def validate_gvhmr(root: str | Path) -> tuple[Path, str]:
    """Locate the GVHMR checkout and the model files consumed by inference."""

    path = Path(root).expanduser().resolve()
    missing = [name for name in ("hmr4d/__init__.py", *GVHMR_ASSETS) if not (path / name).is_file()]
    if missing:
        raise AssetError(f"Missing GVHMR files in {path}: {', '.join(missing)}")
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
    return path, revision


def _unavailable_optional_feature(*args, **kwargs):
    raise NotImplementedError("Wis3D and moving-camera inference require their optional dependencies.")


@contextmanager
def gvhmr_imports(root: Path) -> Iterator[None]:
    """Import unmodified GVHMR with the geometry APIs needed by static inference."""

    root_text = str(root)
    with chdir(root):
        path_added = root_text not in sys.path
        if path_added:
            sys.path.insert(0, root_text)
        try:
            if "pytorch3d" not in sys.modules and importlib.util.find_spec("pytorch3d") is None:
                import roma

                from fdanyone.geometry import ops

                package = ModuleType("pytorch3d")
                package.transforms = ModuleType("pytorch3d.transforms")
                package.transforms.__dict__.update(
                    axis_angle_to_matrix=roma.rotvec_to_rotmat,
                    matrix_to_axis_angle=roma.rotmat_to_rotvec,
                    so3_exp_map=roma.rotvec_to_rotmat,
                    so3_log_map=roma.rotmat_to_rotvec,
                    matrix_to_rotation_6d=ops.matrix_to_rotation_6d,
                    rotation_6d_to_matrix=ops.rotation_6d_to_matrix,
                    matrix_to_quaternion=ops.matrix_to_quaternion,
                    quaternion_to_matrix=ops.quaternion_to_matrix,
                    quaternion_to_axis_angle=ops.quaternion_to_axis_angle,
                    euler_angles_to_matrix=ops.euler_angles_to_matrix,
                )
                package.transforms.__spec__ = ModuleSpec("pytorch3d.transforms", loader=None)
                package.ops = ModuleType("pytorch3d.ops")
                package.ops.knn = ops
                for module in (package, package.ops):
                    module.__path__ = []
                    module.__spec__ = ModuleSpec(module.__name__, loader=None, is_package=True)
                sys.modules.update({
                    "pytorch3d": package,
                    "pytorch3d.transforms": package.transforms,
                    "pytorch3d.ops": package.ops,
                    "pytorch3d.ops.knn": ops,
                })
            for dependency, name, exports in (
                ("wis3d", "hmr4d.utils.wis3d_utils", ("make_wis3d", "add_motion_as_lines")),
                ("pycolmap", "hmr4d.utils.preproc.relpose.simple_vo", ("SimpleVO",)),
            ):
                if name not in sys.modules and importlib.util.find_spec(dependency) is None:
                    module = ModuleType(name)
                    module.__dict__.update(dict.fromkeys(exports, _unavailable_optional_feature))
                    module.__spec__ = ModuleSpec(name, loader=None)
                    sys.modules[name] = module
            yield
        finally:
            if path_added:
                sys.path.remove(root_text)


def _track_video(tracker, video_path: Path, device: str):
    """Keep upstream's indexless YOLO request on the worker's selected GPU."""

    import torch

    requested_device = torch.device(device)
    original_track = tracker.yolo.track

    def track_on_device(*args, **kwargs):
        # A torch.device also avoids CUDA_VISIBLE_DEVICES remapping in older
        # Ultralytics versions. Upstream's explicit device="cuda" must lose.
        kwargs["device"] = requested_device
        return original_track(*args, **kwargs)

    tracker.yolo.track = track_on_device
    try:
        return tracker.track(video_path)
    finally:
        tracker.yolo.track = original_track


def run_gvhmr(
    *,
    clip: CanonicalClip,
    working_video: str | Path,
    output_dir: str | Path,
    gvhmr_root: str | Path,
    device: str,
) -> MotionResult:
    """Recover static-camera human motion from the canonical source clip."""

    root, revision = validate_gvhmr(gvhmr_root)
    working_video = Path(working_video).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    length = len(clip.frames)

    with gvhmr_imports(root):
        import hydra
        import numpy as np
        import torch
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.net_utils import detach_to_cpu, moving_average_smooth
        from hmr4d.utils.preproc.tracker import Tracker
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor, get_batch
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        from hmr4d.utils.pylogger import Log, ch
        from hmr4d.utils.seq_utils import (
            frame_id_to_mask,
            get_frame_id_list_from_mask,
            linear_interpolate_frame_ids,
            rearrange_by_mask,
        )
        from hmr4d.utils.video_io_utils import get_video_reader
        from hydra import compose, initialize_config_module
        from omegaconf import open_dict

        # Import only the model groups referenced by the static-camera config.
        for module in (
            "hmr4d.model.gvhmr.utils.endecoder",
            "hmr4d.network.gvhmr.relative_transformer",
        ):
            importlib.import_module(module)
        if ch in Log.handlers and len(Log.handlers) > 1:
            Log.removeHandler(ch)

        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            cfg = compose(
                config_name="demo",
                overrides=[
                    f"video_name={json.dumps(working_video.stem, ensure_ascii=False)}",
                    "static_cam=true",
                    "verbose=false",
                    "use_dpvo=false",
                    f"output_root={json.dumps(output_root.as_posix(), ensure_ascii=False)}",
                ],
            )
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
        with open_dict(cfg):
            cfg.video_path = str(working_video)

        # Verify the reader before reusing the canonical clip's dimensions.
        with closing(get_video_reader(str(working_video))) as reader:
            for index, (actual, expected) in enumerate(itertools.zip_longest(reader, clip.rgb_frames)):
                if actual is None or expected is None or not np.array_equal(actual, expected):
                    raise VideoContractError(f"GVHMR frame mismatch at index {index}.")

        Log.info("[Preprocess] Start!")
        started = Log.time()
        paths = cfg.paths
        if not Path(paths.bbx).exists():
            tracker = Tracker()
            frame_ids, boxes, track_ids = tracker.sort_track_length(
                _track_video(tracker, working_video, device), working_video
            )
            if track_ids:
                track_id = track_ids[0]
                mask = frame_id_to_mask(torch.tensor(frame_ids[track_id]), length)
                bbx_xyxy = rearrange_by_mask(torch.tensor(boxes[track_id]), mask)
                bbx_xyxy = linear_interpolate_frame_ids(bbx_xyxy, get_frame_id_list_from_mask(~mask))
                for _ in range(2):
                    bbx_xyxy = moving_average_smooth(bbx_xyxy, window_size=5, dim=0)
            else:
                bbx_xyxy = torch.tensor([0.0, 0.0, clip.width, clip.height]).repeat(length, 1)
                Log.warning("No person track found; using a full-frame bbox for %s.", working_video)
            bbx_xyxy = bbx_xyxy.float()
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
            torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
            del tracker
        else:
            bbx_xys = torch.load(paths.bbx, weights_only=True)["bbx_xys"]
            Log.info("[Preprocess] bbx from %s", paths.bbx)

        vitpose_cached = Path(paths.vitpose).exists()
        features_cached = Path(paths.vit_features).exists()
        cropped_images = cropped_bbx_xys = None
        if not vitpose_cached or not features_cached:
            # Both upstream extractors use the same normalized 256px crops.
            # Preserve get_batch's returned boxes for ViTPose's image-space
            # keypoints; the motion model still receives the original boxes.
            cropped_images, cropped_bbx_xys = get_batch(str(working_video), bbx_xys)

        if not vitpose_cached:
            extractor = VitPoseExtractor()
            torch.save(extractor.extract(cropped_images, cropped_bbx_xys), paths.vitpose)
            del extractor
        else:
            Log.info("[Preprocess] vitpose from %s", paths.vitpose)

        if not features_cached:
            extractor = Extractor()
            torch.save(extractor.extract_video_features(cropped_images, cropped_bbx_xys), paths.vit_features)
            del extractor
        else:
            Log.info("[Preprocess] vit_features from %s", paths.vit_features)
        del cropped_images, cropped_bbx_xys
        Log.info("[Preprocess] End. Time elapsed: %.2fs", Log.time() - started)

        data = {
            "length": torch.tensor(length),
            "bbx_xys": bbx_xys,
            "kp2d": torch.load(paths.vitpose, weights_only=True),
            "K_fullimg": estimate_K(clip.width, clip.height).repeat(length, 1, 1),
            "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(length, 1, 1)),
            "f_imgseq": torch.load(paths.vit_features, weights_only=True),
        }
        observed_keypoints_2d = data["kp2d"].detach().cpu()
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().to(device)
        with torch.inference_mode():
            prediction = detach_to_cpu(model.predict(data, static_cam=True))
        del model, data
        torch.cuda.empty_cache()

    result = MotionResult(
        gvhmr_revision=revision,
        fps=clip.fps,
        frame_timestamps_sec=tuple(float(frame.canonical_timestamp) for frame in clip.frames),
        source_frame_indices=tuple(frame.source_index for frame in clip.frames),
        source_pts=tuple(frame.source_pts for frame in clip.frames),
        source_size_bytes=clip.source_size_bytes,
        source_mtime_ns=clip.source_mtime_ns,
        image_height=clip.height,
        image_width=clip.width,
        smpl_params_global={name: prediction["smpl_params_global"][name] for name in SMPL_PARAMETER_NAMES},
        smpl_params_incam={name: prediction["smpl_params_incam"][name] for name in SMPL_PARAMETER_NAMES},
        K_fullimg=prediction["K_fullimg"],
        observed_keypoints_2d=observed_keypoints_2d,
    )
    return result
