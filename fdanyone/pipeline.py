# fdanyone/pipeline.py
from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fdanyone.assets import (
    CHECKPOINT,
    HF_REPO_ID,
    HF_REVISION,
    TURBO_LORA,
    TURBO_LORA_NAME,
    TURBO_LORA_SHA256,
    ensure_example_video,
    ensure_models,
    ensure_smplx,
    resolve_base_assets,
    resolve_checkpoint,
    resolve_foreground_model,
    resolve_regressor,
    resolve_turbo_lora,
)
from fdanyone.attention import get_attention_backend
from fdanyone.config import BASE24, INFERENCE, RANK64_DELTA4
from fdanyone.device import CUDA_ALLOCATOR_CONF, select_cuda_devices
from fdanyone.errors import ConfigurationError
from fdanyone.io import remove_tree, resolve_output_path, write_json
from fdanyone.motion.gvhmr import validate_gvhmr
from fdanyone.motion.result import MotionResult
from fdanyone.output import OutputDirectory, save_run_request, write_output
from fdanyone.video import (
    decode_canonical_clip,
    validate_clip_options,
    verify_lossless_video,
    write_gvhmr_video,
)
from fdanyone.views import resolve_view_plan

LOGGER = logging.getLogger("fdanyone")
PROGRESS = logging.getLogger("fdanyone.progress")
PROGRESS.addHandler(logging.NullHandler())
PROGRESS.propagate = False


def _run_worker(module: str, request_path: Path, request: dict) -> None:
    """Run a short-lived GVHMR worker on this checkout with stable CUDA flags."""

    environment = {
        **os.environ,
        "TORCH_CUDNN_V8_API_DISABLED": "1",
        "CUDNN_FRONTEND_DISABLE": "1",
        "CUDNN_LOGINFO_DBG": "0",
        "CUDNN_LOGDEST_DBG": "stderr",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
    }
    environment.pop("PYTHONHOME", None)
    # The 4 GiB allocator split is tuned for the DiT process, not these short-lived preprocessing workers.
    environment.pop(CUDA_ALLOCATOR_CONF, None)
    # The request lives in scratch, which run_pipeline removes with everything else.
    write_json(request_path, request)
    subprocess.run([os.path.abspath(sys.executable), "-m", module, str(request_path)], check=True, env=environment)


def run_pipeline(
    *,
    video_path: str,
    output_dir: str | None,
    model_dir: str,
    checkpoint_path: str | None,
    mhr70_regressor_path: str | None,
    gvhmr_root: str,
    gpu_ids: list[int] | None,
    attention_backend: str,
    start_time: float,
    target_fps: str | int | float,
    seed: int,
    views_per_layer: int,
    layer_pitches: list[int],
    start_yaw: int,
    yaw_span: int,
    enable_rcp: bool,
    enable_tcr: bool,
    enable_turbo: bool,
) -> dict:
    """Execute inference for one clip, retaining reusable motion."""

    request_options = locals().copy()
    pipeline_started = time.monotonic()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    PROGRESS.info("Checking input and settings", extra={"fraction": 0.02})
    if not isinstance(enable_turbo, bool):
        raise ConfigurationError(f"enable_turbo must be a boolean: {enable_turbo!r}.")
    denoising_profile = RANK64_DELTA4 if enable_turbo else BASE24
    view_plan = resolve_view_plan(
        views_per_layer=views_per_layer,
        layer_pitches=layer_pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        enable_rcp=enable_rcp,
        enable_tcr=enable_tcr,
    )
    canonical_fps = validate_clip_options(
        start_time=start_time,
        fps=None if str(target_fps).lower() == "auto" else target_fps,
    )
    clip_name = Path(video_path).stem
    destination = resolve_output_path(Path("data/fdanyone") / clip_name if output_dir is None else output_dir)
    output = OutputDirectory(destination)
    # Check before downloads and decoding, then again when locking the output.
    output.validate_available()
    devices = select_cuda_devices(gpu_ids)
    device = devices[0]

    from fdanyone.model.distributed import require_nccl, select_worker_devices

    # Only parallel denoising workers need NCCL; independent pose/VAE stages do not.
    if len(select_worker_devices(devices, view_plan.num_groups)) > 1:
        require_nccl()

    # Resolve once before any downloads; every DiT, replicas included, receives this concrete backend.
    attention_backend = get_attention_backend(attention_backend, device=device)
    LOGGER.info("Using attention backend: %s", attention_backend)

    PROGRESS.info("Preparing model assets", extra={"fraction": 0.05})
    ensure_example_video(video_path)
    # Licensed SMPL-X before the large model download: terminals install it interactively, background jobs fail fast.
    ensure_smplx(model_dir, gvhmr_root)
    ensure_models(model_dir, gvhmr_root)
    turbo_lora = resolve_turbo_lora(model_dir) if enable_turbo else None
    gvhmr_root, gvhmr_revision = validate_gvhmr(gvhmr_root)

    checkpoint = resolve_checkpoint(checkpoint_path, model_dir=model_dir)
    # Only the published checkpoint claims the frozen Hugging Face identity; a local override does not.
    if checkpoint_path is None:
        model_identity = {"checkpoint": CHECKPOINT, "repo_id": HF_REPO_ID, "revision": HF_REVISION}
    else:
        model_identity = {"checkpoint": checkpoint.name, "source": "local_override"}
    if turbo_lora is not None:
        model_identity["turbo_lora"] = {"name": TURBO_LORA_NAME, "file": TURBO_LORA, "sha256": TURBO_LORA_SHA256}
    base_assets = resolve_base_assets(model_dir)
    regressor = resolve_regressor(mhr70_regressor_path, model_dir=model_dir)
    foreground_model = resolve_foreground_model(model_dir)
    PROGRESS.info("Preparing the 121-frame clip", extra={"fraction": 0.10})
    clip = decode_canonical_clip(
        video_path,
        num_frames=INFERENCE.num_frames,
        start_time=start_time,
        fps=canonical_fps,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{clip_name}.scratch-", dir=destination.parent))
    try:
        clip_metadata = scratch / "canonical_clip.json"
        clip.write_metadata(clip_metadata)
        working_video = write_gvhmr_video(clip, scratch / "canonical_clip.mp4")

        with output.stage() as work:
            PROGRESS.info("Recovering human motion with GVHMR", extra={"fraction": 0.15})
            reuse_motion = output.motion_dir.exists()
            if reuse_motion:
                LOGGER.info("Reusing GVHMR motion from %s", output.motion_dir)
                motion = MotionResult.load(output.motion_dir)
            else:
                save_run_request(destination, request_options)
                motion_scratch = scratch / "gvhmr"
                _run_worker(
                    "fdanyone.motion.worker",
                    motion_scratch / ".motion-worker-request.json",
                    {
                        "gvhmr_root": str(gvhmr_root),
                        "working_video": str(working_video),
                        "clip_metadata": str(clip_metadata),
                        "output_dir": str(motion_scratch / "runtime"),
                        "result_dir": str(motion_scratch / "result"),
                        "device": device,
                    },
                )
                motion = MotionResult.load(motion_scratch / "result")
            if motion.gvhmr_revision != gvhmr_revision:
                raise ConfigurationError("GVHMR revision mismatch; use a new output directory.")
            motion.validate_against_clip(clip)
            if reuse_motion:
                save_run_request(destination, request_options)
            else:
                output.save_motion(motion)

            PROGRESS.info("Building foreground masks and skeletons", extra={"fraction": 0.30})
            conditioning_scratch = scratch / "conditioning"
            # Import the ~5 s generation stack only once motion is ready, overlapping the skeleton worker.
            with ThreadPoolExecutor(max_workers=1) as executor:
                generation_import = executor.submit(importlib.import_module, "fdanyone.model.inference")
                _run_worker(
                    "fdanyone.skeleton.worker",
                    scratch / ".skeleton-worker-request.json",
                    {
                        "working_video": str(working_video),
                        "clip_metadata": str(clip_metadata),
                        "motion_result_dir": str(output.motion_dir),
                        "regressor_path": str(regressor),
                        "foreground_model_path": str(foreground_model),
                        "gvhmr_root": str(gvhmr_root),
                        "output_dir": str(conditioning_scratch),
                        "device": device,
                        "view_plan": view_plan.to_dict(),
                    },
                )
                generation_import.result()
            from fdanyone.model.inference import generate_views
            from fdanyone.skeleton.pipeline import Conditioning

            conditioning = Conditioning.load(conditioning_scratch)
            # Re-decode the worker-produced source before it becomes a model tensor.
            verify_lossless_video(clip, conditioning.source_video)
            # Generation reads the verified working video; keep only the full-resolution clip's provenance.
            clip_info = clip.info
            del clip
            PROGRESS.info("Generating target-view videos", extra={"fraction": 0.45})
            generated = generate_views(
                conditioning=conditioning,
                checkpoint_path=checkpoint,
                turbo_lora_path=turbo_lora,
                denoising_profile=denoising_profile,
                attention_backend=attention_backend,
                assets=base_assets,
                output_dir=scratch / "generation",
                devices=devices,
                seed=seed,
            )
            PROGRESS.info("Saving and validating generated videos", extra={"fraction": 0.92})
            summary = write_output(
                clip=clip_info,
                conditioning=conditioning,
                generated=generated,
                destination=work,
                motion=motion,
                model_identity=model_identity,
                pipeline_started=pipeline_started,
            )
    finally:
        # Best effort: network filesystems may hold unlinked files busy (EBUSY) until this process exits.
        remove_tree(scratch, ignore_errors=True)
        if scratch.exists():
            LOGGER.warning(
                "Could not remove temporary files at %s. "
                "The result is unaffected; the hidden scratch directory can be removed after this process exits.",
                scratch,
            )
    summary["output_dir"] = str(destination)
    PROGRESS.info("Inference complete", extra={"fraction": 1.0})
    return summary
