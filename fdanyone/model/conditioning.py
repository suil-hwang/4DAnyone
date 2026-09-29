"""Frozen, timestep-independent conditioning used by the denoiser."""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fdanyone.errors import FourDAnyoneError
from fdanyone.views import VIEWS_PER_GROUP

if TYPE_CHECKING:
    from fdanyone.skeleton.pipeline import Conditioning, SkeletonVideo

LOGGER = logging.getLogger("fdanyone")

POSE_FEATURE_SHAPE = (3072, 31, 40, 22)
POSE_ENCODER_BATCH_LIMIT = 6


def load_prompt_context(path: str | Path):
    """Load the frozen UMT5 output consumed by the DiT."""

    from safetensors.torch import load_file

    return load_file(str(Path(path).expanduser().resolve()), device="cpu")["context"].contiguous()


@dataclass(frozen=True)
class PoseEncodingPlan:
    """Fixed-shape PoseEncoder batches for one ordered camera layout."""

    batch_size: int
    packed_views: int
    feature_groups: tuple[tuple[int, ...], ...]


def plan_pose_encoding(*, num_features: int, group_size: int, packed_views: int) -> PoseEncodingPlan:
    """Partition camera indices while reserving packed-view null inputs.

    Every PoseEncoder call uses the same batch shape.  This preserves the
    released CUDA convolution path while bounding full-resolution inputs to six
    videos.  Camera indices remain the only identity used after decoding.
    """

    batch_size = min(POSE_ENCODER_BATCH_LIMIT, group_size + packed_views)
    feature_capacity = batch_size - packed_views
    groups = tuple(
        tuple(range(start, min(start + feature_capacity, num_features)))
        for start in range(0, num_features, feature_capacity)
    )
    return PoseEncodingPlan(batch_size=batch_size, packed_views=packed_views, feature_groups=groups)


@dataclass(frozen=True)
class PoseFeatureBank:
    """Owned CPU BF16 features in canonical camera-index order."""

    features: Tensor
    null_features: Tensor

    @property
    def num_features(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_packed_views(self) -> int:
        return int(self.null_features.shape[0])

    def allocate_group(self, size: int, device: str) -> Tensor:
        """Allocate one reusable group buffer on the requested device."""

        return torch.empty((size, *POSE_FEATURE_SHAPE), dtype=self.features.dtype, device=device)

    def copy_group(self, indices: tuple[int, ...], destination: Tensor) -> None:
        """Copy canonical camera features into a reusable device buffer."""

        for output_index, feature_index in enumerate(indices):
            destination[output_index].copy_(self.features[feature_index])


@dataclass(frozen=True)
class PoseFeatureCache:
    """Pose features grouped by the RCP and target denoising layouts."""

    rcp: PoseFeatureBank | None
    target: PoseFeatureBank


@dataclass(frozen=True)
class _PoseEncodingJob:
    """One complete fixed-shape PoseEncoder call."""

    feature_indices: tuple[int, ...]
    skeletons: tuple[SkeletonVideo, ...]
    batch_size: int
    packed_views: int
    bank: PoseFeatureBank


def _pose_jobs(
    *,
    skeletons: tuple[SkeletonVideo, ...],
    group_size: int,
    packed_views: int,
) -> tuple[tuple[_PoseEncodingJob, ...], PoseFeatureBank]:
    """Plan complete fixed-shape calls and their canonical output owner."""

    plan = plan_pose_encoding(
        num_features=len(skeletons),
        group_size=group_size,
        packed_views=packed_views,
    )
    bank = PoseFeatureBank(
        features=torch.empty((len(skeletons), *POSE_FEATURE_SHAPE), dtype=torch.bfloat16, device="cpu"),
        null_features=torch.empty((packed_views, *POSE_FEATURE_SHAPE), dtype=torch.bfloat16, device="cpu"),
    )
    jobs = tuple(
        _PoseEncodingJob(
            feature_indices=indices,
            skeletons=tuple(skeletons[index] for index in indices),
            batch_size=plan.batch_size,
            packed_views=packed_views,
            bank=bank,
        )
        for indices in plan.feature_groups
    )
    return jobs, bank


def _encode_pose_jobs(
    jobs: tuple[_PoseEncodingJob, ...],
    pose_encoder,
    conditioning: Conditioning,
    device: str,
    stopped: Event,
) -> None:
    """Run independent fixed-shape batches on one worker's PoseEncoder."""

    torch.cuda.set_device(device)
    pose_encoder.to(device)
    prefix = pose_encoder.temporal_prefix
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for job in jobs:
            if stopped.is_set():
                break
            video_shape = None
            for batch_index, skeleton in enumerate(job.skeletons):
                LOGGER.info("Loading skeleton conditioning from %s", skeleton.path.name)
                decoded = conditioning.load_skeleton_tensor([skeleton]).to(dtype=torch.bfloat16, device="cpu").contiguous()
                current_shape = tuple(decoded.shape[1:])
                if video_shape is None:
                    video_shape = current_shape
                    batch = torch.full(
                        (job.batch_size, video_shape[0], video_shape[1] + prefix, *video_shape[2:]),
                        -1, dtype=torch.bfloat16, device=device,
                    )
                elif current_shape != video_shape:
                    raise FourDAnyoneError(f"Skeleton shape {current_shape} != {video_shape}: {skeleton.path}")
                # Prepend the first frame without retaining a second full batch.
                # Null and unused slots stay -1, including their temporal prefix.
                batch[batch_index, :, :prefix].copy_(decoded[0, :, :1])
                batch[batch_index, :, prefix:].copy_(decoded[0])
                del decoded

            encoded = pose_encoder(batch).detach().to(device="cpu").contiguous()
            for batch_index, feature_index in enumerate(job.feature_indices):
                job.bank.features[feature_index].copy_(encoded[batch_index])
            if job.feature_indices[0] == 0:
                null_start = len(job.feature_indices)
                job.bank.null_features.copy_(encoded[null_start : null_start + job.packed_views])
            del batch, encoded


def build_pose_feature_cache(
    *,
    conditioning: Conditioning,
    checkpoint_path: str | Path,
    devices: tuple[str, ...],
) -> PoseFeatureCache:
    """Encode all fixed-shape pose jobs on an ordered CUDA device pool."""

    from fdanyone.model.loader import load_pose_encoder

    view_plan = conditioning.view_plan

    rcp_jobs: tuple[_PoseEncodingJob, ...] = ()
    rcp_bank = None
    if view_plan.enable_rcp:
        rcp_jobs, rcp_bank = _pose_jobs(
            skeletons=conditioning.rcp_skeletons,
            group_size=VIEWS_PER_GROUP,
            packed_views=1,
        )
    target_jobs, target_bank = _pose_jobs(
        skeletons=conditioning.target_skeletons,
        group_size=VIEWS_PER_GROUP,
        packed_views=2 if view_plan.enable_rcp else 1,
    )
    jobs = (*rcp_jobs, *target_jobs)
    worker_count = min(len(jobs), len(devices))
    stopped = Event()
    pose_encoders = [load_pose_encoder(checkpoint_path, "cpu")]
    pose_encoders.extend(deepcopy(pose_encoders[0]) for _ in range(worker_count - 1))

    pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="pose")
    try:
        futures = [
            pool.submit(_encode_pose_jobs, jobs[index::worker_count], pose_encoders[index],
                        conditioning, devices[index], stopped)
            for index in range(worker_count)
        ]
        for future in as_completed(futures):
            future.result()
    finally:
        stopped.set()
        pool.shutdown(wait=True, cancel_futures=True)
        # Offload only after every worker has stopped using its device.
        for index, pose_encoder in enumerate(pose_encoders):
            torch.cuda.set_device(devices[index])
            pose_encoder.to("cpu")
            torch.cuda.empty_cache()
        pose_encoders.clear()
        gc.collect()
    return PoseFeatureCache(rcp=rcp_bank, target=target_bank)
