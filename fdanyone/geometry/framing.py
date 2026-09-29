# fdanyone/geometry/framing.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from fdanyone.config import FRAMING, FramingConfig
from fdanyone.geometry.cameras import Camera

FULL_BODY_BOTTOM = 0.82
CLOSE_UP_BOTTOM = 0.35
HALF_BODY_BOTTOM = 0.42
_ANATOMY_COORDS = (0.0, 0.18, 0.45, 0.72, 0.94, 1.0)
_FINGER_TOKENS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class InputFraming:
    label: str
    visible_body_bottom: float
    visible_body_bottom_p50: float
    visible_body_bottom_p80: float
    visible_height_ratio: float
    wrist_out_ratio_x: float
    wrist_out_ratio_y: float
    valid_frame_ratio: float
    torso_valid_ratio: float
    projection_alignment_error_ratio: float | None
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "fmask_used": True}


@dataclass(frozen=True)
class AdaptiveThresholds:
    closeup_strength: float
    height_target_ratio: float
    height_percentile: float
    width_target_ratio: float
    width_percentile: float


@dataclass(frozen=True)
class RadiusSolve:
    radius: float
    height_ratio: float
    width_ratio: float
    bound: str | None
    limiting_constraint: str


@dataclass(frozen=True)
class FocalSolve:
    focal_normalized: float
    height_ratio: float
    width_ratio: float
    bound: str | None
    limiting_constraint: str


@dataclass(frozen=True)
class ClipFraming:
    radius: float
    target_height: float
    focal_normalized: float
    input: InputFraming
    input_applied: bool
    radius_solve: RadiusSolve
    adaptive_thresholds: AdaptiveThresholds | None = None
    focal_solve: FocalSolve | None = None
    cutoff_ratio: float | None = None
    target_bound: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "sequence_input_profile_static_radius_target_focal",
            "radius": self.radius,
            "target_height": self.target_height,
            "focal_normalized": self.focal_normalized,
            "input_framing_applied": self.input_applied,
            "input_framing": self.input.to_dict(),
            "adaptive_thresholds": (None if self.adaptive_thresholds is None else asdict(self.adaptive_thresholds)),
            "radius_solver": asdict(self.radius_solve),
            "focal_solver": None if self.focal_solve is None else asdict(self.focal_solve),
            "cutoff_ratio": self.cutoff_ratio,
            "target_bound": self.target_bound,
        }


def anatomy_samples(
    keypoints: np.ndarray,
    names: Sequence[str],
    samples_per_segment: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample both head-to-foot paths from named keypoints [frames,keypoints,3]."""
    points = np.asarray(keypoints, dtype=np.float64)
    ids = {str(name).strip().lower().replace("_", "-"): index for index, name in enumerate(names)}
    face = np.mean(
        points[:, [ids[name] for name in ("nose", "left-eye", "right-eye", "left-ear", "right-ear")]], axis=1
    )
    head_top = face + 0.65 * (face - points[:, ids["neck"]])
    weights = np.linspace(0.0, 1.0, samples_per_segment + 1)
    anatomical = np.asarray(_ANATOMY_COORDS)
    coordinates = anatomical[:-1, None] * (1.0 - weights) + anatomical[1:, None] * weights
    coordinates = np.concatenate([coordinates[:, :-1].ravel(), coordinates[-1, -1:]])
    paths = []
    for side in ("left", "right"):
        joints = [points[:, ids[f"{side}-{part}"]] for part in ("shoulder", "hip", "knee", "ankle")]
        foot = points[:, [ids[f"{side}-{part}"] for part in ("big-toe-tip", "small-toe-tip", "heel")]].mean(axis=1)
        anchors = np.stack([head_top, *joints, foot], axis=1)
        segments = anchors[:, :-1, None] * (1.0 - weights[:, None]) + anchors[:, 1:, None] * weights[:, None]
        # Include each joint once, retaining the final endpoint of each path.
        path = segments[:, :, :-1].reshape(len(points), 5 * samples_per_segment, 3)
        paths.append(np.concatenate([path, segments[:, -1, -1:]], axis=1))
    return np.concatenate(paths, axis=1), np.tile(coordinates, 2)


def project_incam(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-space points [frames,keypoints,3] with shared or per-frame K."""
    points = np.asarray(points, dtype=np.float64)
    cameras = np.broadcast_to(np.asarray(intrinsics, dtype=np.float64), (points.shape[0], 3, 3))
    homogeneous = np.einsum("fij,fkj->fki", cameras, points)
    depths = points[..., 2]
    xy = homogeneous[..., :2] / np.maximum(homogeneous[..., 2:3], 1e-8)
    return xy, depths


def analyze_input_framing(
    incam_keypoints: np.ndarray,
    names: Sequence[str],
    intrinsics: np.ndarray,
    vitpose: np.ndarray,
    masks: np.ndarray,
) -> InputFraming:
    """Estimate visible anatomy using foreground masks and reliable COCO joints."""
    masks = np.asarray(masks)
    frame_count, height, width = masks.shape
    detector = np.asarray(vitpose, dtype=np.float64)
    detected = (
        (detector[..., 2] >= 0.3)
        & (detector[..., 0] >= 0) & (detector[..., 0] < width)
        & (detector[..., 1] >= 0) & (detector[..., 1] < height)
    )
    shoulders = detected[:, 5] & detected[:, 6]
    torso_valid = shoulders & (detected[:, 11] | detected[:, 12])
    if torso_valid.mean() < 0.5:
        torso_valid = shoulders

    anatomy, coordinates = anatomy_samples(incam_keypoints, names)
    xy, depths = project_incam(anatomy, intrinsics)
    inside = (
        (depths > 1e-6)
        & (xy[..., 0] >= 0) & (xy[..., 0] < width)
        & (xy[..., 1] >= 0) & (xy[..., 1] < height)
    )
    patch_radius = max(3, round(min(width, height) * 0.005))
    kernel = np.ones((2 * patch_radius + 1,) * 2, dtype=np.uint8)
    bottoms, visible_heights = [], []
    for frame in np.flatnonzero(torso_valid & inside.any(axis=1)):
        candidates = np.flatnonzero(inside[frame])
        pixels = np.rint(xy[frame, candidates]).astype(np.int64)
        pixels = np.clip(pixels, 0, [width - 1, height - 1])
        # Dilation is a window max, so thresholding afterwards matches dilating the binary mask.
        dilated = cv2.dilate(masks[frame], kernel)
        supported = candidates[dilated[pixels[:, 1], pixels[:, 0]] >= 64]
        if supported.size:
            bottoms.append(coordinates[supported].max())
            visible_heights.append(np.clip(np.ptp(xy[frame, supported, 1]) / height, 0.0, 1.0))
    if not bottoms:
        raise ValueError("No valid framing frames.")

    projected, depths = project_incam(incam_keypoints, intrinsics)
    mapping = {str(name).strip().lower().replace("_", "-"): index for index, name in enumerate(names)}
    wrists = [mapping[name] for name in ("left-wrist", "right-wrist")]
    wrist_xy = projected[:, wrists]
    wrist_depth = depths[:, wrists]
    wrist_out_x = (wrist_depth <= 1e-6) | (wrist_xy[..., 0] < 0) | (wrist_xy[..., 0] >= width)
    wrist_out_y = (wrist_depth <= 1e-6) | (wrist_xy[..., 1] < 0) | (wrist_xy[..., 1] >= height)
    coco_names = (
        "nose", "left-eye", "right-eye", "left-ear", "right-ear", "left-shoulder", "right-shoulder",
        "left-elbow", "right-elbow", "left-wrist", "right-wrist", "left-hip", "right-hip",
        "left-knee", "right-knee", "left-ankle", "right-ankle",
    )
    alignment = None
    if all(name in mapping for name in coco_names) and detected.any():
        errors = np.linalg.norm(projected[:, [mapping[name] for name in coco_names]] - detector[..., :2], axis=-1)
        alignment = float(np.median(errors[detected]) / height)
    valid_ratio = len(bottoms) / frame_count
    torso_ratio = float(torso_valid.mean())
    alignment_score = 0.6 if alignment is None else float(np.clip(1.0 - alignment / 0.15, 0.0, 1.0))
    confidence = float(np.clip(0.45 * valid_ratio + 0.35 * torso_ratio + 0.20 * alignment_score, 0.0, 1.0))
    bottom, bottom_p50, bottom_p80 = map(float, np.percentile(bottoms, [20.0, 50.0, 80.0]))
    if bottom >= FULL_BODY_BOTTOM:
        label = "full_body"
    elif bottom >= HALF_BODY_BOTTOM:
        label = "half_body"
    else:
        label = "close_up"
    return InputFraming(
        label=label,
        visible_body_bottom=round(bottom, 6),
        visible_body_bottom_p50=round(bottom_p50, 6),
        visible_body_bottom_p80=round(bottom_p80, 6),
        visible_height_ratio=round(float(np.median(visible_heights)), 6),
        wrist_out_ratio_x=round(float(np.any(wrist_out_x, axis=1).mean()), 6),
        wrist_out_ratio_y=round(float(np.any(wrist_out_y, axis=1).mean()), 6),
        valid_frame_ratio=round(valid_ratio, 6),
        torso_valid_ratio=round(torso_ratio, 6),
        projection_alignment_error_ratio=None if alignment is None else round(alignment, 6),
        confidence=round(confidence, 6),
    )


def _camera_coordinates(points: np.ndarray, cameras: Sequence[Camera]) -> np.ndarray:
    world = np.asarray(points, dtype=np.float64)
    points_h = np.concatenate([world, np.ones((*world.shape[:-1], 1), dtype=np.float64)], axis=-1)
    w2c = np.asarray([camera.world_to_camera for camera in cameras], dtype=np.float64)
    return np.einsum("cij,fkj->cfki", w2c[:, :3], points_h)


def projected_axis_ratios(
    camera_points: np.ndarray,
    focal_normalized: float,
    *,
    axis: int,
    axis_scale: float = 1.0,
) -> np.ndarray:
    """Return each frame's largest extent over [cameras,frames,keypoints,3] as a fraction of H * axis_scale."""
    depths = camera_points[..., 2]
    coordinates = focal_normalized * camera_points[..., axis] / np.maximum(depths, 1e-6) / axis_scale
    ratios = coordinates.max(axis=2) - coordinates.min(axis=2)
    ratios[np.any(depths <= 1e-6, axis=2)] = np.inf
    return ratios.max(axis=0)


def _ratio_percentile(ratios: np.ndarray, percentile: float) -> float:
    """Linear percentile preserving +inf for unprojectable frames."""
    if np.isfinite(ratios).all():
        return float(np.percentile(ratios, percentile))
    ordered = np.sort(ratios)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower, upper = ordered[int(np.floor(rank))], ordered[int(np.ceil(rank))]
    if lower == upper:
        return float(lower)
    fraction = rank - np.floor(rank)
    return float(lower * (1.0 - fraction) + upper * fraction)


def solve_radius(
    points: np.ndarray,
    names: Sequence[str],
    camera_factory: Callable[[float, float], Sequence[Camera]],
    aspect_ratio: float,
    spec: FramingConfig = FRAMING,
) -> RadiusSolve:
    """Find the smallest feasible radius for fixed-pitch rings, within spec bounds."""
    names = [name.lower() for name in names]
    width_ids = [index for index, name in enumerate(names) if not any(token in name for token in _FINGER_TOKENS)]
    width_points = np.asarray(points)[:, width_ids]
    # Wrists widen the frame without setting its height.
    height_columns = [column for column, index in enumerate(width_ids) if "wrist" not in names[index]]

    lower, upper = spec.min_radius, spec.max_radius
    # Evaluate both bounds, then bisect while retaining the feasible upper end.
    for step in range(2 + 24):
        radius = lower if step == 0 else upper if step == 1 else (lower + upper) / 2
        camera_points = _camera_coordinates(width_points, camera_factory(radius, spec.reference_target_height))
        heights = projected_axis_ratios(camera_points[:, :, height_columns], spec.reference_focal_normalized, axis=1)
        widths = projected_axis_ratios(camera_points, spec.reference_focal_normalized, axis=0, axis_scale=aspect_ratio)
        height = _ratio_percentile(heights, spec.height_percentile)
        width = _ratio_percentile(widths, spec.width_percentile)
        score = max(height / spec.height_target_ratio, width / spec.width_target_ratio)
        bound = "min" if step == 0 and score <= 1.0 else "max" if step == 1 and score > 1.0 else None
        limiting = "width" if width / spec.width_target_ratio > height / spec.height_target_ratio else "height"
        candidate = RadiusSolve(radius, height, width, bound, limiting)
        if bound is not None:
            return candidate
        if score > 1.0:
            lower = radius
        else:
            upper, best = radius, candidate
        if step >= 2 and 0.0 <= 1.0 - score <= 1e-4:
            break
    return best


def adaptive_thresholds(profile: InputFraming) -> AdaptiveThresholds:
    """Blend full-body and close-up occupancy constraints by visible body extent."""
    strength = float(
        np.clip((FULL_BODY_BOTTOM - profile.visible_body_bottom) / (FULL_BODY_BOTTOM - CLOSE_UP_BOTTOM), 0.0, 1.0)
    )
    return AdaptiveThresholds(strength, 0.80 + 0.12 * strength, 95.0, 0.90 + 0.20 * strength, 80.0 - 30.0 * strength)


def solve_clip_framing(
    keypoints: np.ndarray,
    names: Sequence[str],
    profile: InputFraming,
    camera_factory: Callable[[float, float], Sequence[Camera]],
    aspect_ratio: float,
    spec: FramingConfig = FRAMING,
) -> ClipFraming:
    """Fit radius, focal length and target height for fixed-pitch camera rings."""
    # Establish full-body framing before applying a reliable input crop.
    radius_result = solve_radius(keypoints, names, camera_factory, aspect_ratio, spec)
    applied = profile.confidence >= spec.input_min_confidence
    thresholds = adaptive_thresholds(profile) if applied else None
    result = ClipFraming(
        radius=radius_result.radius,
        target_height=spec.reference_target_height,
        focal_normalized=spec.reference_focal_normalized,
        input=profile,
        input_applied=applied,
        radius_solve=radius_result,
        adaptive_thresholds=thresholds,
    )
    if thresholds is None or thresholds.closeup_strength <= 0:
        return result

    # Keep the visible anatomy and interpolate its exact lower boundary.
    samples, coordinates = anatomy_samples(keypoints, names)
    bottom = profile.visible_body_bottom
    core = samples[:, coordinates <= bottom + 1e-8]

    # Both anatomical paths share coordinates: interpolate their cutoff together.
    path_length = samples.shape[1] // 2
    paths = samples.reshape(len(samples), 2, path_length, 3)
    path_coordinates = coordinates[:path_length]
    after = int(np.clip(np.searchsorted(path_coordinates, bottom, side="right"), 1, path_length - 1))
    low, high = path_coordinates[after - 1], path_coordinates[after]
    weight = np.clip((bottom - low) / (high - low), 0.0, 1.0)
    cutoff = paths[:, :, after - 1] * (1.0 - weight) + paths[:, :, after] * weight
    if not np.any(np.isclose(coordinates, bottom, rtol=0.0, atol=1e-8)):
        core = np.concatenate([core, cutoff], axis=1)

    # Arms constrain width without changing the visible body's height constraint.
    mapping = {str(name).strip().lower().replace("_", "-"): index for index, name in enumerate(names)}
    arm_tokens = ("shoulder", "acromion", "elbow", "olecranon", "cubital-fossa", "wrist")
    arm_ids = [index for name, index in mapping.items() if any(token in name for token in arm_tokens)]
    arms = np.asarray(keypoints)[:, arm_ids]
    # Project once per step: core sets height, core and arms set width, cutoff sets position.
    points = np.concatenate([core, arms, cutoff], axis=1).astype(np.float32)
    core_end, width_end = core.shape[1], core.shape[1] + arms.shape[1]
    core_y = points[:, :core_end, 1]

    # Move closer crops toward the median vertical center of the visible anatomy.
    centers = (core_y.min(axis=1) + core_y.max(axis=1)) / 2
    anatomical_target = float(np.median(centers))
    alignment = min(1.0, 2.0 * thresholds.closeup_strength)
    initial_target = (spec.reference_target_height * (1.0 - alignment) + anatomical_target * alignment)

    # Evaluate both bounds, then keep the closest result over at most 20 midpoints.
    lower, upper = initial_target - 0.5, initial_target + 0.5
    best_error = np.inf
    for step in range(2 + 20):
        target = lower if step == 0 else upper if step == 1 else (lower + upper) / 2
        camera_points = _camera_coordinates(points, camera_factory(result.radius, target))

        # Extents scale linearly with focal length; a zero extent imposes no limit.
        heights = projected_axis_ratios(camera_points[:, :, :core_end], 1.0, axis=1)
        widths = projected_axis_ratios(camera_points[:, :, :width_end], 1.0, axis=0, axis_scale=aspect_ratio)
        unit_height = _ratio_percentile(heights, thresholds.height_percentile)
        unit_width = _ratio_percentile(widths, thresholds.width_percentile)

        height_focal = thresholds.height_target_ratio / unit_height if unit_height > 0 else np.inf
        width_focal = thresholds.width_target_ratio / unit_width if unit_width > 0 else np.inf
        unconstrained = min(height_focal, width_focal)
        focal = float(np.clip(unconstrained, spec.reference_focal_normalized, spec.max_focal_normalized))

        # Measure the lower boundary in image-height units, excluding points behind cameras.
        cutoff_points = camera_points[:, :, width_end:]
        positions = 0.5 + focal * cutoff_points[..., 1] / np.maximum(cutoff_points[..., 2], 1e-6)
        visible = (cutoff_points[..., 2] > 1e-6) & np.isfinite(positions)
        cutoff_ratio = float(np.percentile(positions[visible], spec.cutoff_percentile))
        error = cutoff_ratio - spec.cutoff_target_ratio

        # Retain the closest candidate, including either endpoint.
        if abs(error) < best_error:
            best_error = abs(error)
            focal_bound = (
                "min"
                if unconstrained < spec.reference_focal_normalized
                else "max"
                if unconstrained > spec.max_focal_normalized
                else None
            )
            # Positive focal scaling commutes with the linear percentile.
            result = replace(
                result,
                target_height=target,
                focal_normalized=focal,
                cutoff_ratio=cutoff_ratio,
                focal_solve=FocalSolve(
                    focal,
                    unit_height * focal,
                    unit_width * focal,
                    focal_bound,
                    "width" if width_focal < height_focal else "height",
                ),
            )

        # Establish the bracket, then bisect assuming cutoff position is monotone in height.
        if step == 0:
            lower_error = error
        elif step == 1:
            increasing = error > lower_error
            if not min(lower_error, error) <= 0 <= max(lower_error, error):
                return replace(
                    result, target_bound="min" if result.target_height == lower else "max"
                )
        elif abs(error) <= 1e-4:
            break
        elif (error < 0) == increasing:
            lower = target
        else:
            upper = target

    return result
