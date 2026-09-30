# fdanyone/video.py
from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from fdanyone.config import INFERENCE
from fdanyone.errors import VideoContractError

AUTO_DOWNSAMPLE_FPS = tuple(
    Fraction(numerator, denominator) for numerator, denominator in INFERENCE.auto_downsample_fps
)
SAMPLING_POLICY = INFERENCE.temporal_sampling_policy
INTEGER_RATIO_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CanonicalFrame:
    rgb: np.ndarray
    source_index: int
    source_pts: int | None
    source_timestamp: Fraction
    canonical_timestamp: Fraction


@dataclass(frozen=True)
class ClipInfo:
    """Input provenance and raster information without decoded pixel storage."""

    source_path: Path
    fps: Fraction
    start_time: Fraction
    num_frames: int
    height: int
    width: int

    @property
    def fps_num(self) -> int:
        return self.fps.numerator

    @property
    def fps_den(self) -> int:
        return self.fps.denominator


@dataclass(frozen=True)
class CanonicalClip:
    source_path: Path
    source_size_bytes: int
    source_mtime_ns: int
    fps: Fraction
    input_rate: Fraction
    source_time_base: Fraction
    start_time: Fraction
    frames: tuple[CanonicalFrame, ...]
    rotation_degrees: int

    @property
    def info(self) -> ClipInfo:
        return ClipInfo(
            source_path=self.source_path,
            fps=self.fps,
            start_time=self.start_time,
            num_frames=len(self.frames),
            height=self.height,
            width=self.width,
        )

    @property
    def height(self) -> int:
        return self.frames[0].rgb.shape[0]

    @property
    def width(self) -> int:
        return self.frames[0].rgb.shape[1]

    @property
    def fps_num(self) -> int:
        return self.fps.numerator

    @property
    def fps_den(self) -> int:
        return self.fps.denominator

    @property
    def rgb_frames(self) -> tuple[np.ndarray, ...]:
        return tuple(frame.rgb for frame in self.frames)

    def metadata(self) -> dict:
        return {
            # Basename only: workers need the source identity, not this machine's absolute path.
            "source_path": self.source_path.name,
            "source_size_bytes": self.source_size_bytes,
            "source_mtime_ns": self.source_mtime_ns,
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "input_rate_num": self.input_rate.numerator,
            "input_rate_den": self.input_rate.denominator,
            "source_time_base_num": self.source_time_base.numerator,
            "source_time_base_den": self.source_time_base.denominator,
            "start_time_num": self.start_time.numerator,
            "start_time_den": self.start_time.denominator,
            "sampling_policy": SAMPLING_POLICY,
            "num_frames": len(self.frames),
            "height": self.height,
            "width": self.width,
            "rotation_degrees_applied": self.rotation_degrees,
            "frames": [
                {
                    "canonical_index": index,
                    "canonical_timestamp_sec": float(frame.canonical_timestamp),
                    "source_index": frame.source_index,
                    "source_pts": frame.source_pts,
                    "source_timestamp_sec": float(frame.source_timestamp),
                }
                for index, frame in enumerate(self.frames)
            ],
        }

    def write_metadata(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    pts: int | None
    timestamp: Fraction
    rotation_degrees: int
    # Released once a caller keeps the RGB, so selected frames do not pin decoder buffers.
    frame: av.VideoFrame | None = field(default=None, repr=False, compare=False)

    @property
    def rgb(self) -> np.ndarray:
        """Convert on demand; callers use only a subset of the decoded frames."""

        rgb = self.frame.to_ndarray(format="rgb24")
        return np.ascontiguousarray(np.rot90(rgb, k=self.rotation_degrees // 90))


def stream_rate(stream: av.video.stream.VideoStream) -> Fraction:
    return Fraction(stream.average_rate or stream.guessed_rate or stream.base_rate)


def choose_canonical_fps(input_rate: Fraction) -> Fraction:
    """Keep the source rate unless it supports clean integer downsampling."""

    for candidate in sorted(AUTO_DOWNSAMPLE_FPS, reverse=True):
        ratio = float(input_rate / candidate)
        multiple = round(ratio)
        if multiple >= 2 and abs(ratio - multiple) <= INTEGER_RATIO_TOLERANCE:
            return input_rate / multiple
    return input_rate


def validate_clip_options(
    *,
    start_time: float,
    fps: str | int | float | Fraction | None,
) -> Fraction | None:
    """Validate clip options without video I/O and normalize the requested FPS."""

    if not math.isfinite(start_time) or start_time < 0:
        raise VideoContractError(f"Invalid start_time: {start_time}; expected a finite value >= 0.")
    if fps is None:
        return None
    rate = Fraction(str(fps))
    if rate <= 0:
        raise VideoContractError(f"FPS must be positive: {rate}.")
    return rate


def decode_frames(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
) -> Iterator[DecodedFrame]:
    rotations = set()
    for index, frame in enumerate(container.decode(stream)):
        # FFmpeg reports rotation only as per-frame display-matrix side data (CCW degrees).
        rotation = round(frame.rotation) % 360
        rotations.add(rotation)
        if rotation % 90 or len(rotations) > 1:
            raise VideoContractError(f"Video rotation must be a constant multiple of 90 degrees; frame {index} has {frame.rotation!r}.")
        yield DecodedFrame(
            index=index,
            pts=frame.pts,
            timestamp=(
                Fraction(frame.pts * frame.time_base)
                if frame.pts is not None
                else Fraction(index, 1) / stream_rate(stream)
            ),
            rotation_degrees=rotation,
            frame=frame,
        )


def decode_canonical_clip(
    video_path: str | Path,
    *,
    num_frames: int = 121,
    start_time: float = 0.0,
    fps: str | int | float | Fraction | None = None,
) -> CanonicalClip:
    """Decode one canonical clip, selecting frames by source presentation time."""

    path = Path(video_path).expanduser().resolve()
    fps = validate_clip_options(start_time=start_time, fps=fps)

    source_stat = path.stat()

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        # Frame threading: same frames and timestamps as the slice-only default, several times faster.
        stream.thread_type = "AUTO"
        input_rate = stream_rate(stream)
        output_rate = choose_canonical_fps(input_rate) if fps is None else fps
        decoded = decode_frames(container, stream)
        previous = next(decoded)

        origin = previous.timestamp
        start_offset = Fraction(str(start_time))
        start = origin + start_offset
        targets = [start + Fraction(index, 1) / output_rate for index in range(num_frames)]
        # 3/4 of the slower clock's period: upsampling follows the source clock, downsampling keeps its bound.
        max_error = Fraction(3, 4) / min(input_rate, output_rate)
        # Only selected frames are converted (once each); unselected ones never leave YUV.
        selected: list[DecodedFrame] = []
        rgb: dict[int, np.ndarray] = {}
        target_index = 0
        current = previous

        for current in decoded:
            if current.timestamp < previous.timestamp:
                raise VideoContractError(f"Non-monotonic video PTS at frame {current.index}.")
            if (
                output_rate > input_rate
                and previous.timestamp + max_error < targets[-1]
                and current.timestamp - max_error > start
                and current.timestamp - previous.timestamp > 2 * max_error
            ):
                # Offset targets can straddle a hole, so reject any in-clip source gap beyond 2 * max_error.
                raise VideoContractError(f"Video timestamp gap at frame {current.index}.")
            while target_index < num_frames and targets[target_index] <= current.timestamp:
                target = targets[target_index]
                candidate = previous if abs(previous.timestamp - target) <= abs(current.timestamp - target) else current
                if candidate.index not in rgb:
                    rgb[candidate.index] = candidate.rgb
                selected.append(replace(candidate, frame=None))
                target_index += 1
            if target_index >= num_frames:
                break
            previous = current

        # Reuse the last source frame while within max_error; beyond that the input is too short, never padded.
        while target_index < num_frames and abs(current.timestamp - targets[target_index]) <= max_error:
            if current.index not in rgb:
                rgb[current.index] = current.rgb
            selected.append(replace(current, frame=None))
            target_index += 1

        if len(selected) != num_frames:
            duration = float(current.timestamp - origin)
            required = float(Fraction(num_frames - 1, 1) / output_rate + start_offset)
            raise VideoContractError(f"Video too short: {duration:.3f}s; need {required:.3f}s.")

        errors = [abs(frame.timestamp - target) for frame, target in zip(selected, targets, strict=True)]
        if max(errors) > max_error:
            raise VideoContractError(f"Video timestamp gap: {float(max(errors)):.4f}s > {float(max_error):.4f}s.")
        if len({image.shape for image in rgb.values()}) > 1:
            raise VideoContractError("Video frame dimensions change within the clip.")
        source_time_base = Fraction(stream.time_base)

        canonical_frames = tuple(
            CanonicalFrame(
                rgb=rgb[frame.index],
                source_index=frame.index,
                source_pts=frame.pts,
                source_timestamp=frame.timestamp,
                canonical_timestamp=Fraction(index, 1) / output_rate,
            )
            for index, frame in enumerate(selected)
        )

    final_stat = path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise VideoContractError(f"Video changed during decoding: {path}.")
    return CanonicalClip(
        source_path=path,
        source_size_bytes=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
        fps=output_rate,
        input_rate=input_rate,
        source_time_base=source_time_base,
        start_time=start_offset,
        frames=canonical_frames,
        rotation_degrees=selected[0].rotation_degrees,
    )


def iter_rgb_video(path: str | Path) -> Iterator[np.ndarray]:
    """Stream RGB frames without interpreting or resampling timestamps."""

    video_path = Path(path).expanduser().resolve()
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield np.ascontiguousarray(frame.to_ndarray(format="rgb24"))


def verify_lossless_video(clip: CanonicalClip, path: str | Path) -> None:
    with closing(iter_rgb_video(path)) as decoded:
        for index, (actual, expected) in enumerate(zip(decoded, clip.rgb_frames, strict=True)):
            if not np.array_equal(actual, expected):
                raise VideoContractError(f"Lossless video mismatch at frame {index}.")


def write_lossless_video(clip: CanonicalClip, path: str | Path) -> Path:
    """Write an FFV1 working video. A subsequent decode must be RGB-identical."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=clip.fps)
        stream.width = clip.width
        stream.height = clip.height
        # FFV1 lacks 8-bit GBR planar; BGR0 (fourth byte padding) round-trips rgb24 exactly.
        stream.pix_fmt = "bgr0"
        for index, canonical_frame in enumerate(clip.frames):
            frame = av.VideoFrame.from_ndarray(canonical_frame.rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 1) / clip.fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    verify_lossless_video(clip, output_path)
    return output_path


def write_gvhmr_video(clip: CanonicalClip, path: str | Path) -> Path:
    """Write the frame-counted, RGB-lossless MP4 consumed by GVHMR."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264rgb", rate=clip.fps)
        stream.width = clip.width
        stream.height = clip.height
        stream.pix_fmt = "rgb24"
        # CRF 0 is lossless at any preset; ultrafast's default slices encode ~3x faster and decode in parallel.
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for index, canonical_frame in enumerate(clip.frames):
            frame = av.VideoFrame.from_ndarray(canonical_frame.rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 1) / clip.fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    verify_lossless_video(clip, output_path)
    return output_path


def load_canonical_working_clip(video_path: str | Path, metadata_path: str | Path) -> CanonicalClip:
    """Rehydrate the canonical clip inside a short-lived worker."""

    video_path = Path(video_path).expanduser().resolve()
    # Written by CanonicalClip.write_metadata in the same run, so every number is already an int.
    metadata = json.loads(Path(metadata_path).read_text())
    fps = Fraction(metadata["fps_num"], metadata["fps_den"])
    frames = []
    with closing(iter_rgb_video(video_path)) as decoded:
        for canonical_index, (rgb, record) in enumerate(zip(decoded, metadata["frames"], strict=True)):
            frames.append(
                CanonicalFrame(
                    rgb=rgb,
                    source_index=record["source_index"],
                    source_pts=record["source_pts"],
                    source_timestamp=Fraction(str(record["source_timestamp_sec"])),
                    canonical_timestamp=Fraction(canonical_index, 1) / fps,
                )
            )
    return CanonicalClip(
        source_path=Path(metadata["source_path"]),
        source_size_bytes=metadata["source_size_bytes"],
        source_mtime_ns=metadata["source_mtime_ns"],
        fps=fps,
        input_rate=Fraction(metadata["input_rate_num"], metadata["input_rate_den"]),
        source_time_base=Fraction(metadata["source_time_base_num"], metadata["source_time_base_den"]),
        start_time=Fraction(metadata["start_time_num"], metadata["start_time_den"]),
        frames=tuple(frames),
        rotation_degrees=metadata["rotation_degrees_applied"],
    )


def write_video(
    frames: Iterator[np.ndarray] | tuple[np.ndarray, ...],
    path: str | Path,
    fps: Fraction,
    *,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Encode RGB frames as a broadly playable H.264 MP4."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}
        for index, rgb in enumerate(itertools.chain((first,), iterator)):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 1) / fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output_path
