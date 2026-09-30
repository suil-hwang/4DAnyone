"""One output directory, its saved CLI request and original source identity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fdanyone.assets import ensure_example_video
from fdanyone.attention import DEFAULT_ATTENTION_BACKEND, normalize_attention_backend
from fdanyone.errors import ConfigurationError, FourDAnyoneError
from fdanyone.io import resolve_output_path, sha256_file, write_json
from fdanyone.output import read_run_request
from fdanyone.space.settings import complete_options, validate_options
from fdanyone.space.viewer import read_result

REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SpaceConfig:
    output_dir: Path
    cache_dir: Path
    model_dir: Path
    gvhmr_root: Path
    gpu_ids: tuple[int, ...] | None = None
    attention_backend: str = DEFAULT_ATTENTION_BACKEND
    video_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention_backend", normalize_attention_backend(self.attention_backend))


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else REPOSITORY / path))


def path_label(value: str | Path) -> str:
    path = repository_path(value)
    return str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else str(path)


def ensure_new_output(directory: Path) -> None:
    if directory.is_symlink() or directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ConfigurationError(f"Output directory is not empty: {directory}")


def resolve_task(video_path: str | Path | None, output_dir: str | Path | None) -> tuple[Path | None, Path]:
    if video_path is None and output_dir is None:
        raise ConfigurationError("Provide --video_path or --output_dir.")
    source = repository_path(video_path).resolve() if video_path is not None else None
    if output_dir is None:
        stem = re.sub(r"[^\w-]", "_", source.stem)[:60] or "clip"
        output_dir = Path("data/fdanyone") / stem / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    destination = resolve_output_path(repository_path(output_dir))
    if source is not None:
        ensure_new_output(destination)
        ensure_example_video(source)
    elif read_run_request(destination) is None:
        raise ConfigurationError(f"No saved task in {destination}; pass --video_path to start one.")
    return source, destination


def source_matches(source: Path, identity: dict, cache_dir: Path) -> bool:
    """Use the saved content hash and the motion cache's exact source clock."""
    try:
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["size_bytes"], identity["mtime_ns"]):
            return False
        key = hashlib.sha256(f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        cached = cache_dir / "identities" / f"{key}.json"
        with suppress(OSError, ValueError, KeyError):
            return json.loads(cached.read_text())["sha256"] == identity["sha256"]
        digest = sha256_file(source)
        if source.stat().st_mtime_ns != stat.st_mtime_ns:
            return False
        # The hash cache only saves time; failing to store it must not change the answer.
        with suppress(OSError):
            write_json(cached, {"sha256": digest})
        return digest == identity["sha256"]
    except (OSError, KeyError):
        return False


@dataclass(frozen=True)
class SavedTask:
    directory: Path
    options: dict
    source: Path | None
    complete: bool
    error: str = ""
    log_dir: Path | None = None

    @property
    def resumable(self) -> bool:
        return not self.complete and not self.error and self.source is not None


def read_task(directory: Path, cache_dir: Path) -> SavedTask:
    request = read_run_request(directory)
    options = complete_options(request["options"])
    validate_options(options)
    source = Path(options["video_path"])
    if not source_matches(source, request["source"], cache_dir):
        source = None
    complete, error = False, ""
    try:
        read_result(directory)
        complete = True
    except (FourDAnyoneError, OSError, ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        # An absent metadata file means the CLI has not published a result yet.
        # An invalid published result requires repair, not another inference run.
        if (directory / "metadata.json").exists():
            error = str(exc)
    return SavedTask(
        directory,
        options,
        source,
        complete,
        error,
        Path(request["space_job"]) if request.get("space_job") else None,
    )
