"""Download the published 4DAnyone assets from Hugging Face.

Missing model checkpoints and bundled example clips are fetched automatically
when inference needs them; the scripts under ``scripts/`` pre-fetch the same
files. SMPL-X is licensed separately, so first-run inference starts its
interactive installer only when a terminal is available.
"""

from __future__ import annotations

import getpass
import logging
import os
import shlex
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from fdanyone.assets import (
    BIREFNET_DIR,
    BIREFNET_FILES,
    BIREFNET_REPO_ID,
    BIREFNET_REVISION,
    EXAMPLE_FILES,
    GVHMR_LINKS,
    HF_REPO_ID,
    HF_REVISION,
    MODEL_FILES,
    PERCEPTUAL_VGG19,
    SMPLX_MODEL,
    resolve_perceptual_vgg19,
)
from fdanyone.errors import AssetError

LOGGER = logging.getLogger("fdanyone")

SMPLX_HOME = "https://smpl-x.is.tue.mpg.de/"
SMPLX_DOWNLOAD_URL = "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip"
SMPLX_ARCHIVE_MEMBER = ("models", "smplx", "SMPLX_NEUTRAL.npz")


def _snapshot(
    allow_patterns: list[str],
    local_dir: Path,
    *,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=local_dir,
    )


def ensure_foreground_model(model_dir: str | Path = "models") -> Path:
    root = Path(model_dir).expanduser().resolve() / BIREFNET_DIR
    missing = [relative for relative in BIREFNET_FILES if not (root / relative).is_file()]
    if missing:
        LOGGER.info("Downloading BiRefNet foreground model (first run only)")
        _snapshot(
            missing,
            root,
            repo_id=BIREFNET_REPO_ID,
            revision=BIREFNET_REVISION,
        )
    return root


def require_gvhmr_checkout(gvhmr_root: str | Path) -> Path:
    root = Path(gvhmr_root).expanduser().resolve()
    if not (root / "hmr4d/__init__.py").is_file():
        raise AssetError(f"GVHMR checkout missing: {root}")
    return root


def _ensure_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        try:
            if destination.resolve(strict=True).samefile(source):
                return
        except FileNotFoundError:
            pass
        destination.unlink()
    elif destination.exists():
        if destination.samefile(source):
            return
        raise AssetError(f"GVHMR asset path is occupied: {destination}")
    if os.name == "nt":
        # Hardlinks need no Developer Mode or elevation and do not duplicate
        # large checkpoints. They also preserve the samefile identity above.
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    try:
        target = os.path.relpath(source, start=destination.parent)
    except ValueError:
        # relpath cannot cross Windows drives; a symlink can use an absolute path.
        target = str(source)
    destination.symlink_to(target)


def create_classic_gvhmr_links(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = "third_party/GVHMR",
    *,
    require_models: bool = True,
    require_smplx: bool = True,
) -> Path:
    """Create the ignored compatibility links expected by upstream GVHMR."""

    root = require_gvhmr_checkout(gvhmr_root)
    models = Path(model_dir).expanduser().resolve()
    for relative, target in GVHMR_LINKS:
        source = models / relative
        required = require_smplx if relative == SMPLX_MODEL else require_models
        if not required and not source.is_file():
            continue
        _ensure_link(source, root / target)
    return root


def ensure_models(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = "third_party/GVHMR",
) -> Path:
    """Download any missing published model file and refresh the GVHMR links."""

    require_gvhmr_checkout(gvhmr_root)
    models = Path(model_dir).expanduser().resolve()
    missing = [relative for relative in MODEL_FILES if not (models / relative).is_file()]
    if missing:
        LOGGER.info("Downloading %d model files from %s (first run only)", len(missing), HF_REPO_ID)
        # Repository paths match the local layout, so download straight into
        # place; huggingface_hub stages and resumes partial files itself.
        _snapshot(missing, models)
    ensure_foreground_model(models)
    create_classic_gvhmr_links(models, gvhmr_root, require_smplx=False)
    return models


def ensure_perceptual_vgg19(model_dir: str | Path = "models") -> Path:
    """Download only the optional VGG-19 reconstruction asset when missing."""

    models = Path(model_dir).expanduser().resolve()
    destination = models / PERCEPTUAL_VGG19
    if not destination.is_file():
        LOGGER.info("Downloading the perceptual VGG-19 model (first use only)")
        _snapshot([PERCEPTUAL_VGG19], models)
    return resolve_perceptual_vgg19(models)


def download_model(
    model_dir: str = "models",
    gvhmr_root: str = "third_party/GVHMR",
) -> dict[str, str]:
    """Download the published model checkpoints."""

    models = ensure_models(model_dir, gvhmr_root)
    return {
        "models": str(models),
        "revision": HF_REVISION,
        "foreground_revision": BIREFNET_REVISION,
    }


def download_example(data_dir: str = "data") -> dict[str, str]:
    """Download the bundled example clips."""

    data = Path(data_dir).expanduser().resolve()
    destinations = {relative: data / Path(relative).relative_to("data") for relative in EXAMPLE_FILES}
    missing = [relative for relative, destination in destinations.items() if not destination.is_file()]
    if missing:
        # Repository paths carry a leading ``data/`` prefix while --data_dir is
        # the local root itself, so stage the snapshot and move each file.
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=data) as temporary:
            staging = Path(temporary)
            _snapshot(missing, staging)
            for relative in missing:
                destination = destinations[relative]
                destination.parent.mkdir(parents=True, exist_ok=True)
                (staging / relative).replace(destination)
    return {"examples": str(data / "source/pexels"), "revision": HF_REVISION}


def ensure_example_video(video_path: str | Path) -> Path:
    """Fetch a bundled example clip when its expected file is missing."""

    path = Path(video_path).expanduser()
    if path.is_file():
        return path
    matches = [relative for relative in EXAMPLE_FILES if PurePosixPath(relative).name == path.name]
    if not matches:
        raise AssetError(f"Input video missing: {path.resolve()}")
    LOGGER.info("Downloading the bundled example clip %s", path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so the final rename stays on one filesystem.
    # Each invocation owns its staging directory. Concurrent first-use downloads
    # must not move or remove another invocation's files.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=path.parent) as temporary:
        staging = Path(temporary)
        _snapshot(matches[:1], staging)
        (staging / matches[0]).replace(path)
    return path


def _parse_interactive_path(value: str) -> Path:
    path, = shlex.split(value.strip(), posix=os.name != "nt")
    if os.name == "nt" and len(path) >= 2 and path[0] == path[-1] and path[0] in "'\"":
        path = path[1:-1]
    return Path(path).expanduser()


def _copy_model_from_source(source: Path, destination: Path) -> None:
    if source.name == "SMPLX_NEUTRAL.npz":
        shutil.copyfile(source, destination)
        return
    with zipfile.ZipFile(source) as archive:
        model_info, = [
            info
            for info in archive.infolist()
            if not info.is_dir() and PurePosixPath(info.filename).parts[-3:] == SMPLX_ARCHIVE_MEMBER
        ]
        with archive.open(model_info) as model, destination.open("wb") as output:
            shutil.copyfileobj(model, output, length=8 * 1024 * 1024)


def install_smplx(
    source_path: str | Path,
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = "third_party/GVHMR",
) -> Path:
    """Install a user-provided official ZIP or neutral NPZ."""

    source = Path(source_path).expanduser().resolve()
    target = Path(model_dir).expanduser().resolve() / SMPLX_MODEL
    target.parent.mkdir(parents=True, exist_ok=True)
    link_relative = next(link for model, link in GVHMR_LINKS if model == SMPLX_MODEL)
    link = Path(gvhmr_root).expanduser().resolve() / link_relative
    previous_link = None
    if link.exists() and not link.is_symlink():
        if not target.is_file() or not link.samefile(target):
            raise AssetError(f"GVHMR asset path is occupied: {link}")
        previous_link = link.stat(follow_symlinks=False)

    with tempfile.TemporaryDirectory(prefix=f".{target.name}.download-", dir=target.parent) as staging:
        temporary = Path(staging) / target.name
        staged_link = Path(staging) / "gvhmr-link.npz"
        _copy_model_from_source(source, temporary)
        if previous_link is not None:
            # Atomic replacement creates a new inode. Stage another name for it
            # so a verified existing hardlink can be refreshed without copying.
            staged_link.hardlink_to(temporary)
        temporary.replace(target)
        if previous_link is not None:
            try:
                unchanged = os.path.samestat(previous_link, link.stat(follow_symlinks=False))
            except FileNotFoundError:
                unchanged = False
            if not unchanged:
                raise AssetError(f"GVHMR asset changed during install: {link}")
            staged_link.replace(link)
    create_classic_gvhmr_links(model_dir, gvhmr_root, require_models=False, require_smplx=True)
    return target


def _download_official(username: str, password: str, destination: Path) -> None:
    payload = urllib.parse.urlencode({"username": username, "password": password}).encode()
    request = urllib.request.Request(
        SMPLX_DOWNLOAD_URL,
        data=payload,
        headers={"User-Agent": "4DAnyone SMPL-X installer"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def _prompt_for_archive(model_dir: str, gvhmr_root: str) -> dict[str, str] | None:
    print(f"Download models_smplx_v1_1.zip from:\n  {SMPLX_DOWNLOAD_URL}")
    while True:
        try:
            value = input("Archive path (drag the downloaded ZIP here): ").strip()
        except EOFError:
            value = ""
        if not value:
            print("SMPL-X setup cancelled; the downloaded ZIP was not modified.")
            return None
        try:
            installed = install_smplx(_parse_interactive_path(value), model_dir, gvhmr_root)
        except (AssetError, OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"error: {exc}")
            continue
        return {"installed": str(installed)}


def download_smplx(
    archive_path: str | None = None,
    model_dir: str = "models",
    gvhmr_root: str = "third_party/GVHMR",
) -> dict[str, str] | None:
    """Install the separately licensed SMPL-X neutral body model."""

    target = Path(model_dir).expanduser().resolve() / SMPLX_MODEL
    if target.is_file():
        create_classic_gvhmr_links(model_dir, gvhmr_root, require_models=False, require_smplx=True)
        return {"installed": str(target)}

    if archive_path is not None:
        return {"installed": str(install_smplx(archive_path, model_dir, gvhmr_root))}

    print(f"SMPL-X requires a free account and license acceptance at {SMPLX_HOME}")
    try:
        accepted = input("Have you registered and accepted the SMPL-X license? [y/N]: ").strip().lower()
    except EOFError:
        accepted = ""
    if accepted in {"y", "yes"}:
        username = input("SMPL-X username or email: ").strip()
        password = getpass.getpass("SMPL-X password: ")
        if username and password:
            with tempfile.TemporaryDirectory(prefix="fdanyone-smplx-") as temporary_dir:
                archive = Path(temporary_dir) / "models_smplx_v1_1.zip"
                try:
                    _download_official(username, password, archive)
                    installed = install_smplx(archive, model_dir, gvhmr_root)
                except (AssetError, OSError, ValueError, zipfile.BadZipFile) as exc:
                    print(f"Automatic download was unavailable: {exc}")
                else:
                    return {"installed": str(installed)}
    return _prompt_for_archive(model_dir, gvhmr_root)


def ensure_smplx(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = "third_party/GVHMR",
) -> Path:
    """Install SMPL-X interactively on first use, without blocking jobs."""

    require_gvhmr_checkout(gvhmr_root)
    target = Path(model_dir).expanduser().resolve() / SMPLX_MODEL
    if target.is_file():
        create_classic_gvhmr_links(model_dir, gvhmr_root, require_models=False, require_smplx=True)
        return target

    if not getattr(sys.stdin, "isatty", lambda: False)():
        raise AssetError(f"SMPL-X missing: {target}; run python scripts/download_smplx.py.")

    print("SMPL-X is required and has not been installed; starting its licensed setup.")
    result = download_smplx(model_dir=str(model_dir), gvhmr_root=str(gvhmr_root))
    if result is None or not target.is_file():
        raise AssetError("SMPL-X setup cancelled.")
    LOGGER.info("SMPL-X installed; continuing inference")
    return target
