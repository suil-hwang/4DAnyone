"""Filesystem helpers for crash-safe result publication."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from contextlib import AbstractContextManager, contextmanager, suppress
from pathlib import Path

from fdanyone.errors import FourDAnyoneError

_RETRYABLE_TREE_ERRORS = {errno.EBUSY, errno.ENOTEMPTY, errno.ESTALE}


def resolve_output_path(path: str | Path) -> Path:
    """Normalize an output path without following a possibly dangling leaf symlink."""

    expanded = Path(path).expanduser()
    return expanded.parent.resolve() / expanded.name


@contextmanager
def lock_output(path: Path):
    """Hold a nonblocking exclusive lock on a persistent sidecar file.

    Never unlink the sidecar: replacing it could let concurrent writers acquire
    different locks. Closing the file or exiting the process releases the lock.
    """

    lock_path = path.with_name(f".{path.name}.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(lock_path, flags, 0o666), "r+b", buffering=0) as lock_file:
        descriptor = lock_file.fileno()
        # Reject symlinks and replaced files even when O_NOFOLLOW is unavailable.
        entry = lock_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode) or not os.path.samestat(entry, os.fstat(descriptor)):
            raise FourDAnyoneError(f"Invalid output lock file: {lock_path}.")
        if os.name == "nt":
            import msvcrt

            # A newly opened descriptor starts at byte zero, even in an empty file.
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without materializing it in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: object, *, sort_keys: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=sort_keys) + "\n")
        os.replace(temporary, target)
    finally:
        # Preserve the original write error if cleanup also fails.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def remove_tree(
    path: str | Path,
    *,
    ignore_errors: bool = False,
) -> None:
    """Remove a tree, tolerating short directory-entry lag on network filesystems."""

    target = Path(path)
    attempts = 8
    for attempt in range(attempts):
        try:
            shutil.rmtree(target)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            retryable = exc.errno in _RETRYABLE_TREE_ERRORS and attempt + 1 < attempts
            if not retryable:
                if ignore_errors:
                    return
                raise
            time.sleep(0.1 * (2**attempt))


class AtomicResultDirectory(AbstractContextManager[Path]):
    """Build beside the destination and rename only after all validation passes."""

    def __init__(self, destination: str | Path):
        self.destination = resolve_output_path(destination)
        self.working = self.destination.with_name(f".{self.destination.name}.work-{uuid.uuid4().hex[:10]}")
        self._committed = False

    def __enter__(self) -> Path:
        if os.path.lexists(self.destination):
            raise FourDAnyoneError(f"Output already exists: {self.destination}.")
        self.working.mkdir(parents=True)
        return self.working

    def commit(self) -> Path:
        if self._committed:
            return self.destination
        if os.path.lexists(self.destination):
            raise FourDAnyoneError(f"Output appeared during inference: {self.destination}.")
        os.replace(self.working, self.destination)
        self._committed = True
        return self.destination

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            try:
                self.commit()
            except BaseException:
                remove_tree(self.working, ignore_errors=True)
                raise
        else:
            remove_tree(self.working, ignore_errors=True)
        return False
