"""Execute the ordinary inference entrypoint in an owned, cancellable process."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import suppress
from pathlib import Path


class ProgressHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        write_status(self.path, record.getMessage(), float(record.fraction))


def write_status(path: Path, message: str, fraction: float, **fields) -> None:
    """Readers always see one complete status document."""

    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"message": message, "fraction": fraction, **fields}) + "\n")
    # Windows refuses to replace a file while the Space server is reading it.
    for _ in range(100):
        with suppress(PermissionError):
            temporary.replace(path)
            return
        time.sleep(0.01)
    temporary.replace(path)


def main(request_path: str) -> None:
    request = Path(request_path)
    status = request.with_name("status.json")
    progress = logging.getLogger("fdanyone.progress")
    progress.setLevel(logging.INFO)
    progress.addHandler(ProgressHandler(status))
    try:
        from inference import inference

        summary = inference(**json.loads(request.read_text()))
        write_status(status, "Inference complete", 1.0, summary=summary)
    except Exception as exc:
        # The Space shows this line; the interpreter still prints the traceback and exits with status 1.
        message = (str(exc).strip() or type(exc).__name__).splitlines()[0]
        write_status(status, message, 0.0, error=True)
        raise


if __name__ == "__main__":
    main(sys.argv[1])
