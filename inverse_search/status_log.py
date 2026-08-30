"""
Lightweight periodic status file for long-running scripts -- so
progress can be watched from outside (open the file yourself anytime)
instead of relying on someone else polling and summarizing scrollback.

Not a replacement for the checkpoint system (checkpoint.py) -- this is
read-only, human-facing status, not something meant to be loaded back
to resume anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class StatusLogger:
    def __init__(self, path: str | Path, min_interval_s: float = 120):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self._last_write = 0.0

    def update(self, force: bool = False, **fields) -> None:
        """Writes fields to the status file, throttled to at most once
        per min_interval_s unless force=True (use force for the first
        and last update of a run, and anything you don't want to risk
        missing)."""
        now = time.monotonic()
        if not force and (now - self._last_write) < self.min_interval_s:
            return
        self._last_write = now

        data = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}
        # Same atomic-ish write pattern as checkpoint.py -- avoids a
        # half-written, unreadable status file if read mid-write.
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(self.path)
