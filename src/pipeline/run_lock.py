"""
Machine-wide, cross-process lock around one market's trading cycle.

Every trading run (scheduled task, dashboard button, CLI) goes through run_daily_pipeline(), which
holds this lock for the whole cycle. The lock is an operating-system file lock on
<project>/data/locks/pipeline_<market>.lock (msvcrt byte-range lock on Windows, fcntl.flock elsewhere):

- It is exclusive across processes and across threads (each attempt opens its own handle).
- A second attempt never waits: it raises PipelineLockBusy immediately and places no orders.
- The OS releases the lock when the holding process exits for any reason, including a crash or a
  hard kill, so a lock can never be left stale. The lock file remaining on disk means nothing;
  only a live lock blocks.

Holder details (pid, host, start time, command) are written to a separate .holder.json file, because
on Windows other processes cannot read a locked byte range.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from config.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if os.name == "nt":
    import msvcrt

    def _try_lock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class PipelineLockBusy(RuntimeError):
    """Another process (or thread) is already running this market's trading cycle."""

    def __init__(self, market: str, holder: Dict[str, Any]) -> None:
        self.market = market
        self.holder = holder
        who = f"pid {holder.get('pid')} since {holder.get('acquired_at_utc')}" if holder else "holder unknown"
        super().__init__(f"{market} trading cycle is already running ({who})")


def lock_dir() -> Path:
    d = Path(settings.pipeline_lock_dir)
    return d if d.is_absolute() else PROJECT_ROOT / d


def lock_paths(market: str) -> Tuple[Path, Path]:
    stem = f"pipeline_{market.strip().lower()}"
    return lock_dir() / f"{stem}.lock", lock_dir() / f"{stem}.holder.json"


def read_holder(market: str) -> Dict[str, Any]:
    try:
        return json.loads(lock_paths(market)[1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@contextmanager
def pipeline_run_lock(market: str) -> Iterator[Dict[str, Any]]:
    """Hold the machine-wide lock for `market` for the duration of the block, or raise PipelineLockBusy."""
    lock_path, holder_path = lock_paths(market)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    try:
        try:
            _try_lock(f)
        except OSError:
            raise PipelineLockBusy(market, read_holder(market)) from None
        holder = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "market": market,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv)[:300],
        }
        try:
            holder_path.write_text(json.dumps(holder), encoding="utf-8")
        except OSError:
            pass
        try:
            yield holder
        finally:
            try:
                holder_path.unlink()
            except OSError:
                pass
            _unlock(f)
    finally:
        f.close()
