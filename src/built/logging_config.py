"""Unified logging for every built.* module (worker, reviver, curator, archiver,
...). configure_logging() sets up two handlers on the "built" logger — every
built.foo.bar logger is a child of it and propagates up by default, so nothing
else needs to change to be captured:

- a normal stream handler, for whoever's running the process directly or
  tailing it via systemd/docker logs
- an in-memory ring buffer a UI view can poll, mirroring the same seq-cursor
  idiom CardEvent already uses for the per-card transcript (see
  services/card_service.list_events) — process-local and cleared on restart,
  same as the rest of this app's "what's happening right now" views.

Scoped to the "built" logger rather than the root logger on purpose: third-party
library logs (sqlalchemy, httpx, uvicorn.access) log under their own top-level
names, so this stays a log of the app's own activity, not framework noise.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count

_MAX_BUFFERED_RECORDS = 1000


@dataclass
class LogEntry:
    seq: int
    created_at: datetime
    level: str
    logger_name: str
    message: str


class _RingBufferHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._buffer: deque[LogEntry] = deque(maxlen=_MAX_BUFFERED_RECORDS)
        self._next_seq = count(1)

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(
            LogEntry(
                seq=next(self._next_seq),
                created_at=datetime.now(UTC),
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
            )
        )

    def since(self, seq: int, *, limit: int = 500) -> list[LogEntry]:
        matched = [r for r in self._buffer if r.seq > seq]
        return matched[-limit:]


_ring_buffer = _RingBufferHandler()
_ring_buffer.setFormatter(logging.Formatter("%(message)s"))


def configure_logging(level: int | str = logging.INFO) -> None:
    """Idempotent — safe to call more than once (e.g. once per test) without
    stacking duplicate handlers."""
    app_logger = logging.getLogger("built")
    app_logger.setLevel(level)
    if _ring_buffer in app_logger.handlers:
        return
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    app_logger.addHandler(stream_handler)
    app_logger.addHandler(_ring_buffer)


def get_logs(*, since_seq: int = 0, limit: int = 500) -> list[LogEntry]:
    return _ring_buffer.since(since_seq, limit=limit)
