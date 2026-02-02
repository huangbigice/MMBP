"""
Audit logger: JSONL append, date-based file rotation, API key hashing.
Satisfies compliance: who, when, which symbol, what action, what result.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Paths that are audited (prefix match; path params extracted by middleware).
AUDITED_PATH_PREFIX = "/api/v1/"
AUDITED_PATHS = (
    "/api/v1/predict",
    "/api/v1/backtest",
    "/api/v1/backtest/portfolio",
    "/api/v1/stock/",
)
# For /api/v1/stock/{symbol}/indicators and /api/v1/stock/{symbol}/data
STOCK_PATH_PATTERN = re.compile(r"^/api/v1/stock/([^/]+)/(indicators|data)$")


def _is_audited_path(path: str) -> bool:
    if not path.startswith(AUDITED_PATH_PREFIX):
        return False
    if path in ("/api/v1/predict", "/api/v1/backtest", "/api/v1/backtest/portfolio"):
        return True
    if path.startswith("/api/v1/stock/") and STOCK_PATH_PATTERN.match(path):
        return True
    return False


def hash_api_key(value: str, secret: str) -> str:
    """HMAC-SHA256 of value with secret; return hex digest. Never store raw key."""
    if not value or not secret:
        return ""
    return hmac.new(
        secret.encode("utf-8"),
        value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class AuditLogger:
    """
    Append-only JSONL audit log with one file per day.
    Thread-safe; flush after each write. Optional chmod 0o600 on new files.
    """

    def __init__(
        self,
        log_dir: Path | str,
        *,
        retention_days: int = 180,
        hmac_secret: str = "",
        file_mode: int = 0o600,
    ) -> None:
        self._log_dir = Path(log_dir).expanduser().resolve()
        self._retention_days = retention_days
        self._hmac_secret = hmac_secret or ""
        self._file_mode = file_mode
        self._lock = threading.Lock()

    def hash_api_key(self, value: str) -> str:
        return hash_api_key(value, self._hmac_secret)

    def _today_file(self) -> Path:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        name = f"audit-{date.today().isoformat()}.jsonl"
        return self._log_dir / name

    def log(self, entry: dict[str, Any]) -> None:
        """Append one JSON object as a single line. Thread-safe; flush after write."""
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            p = self._today_file()
            existed = p.exists()
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            if not existed and self._file_mode is not None:
                try:
                    os.chmod(p, self._file_mode)
                except OSError:
                    pass

    def cleanup_expired(self) -> int:
        """
        Remove audit files older than retention_days (by filename audit-YYYY-MM-DD.jsonl).
        Call at startup. Returns number of files removed.
        """
        if not self._log_dir.exists():
            return 0
        cutoff = date.today()
        removed = 0
        pattern = re.compile(r"^audit-\d{4}-\d{2}-\d{2}\.jsonl$")
        for f in self._log_dir.iterdir():
            if not f.is_file() or not pattern.match(f.name):
                continue
            try:
                # parse YYYY-MM-DD from filename
                day_str = f.name[6:16]
                file_date = date.fromisoformat(day_str)
                if (cutoff - file_date).days > self._retention_days:
                    f.unlink()
                    removed += 1
            except (ValueError, OSError):
                continue
        return removed


def is_audited_path(path: str) -> bool:
    """Whether this path should be audited (for middleware)."""
    return _is_audited_path(path)
