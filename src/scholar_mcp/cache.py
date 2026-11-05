"""Small SQLite TTL cache so repeated tool calls don't burn upstream rate limits."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path


class Cache:
    def __init__(self, path: str | Path = ":memory:", clock: Callable[[], float] = time.time):
        if str(path) != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            path = Path(path).expanduser()
        self._clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "expires REAL NOT NULL)"
        )
        self._db.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(*parts: object) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT value, expires FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            value, expires = row
            if expires < self._clock():
                self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._db.commit()
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key: str, value: str, ttl: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires) VALUES (?, ?, ?)",
                (key, value, self._clock() + ttl),
            )
            self._db.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            (entries,) = self._db.execute("SELECT COUNT(*) FROM cache").fetchone()
        return {"entries": entries, "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        with self._lock:
            self._db.close()
