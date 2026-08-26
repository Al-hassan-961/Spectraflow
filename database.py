"""
Spectraflow — SQLite persistence, tuned for mobile.

The old code opened a connection and committed per scan; on flash storage
each fsync wakes the CPU and generates heat.  We now:

  * enable WAL (readers never block the writer, commits are cheaper),
  * batch inserts: one transaction every DB_FLUSH_INTERVAL_S,
  * prune old rows so scans.db does not grow without bound.
"""
from __future__ import annotations

import sqlite3
import threading
import time

import config

DB_NAME = "scans.db"
_PRUNE_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                bssid TEXT,
                ssid TEXT,
                rssi REAL,
                frequency INTEGER,
                distance REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(timestamp)")
    prune(max_rows=config.OUTPUT["DB_MAX_ROWS"])


def prune(max_rows: int | None = None) -> int:
    """Delete rows beyond the cap.  Returns rows deleted."""
    cap = max_rows or config.OUTPUT["DB_MAX_ROWS"]
    with _PRUNE_LOCK:
        with _connect() as conn:
            try:
                cur = conn.execute(
                    "DELETE FROM scans WHERE id IN "
                    "(SELECT id FROM scans ORDER BY timestamp DESC LIMIT -1 OFFSET ?)",
                    (cap,),
                )
                return cur.rowcount
            except sqlite3.OperationalError:
                return 0


class ScanBuffer:
    """
    Accumulates scan rows in memory and flushes them in a single
    transaction.  `flush()` is cheap to call from the sensing loop; it
    only touches the disk every DB_FLUSH_INTERVAL_S.
    """

    def __init__(self, flush_interval_s: float | None = None):
        self.rows: list[tuple] = []
        self.lock = threading.Lock()
        self.flush_interval = flush_interval_s or config.OUTPUT["DB_FLUSH_INTERVAL_S"]
        self._last_flush = 0.0

    def add(self, timestamp, bssid, ssid, rssi, frequency, distance) -> None:
        with self.lock:
            self.rows.append((timestamp, bssid, ssid, rssi, frequency, distance))

    def maybe_flush(self, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        if now - self._last_flush < self.flush_interval:
            return 0
        self._last_flush = now
        with self.lock:
            rows, self.rows = self.rows, []
        if not rows:
            return 0
        with _connect() as conn:
            conn.executemany(
                "INSERT INTO scans (timestamp, bssid, ssid, rssi, frequency, distance) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)


def get_history(limit: int = 100) -> list[tuple]:
    with _connect() as conn:
        return conn.execute(
            "SELECT timestamp, bssid, ssid, rssi, frequency, distance "
            "FROM scans ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
