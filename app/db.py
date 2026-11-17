"""SQLite connection management.

Two API instances share one SQLite file (e.g. via a Docker volume). Every
request opens its own connection; WAL mode + a busy timeout let multiple
readers and one writer cooperate across processes.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

DB_PATH = os.environ.get("RAIN_DB_PATH", "/data/rainfall.db")
BUSY_TIMEOUT_MS = int(os.environ.get("RAIN_BUSY_TIMEOUT_MS", "10000"))

# Tables are created first; columns added by later versions are then patched
# onto pre-existing databases (see _migrate), and indexes last so that indexes
# referencing a newly added column never run against an old table.
_TABLES = """
CREATE TABLE IF NOT EXISTS groups (
    group_id        TEXT PRIMARY KEY,
    forecast_time   TEXT NOT NULL,
    rows            INTEGER NOT NULL,
    cols            INTEGER NOT NULL,
    thresholds      TEXT NOT NULL,
    change_tolerance REAL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS members (
    group_id    TEXT NOT NULL REFERENCES groups(group_id),
    member_idx  INTEGER NOT NULL,
    member_id   TEXT NOT NULL,
    PRIMARY KEY (group_id, member_idx),
    UNIQUE (group_id, member_id)
);

CREATE TABLE IF NOT EXISTS member_revisions (
    group_id     TEXT NOT NULL,
    member_idx   INTEGER NOT NULL,
    revision     INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    rain_blob    BLOB NOT NULL,
    miss_blob    BLOB NOT NULL,
    status       TEXT NOT NULL DEFAULT 'adopted',
    anomaly_cells INTEGER NOT NULL DEFAULT 0,
    reviewed_at  TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (group_id, member_idx, revision)
);

CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id     TEXT NOT NULL,
    revision_map TEXT NOT NULL,
    is_current   INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (group_id, revision_map)
);

CREATE TABLE IF NOT EXISTS product_thresholds (
    product_id        INTEGER NOT NULL REFERENCES products(product_id),
    threshold         REAL NOT NULL,
    probability_blob  BLOB NOT NULL,
    denom_blob        BLOB NOT NULL,
    PRIMARY KEY (product_id, threshold)
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_products_group_current
    ON products(group_id, is_current);

CREATE INDEX IF NOT EXISTS idx_member_revisions_status
    ON member_revisions(group_id, member_idx, status, revision);

-- hard guarantee: a member can never have two revisions awaiting review
CREATE UNIQUE INDEX IF NOT EXISTS idx_member_revisions_one_pending
    ON member_revisions(group_id, member_idx) WHERE status = 'pending';
"""

# column name -> DDL used when patching a database created before the column existed
_ADDED_COLUMNS = {
    "groups": {
        "change_tolerance": "REAL",
    },
    "member_revisions": {
        "status": "TEXT NOT NULL DEFAULT 'adopted'",
        "anomaly_cells": "INTEGER NOT NULL DEFAULT 0",
        "reviewed_at": "TEXT",
    },
}


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(
        DB_PATH,
        timeout=BUSY_TIMEOUT_MS / 1000.0,
        isolation_level=None,  # explicit BEGIN/COMMIT; do not let sqlite3 autotransact
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release (idempotent, additive)."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_db() -> None:
    """Create schema and switch the file into WAL mode (idempotent)."""
    conn = connect()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(_TABLES)
        conn.execute("BEGIN IMMEDIATE")
        try:
            _migrate(conn)
            for statement in _INDEXES.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()
