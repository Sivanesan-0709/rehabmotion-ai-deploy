"""
AI Exercise Form & Posture Intelligence Platform
Database Connection Manager & Schema Initialization
"""

import sqlite3
from pathlib import Path
from typing import Optional
from app.config import CONFIG


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    exercise TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    duration REAL DEFAULT 0.0,
    total_reps INTEGER DEFAULT 0,
    correct_reps INTEGER DEFAULT 0,
    incorrect_reps INTEGER DEFAULT 0,
    average_score REAL DEFAULT 0.0,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS repetitions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    rep_number INTEGER NOT NULL,
    score REAL NOT NULL,
    duration REAL NOT NULL,
    rom REAL NOT NULL,
    symmetry REAL NOT NULL,
    smoothness REAL NOT NULL,
    torso_stability REAL NOT NULL,
    deviation REAL DEFAULT 0.0,
    status TEXT NOT NULL,
    issues_json TEXT,
    breakdown_json TEXT,
    timestamp_start REAL NOT NULL,
    timestamp_end REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS motion_profiles (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    exercise TEXT NOT NULL,
    total_reps INTEGER NOT NULL,
    consistency_index REAL NOT NULL,
    metrics_json TEXT NOT NULL,
    fingerprint_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedback_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    rep_number INTEGER,
    message TEXT NOT NULL,
    priority TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_reps_session ON repetitions(session_id);
CREATE INDEX IF NOT EXISTS idx_profiles_user_exercise ON motion_profiles(user_id, exercise);
"""


import contextlib

class DatabaseManager:
    """Manages SQLite database connection lifecycle and schema provisioning."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or CONFIG.database.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextlib.contextmanager
    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()


# Global Database Manager Singleton
DB = DatabaseManager()
