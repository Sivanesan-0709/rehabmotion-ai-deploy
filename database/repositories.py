"""
AI Exercise Form & Posture Intelligence Platform
Data Access Objects & Repository Layer
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional
import sqlite3

from database.db import DB, DatabaseManager
from exercises.base import RepetitionSummary
from intelligence.motion_dna import MetricDistribution, MotionDNAProfile


class UserRepository:
    def __init__(self, db_manager: DatabaseManager = DB):
        self.db = db_manager

    def get_or_create_user(self, name: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
            if row:
                return dict(row)

            user_id = f"usr_{uuid.uuid4().hex[:8]}"
            now = time.time()
            conn.execute(
                "INSERT INTO users (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, name, now, now),
            )
            conn.commit()
            return {"id": user_id, "name": name, "created_at": now, "updated_at": now}

    def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_all_users(self) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    def delete_user(self, user_id: str) -> None:
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()


class SessionRepository:
    def __init__(self, db_manager: DatabaseManager = DB):
        self.db = db_manager

    def create_session(self, user_id: str, exercise: str) -> str:
        session_id = f"ses_{uuid.uuid4().hex[:10]}"
        now = time.time()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, user_id, exercise, start_time, duration, total_reps, correct_reps, incorrect_reps, average_score)
                VALUES (?, ?, ?, ?, 0.0, 0, 0, 0, 0.0)
                """,
                (session_id, user_id, exercise, now),
            )
            conn.commit()
        return session_id

    def end_session(self, session_id: str) -> None:
        now = time.time()
        with self.db.get_connection() as conn:
            # Aggregate stats from reps table
            rep_stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'Correct' THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN status != 'Correct' THEN 1 ELSE 0 END) as incorrect,
                    AVG(score) as avg_score
                FROM repetitions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            session_row = conn.execute("SELECT start_time FROM sessions WHERE id = ?", (session_id,)).fetchone()
            start_t = session_row["start_time"] if session_row else now
            duration = max(0.0, now - start_t)

            total = rep_stats["total"] or 0
            correct = rep_stats["correct"] or 0
            incorrect = rep_stats["incorrect"] or 0
            avg_score = float(rep_stats["avg_score"] or 0.0)

            conn.execute(
                """
                UPDATE sessions 
                SET end_time = ?, duration = ?, total_reps = ?, correct_reps = ?, incorrect_reps = ?, average_score = ?
                WHERE id = ?
                """,
                (now, duration, total, correct, incorrect, avg_score, session_id),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None

    def list_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY start_time DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM repetitions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM feedback_events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

    def delete_user_sessions(self, user_id: str) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                "DELETE FROM repetitions WHERE session_id IN (SELECT id FROM sessions WHERE user_id = ?)",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM feedback_events WHERE session_id IN (SELECT id FROM sessions WHERE user_id = ?)",
                (user_id,),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.commit()


class RepetitionRepository:
    def __init__(self, db_manager: DatabaseManager = DB):
        self.db = db_manager

    def save_repetition(self, session_id: str, summary: RepetitionSummary, deviation_pct: float = 0.0) -> str:
        rep_db_id = f"rep_{uuid.uuid4().hex[:10]}"
        status_label = "Correct" if summary.is_correct else "Needs Improvement"

        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO repetitions (
                    id, session_id, rep_number, score, duration, rom, symmetry, smoothness, torso_stability, deviation,
                    status, issues_json, breakdown_json, timestamp_start, timestamp_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rep_db_id,
                    session_id,
                    summary.rep_number,
                    summary.form_score,
                    summary.duration_seconds,
                    summary.range_of_motion,
                    summary.symmetry_score,
                    summary.smoothness_score,
                    summary.torso_stability_score,
                    deviation_pct,
                    status_label,
                    json.dumps(summary.issues_detected),
                    json.dumps(summary.score_breakdown),
                    summary.timestamp_start,
                    summary.timestamp_end,
                ),
            )
            # Update running session totals
            conn.execute(
                """
                UPDATE sessions 
                SET total_reps = total_reps + 1,
                    correct_reps = correct_reps + (CASE WHEN ? = 'Correct' THEN 1 ELSE 0 END),
                    incorrect_reps = incorrect_reps + (CASE WHEN ? != 'Correct' THEN 1 ELSE 0 END)
                WHERE id = ?
                """,
                (status_label, status_label, session_id),
            )
            conn.commit()
        return rep_db_id

    def get_session_repetitions(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM repetitions WHERE session_id = ? ORDER BY rep_number ASC",
                (session_id,),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["issues"] = json.loads(d["issues_json"] or "[]")
                d["breakdown"] = json.loads(d["breakdown_json"] or "{}")
                results.append(d)
            return results

    def get_user_all_repetitions(self, user_id: str, exercise: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            if exercise:
                query = """
                SELECT r.*, s.exercise 
                FROM repetitions r 
                JOIN sessions s ON r.session_id = s.id 
                WHERE s.user_id = ? AND s.exercise = ?
                ORDER BY r.timestamp_start ASC
                """
                rows = conn.execute(query, (user_id, exercise)).fetchall()
            else:
                query = """
                SELECT r.*, s.exercise 
                FROM repetitions r 
                JOIN sessions s ON r.session_id = s.id 
                WHERE s.user_id = ?
                ORDER BY r.timestamp_start ASC
                """
                rows = conn.execute(query, (user_id,)).fetchall()

            results = []
            for r in rows:
                d = dict(r)
                d["issues"] = json.loads(d["issues_json"] or "[]")
                d["breakdown"] = json.loads(d["breakdown_json"] or "{}")
                results.append(d)
            return results


class MotionProfileRepository:
    def __init__(self, db_manager: DatabaseManager = DB):
        self.db = db_manager

    def save_profile(self, profile: MotionDNAProfile) -> None:
        metrics_dict = {k: v.to_dict() for k, v in profile.metrics.items()}
        profile_id = f"dna_{profile.user_id}_{profile.exercise_name.lower().replace(' ', '_')}"

        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO motion_profiles (id, user_id, exercise, total_reps, consistency_index, metrics_json, fingerprint_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    total_reps = excluded.total_reps,
                    consistency_index = excluded.consistency_index,
                    metrics_json = excluded.metrics_json,
                    fingerprint_json = excluded.fingerprint_json,
                    updated_at = excluded.updated_at
                """,
                (
                    profile_id,
                    profile.user_id,
                    profile.exercise_name,
                    profile.total_baseline_reps,
                    profile.consistency_index,
                    json.dumps(metrics_dict),
                    json.dumps(profile.fingerprint_scores),
                    profile.created_at,
                    profile.updated_at,
                ),
            )
            conn.commit()

    def get_profile(self, user_id: str, exercise: str) -> Optional[MotionDNAProfile]:
        profile_id = f"dna_{user_id}_{exercise.lower().replace(' ', '_')}"
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM motion_profiles WHERE id = ?", (profile_id,)).fetchone()
            if not row:
                return None

            metrics_data = json.loads(row["metrics_json"])
            metrics_dist = {}
            for k, d in metrics_data.items():
                metrics_dist[k] = MetricDistribution(
                    mean=d["mean"],
                    std=d["std"],
                    median=d["median"],
                    p25=d["p25"],
                    p75=d["p75"],
                    min_val=d["min"],
                    max_val=d["max"],
                )

            return MotionDNAProfile(
                user_id=row["user_id"],
                exercise_name=row["exercise"],
                total_baseline_reps=row["total_reps"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metrics=metrics_dist,
                fingerprint_scores=json.loads(row["fingerprint_json"]),
                consistency_index=row["consistency_index"],
            )


class FeedbackRepository:
    def __init__(self, db_manager: DatabaseManager = DB):
        self.db = db_manager

    def log_event(self, session_id: str, message: str, rep_number: Optional[int] = None, priority: str = "normal") -> None:
        event_id = f"fdb_{uuid.uuid4().hex[:10]}"
        now = time.time()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback_events (id, session_id, rep_number, message, priority, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, session_id, rep_number, message, priority, now),
            )
            conn.commit()

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback_events WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
