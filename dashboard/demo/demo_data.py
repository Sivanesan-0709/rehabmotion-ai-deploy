"""
AI Exercise Form & Posture Intelligence Platform
Isolated Synthetic Demo Datasets (Fixed Seed for Deterministic Presentation)
All data in this module is clearly identified and tagged as DEMO DATA.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Set fixed random seed for reproducible demo analytics
np.random.seed(42)


def get_demo_notifications() -> List[Dict[str, Any]]:
    """Returns deterministic notification items for the top bar drawer."""
    return [
        {
            "id": "notif_01",
            "title": "Squat Milestone Achieved",
            "message": "Bilateral symmetry exceeded 95% across your last 10 repetitions.",
            "time": "15 min ago",
            "type": "success",
            "read": False,
        },
        {
            "id": "notif_02",
            "title": "Motion DNA Updated",
            "message": "Your personal baseline for Bicep Curl was calibrated with 8 verified reps.",
            "time": "2 hours ago",
            "type": "info",
            "read": False,
        },
        {
            "id": "notif_03",
            "title": "Weekly Consistency Alert",
            "message": "You've trained 4 consecutive days this week. 1 day to beat your record!",
            "time": "Yesterday",
            "type": "warning",
            "read": True,
        },
    ]


def get_demo_kpi_summary() -> Dict[str, Any]:
    """Returns top-level KPI metrics for the Command Center."""
    return {
        "total_reps": 482,
        "avg_form_score": 89.4,
        "active_streak_days": 5,
        "est_calories_burned": 1420,
        "reps_trend": "+18% vs last week",
        "score_trend": "+4.2% improvement",
    }


def get_demo_weekly_activity() -> pd.DataFrame:
    """Returns deterministic weekly repetition breakdown."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    reps = [65, 80, 0, 95, 110, 75, 57]
    scores = [86.5, 88.2, 0.0, 91.4, 89.8, 92.0, 89.4]
    return pd.DataFrame({"Day": days, "Repetitions": reps, "Form Score": scores})


def get_demo_recent_sessions() -> List[Dict[str, Any]]:
    """Returns formatted recent workout sessions for showcase tables."""
    return [
        {
            "id": "ses_demo_01",
            "session_id": "ses_demo_01",
            "date": "Today, 08:15 AM",
            "exercise": "Squat",
            "reps": 25,
            "avg_score": 92.4,
            "duration": 252.0,
            "status": "Optimal Form",
        },
        {
            "id": "ses_demo_02",
            "session_id": "ses_demo_02",
            "date": "Yesterday, 06:45 PM",
            "exercise": "Bicep Curl",
            "reps": 30,
            "avg_score": 88.0,
            "duration": 330.0,
            "status": "Minor Flaws",
        },
        {
            "id": "ses_demo_03",
            "session_id": "ses_demo_03",
            "date": "06 Oct, 07:30 AM",
            "exercise": "Push-up",
            "reps": 20,
            "avg_score": 85.5,
            "duration": 225.0,
            "status": "Optimal Form",
        },
        {
            "id": "ses_demo_04",
            "session_id": "ses_demo_04",
            "date": "05 Oct, 05:20 PM",
            "exercise": "Shoulder Press",
            "reps": 22,
            "avg_score": 91.0,
            "duration": 290.0,
            "status": "Optimal Form",
        },
        {
            "id": "ses_demo_05",
            "session_id": "ses_demo_05",
            "date": "04 Oct, 06:10 PM",
            "exercise": "Lunge",
            "reps": 18,
            "avg_score": 89.5,
            "duration": 210.0,
            "status": "Optimal Form",
        },
    ]


def get_demo_achievements() -> List[Dict[str, Any]]:
    """Returns achievement badges and progress trackers."""
    return [
        {
            "id": "ach_01",
            "title": "Kinematic Centurion",
            "name": "Kinematic Centurion",
            "icon": "🛡️",
            "description": "Complete 100 total verified repetitions across all exercises.",
            "desc": "Complete 100 total verified repetitions across all exercises.",
            "unlocked": True,
            "progress": "100%",
            "tier": "Gold",
        },
        {
            "id": "ach_02",
            "title": "Symmetry Master",
            "name": "Symmetry Master",
            "icon": "⚖️",
            "description": "Maintain over 95% left/right bilateral symmetry in a 15-rep set.",
            "desc": "Maintain over 95% left/right bilateral symmetry in a 15-rep set.",
            "unlocked": True,
            "progress": "100%",
            "tier": "Platinum",
        },
        {
            "id": "ach_03",
            "title": "Depth Perfectionist",
            "name": "Depth Perfectionist",
            "icon": "📐",
            "description": "Achieve optimal joint angle depth (<90° knee flexion) on 10 consecutive squats.",
            "desc": "Achieve optimal joint angle depth (<90° knee flexion) on 10 consecutive squats.",
            "unlocked": False,
            "progress": "80%",
            "tier": "Diamond",
        },
        {
            "id": "ach_04",
            "title": "7-Day Streak Warrior",
            "name": "7-Day Streak Warrior",
            "icon": "🔥",
            "description": "Train for 7 consecutive days with real-time pose estimation.",
            "desc": "Train for 7 consecutive days with real-time pose estimation.",
            "unlocked": False,
            "progress": "71%",
            "tier": "Master",
        },
        {
            "id": "ach_05",
            "title": "Zero-Degradation Hero",
            "name": "Zero-Degradation Hero",
            "icon": "⚡",
            "description": "Complete a 20-rep set without triggering any Movement Consistency Alerts.",
            "desc": "Complete a 20-rep set without triggering any Movement Consistency Alerts.",
            "unlocked": True,
            "progress": "100%",
            "tier": "Gold",
        },
        {
            "id": "ach_06",
            "title": "Form Master",
            "name": "Form Master",
            "icon": "🎯",
            "description": "Achieve a 95%+ overall form score in a 25-repetition workout.",
            "desc": "Achieve a 95%+ overall form score in a 25-repetition workout.",
            "unlocked": True,
            "progress": "100%",
            "tier": "Elite",
        },
    ]


def get_demo_leaderboard(exercise: str = "Squat", timeframe: str = "This Week") -> List[Dict[str, Any]]:
    """Returns deterministic leaderboard rankings for the exercise community."""
    return [
        {"rank": 1, "athlete": "Alex Rivera", "exercise": "Squat", "reps": 140, "score": 96.4, "badge": "Elite Tier"},
        {"rank": 2, "athlete": "Jordan Smith", "exercise": "Squat", "reps": 125, "score": 94.8, "badge": "Elite Tier"},
        {"rank": 3, "athlete": "Elena Rostova", "exercise": "Push-up", "reps": 110, "score": 93.1, "badge": "Pro Tier"},
        {"rank": 4, "athlete": "Marcus Chen", "exercise": "Bicep Curl", "reps": 105, "score": 91.5, "badge": "Pro Tier"},
        {"rank": 5, "athlete": "Sarah Jenkins", "exercise": "Shoulder Press", "reps": 95, "score": 89.2, "badge": "Advanced Tier"},
        {"rank": 6, "athlete": "Liam Patel", "exercise": "Lunge", "reps": 80, "score": 87.6, "badge": "Advanced Tier"},
    ]


def get_demo_exercise_catalog() -> List[Dict[str, Any]]:
    """Returns the comprehensive exercise catalog for testing and showcase."""
    return [
        {"name": "Squat", "muscle_group": "Lower Body", "difficulty": "Intermediate"},
        {"name": "Bicep Curl", "muscle_group": "Upper Body", "difficulty": "Beginner"},
        {"name": "Push-up", "muscle_group": "Upper Body", "difficulty": "Intermediate"},
        {"name": "Lunge", "muscle_group": "Lower Body", "difficulty": "Advanced"},
        {"name": "Shoulder Press", "muscle_group": "Upper Body", "difficulty": "Intermediate"},
    ]


def get_demo_motion_dna_profile(exercise: str = "Squat") -> Dict[str, Any]:
    """Returns a calibrated Motion DNA profile for an exercise."""
    return {
        "exercise": exercise,
        "baseline_reps": 12,
        "consistency_index": 93.8,
        "scores": {
            "ROM Depth": 92.5,
            "Symmetry": 94.0,
            "Cadence Stability": 91.0,
            "Torso Neutrality": 95.0,
            "Joint Smoothness": 96.5,
        },
        "metrics_table": [
            {"Biomechanical Metric": "Range Of Motion", "Mean ± Std": "88.4 ± 3.2 deg", "Median": "88.0 deg", "IQR (25%-75%)": "86.0 - 90.5", "Observed Range": "82.0 - 94.0"},
            {"Biomechanical Metric": "Bilateral Symmetry", "Mean ± Std": "93.8 ± 2.1 %", "Median": "94.0 %", "IQR (25%-75%)": "92.0 - 95.5", "Observed Range": "89.0 - 97.0"},
            {"Biomechanical Metric": "Rep Duration", "Mean ± Std": "2.4 ± 0.2 sec", "Median": "2.3 sec", "IQR (25%-75%)": "2.2 - 2.5", "Observed Range": "2.0 - 2.8"},
            {"Biomechanical Metric": "Torso Stability", "Mean ± Std": "95.2 ± 1.8 %", "Median": "95.0 %", "IQR (25%-75%)": "94.0 - 96.5", "Observed Range": "91.0 - 98.0"},
        ],
    }


def get_demo_workout_plan() -> Dict[str, List[Dict[str, Any]]]:
    """Returns a structured 7-day training schedule with exercises, target reps, and focus."""
    return {
        "Monday (Lower Body)": [
            {"exercise": "Squat", "sets": 3, "target_reps": 12, "focus": "Depth & Torso Stability"},
            {"exercise": "Lunge", "sets": 3, "target_reps": 10, "focus": "Bilateral Knee Balance"},
        ],
        "Tuesday (Upper Body)": [
            {"exercise": "Push-up", "sets": 3, "target_reps": 15, "focus": "Elbow Flaring Prevention"},
            {"exercise": "Shoulder Press", "sets": 3, "target_reps": 10, "focus": "Spine Neutrality"},
        ],
        "Wednesday (Active Recovery)": [
            {"exercise": "Bicep Curl", "sets": 2, "target_reps": 12, "focus": "Controlled Cadence & ROM"},
        ],
        "Thursday (Full Body Power)": [
            {"exercise": "Squat", "sets": 4, "target_reps": 10, "focus": "Explosive Ascent & Smoothness"},
            {"exercise": "Push-up", "sets": 3, "target_reps": 12, "focus": "Full Extension Lockout"},
        ],
        "Friday (Hypertrophy & Arms)": [
            {"exercise": "Bicep Curl", "sets": 4, "target_reps": 12, "focus": "Elbow Stability & Peak Squeeze"},
            {"exercise": "Shoulder Press", "sets": 3, "target_reps": 12, "focus": "Overhead Lockout"},
        ],
        "Saturday (Endurance & Form)": [
            {"exercise": "Squat", "sets": 3, "target_reps": 15, "focus": "Fatigue Resistance & Cadence"},
            {"exercise": "Lunge", "sets": 2, "target_reps": 12, "focus": "Stride Symmetry"},
        ],
        "Sunday (Rest & Review)": [],
    }


def get_demo_form_tip() -> Dict[str, str]:
    """Returns a daily rotational biomechanical coaching tip."""
    tips = [
        {
            "title": "Squat Knee Tracking",
            "tip": "Keep knees aligned over the second toe during descent to eliminate knee valgus torque and protect ligaments.",
            "joint": "Knee & Hip",
        },
        {
            "title": "Bicep Curl Elbow Pinning",
            "tip": "Lock your elbows against your ribs throughout the repetition to ensure pure bicep recruitment without shoulder sway.",
            "joint": "Elbow & Shoulder",
        },
        {
            "title": "Push-up Spine Neutrality",
            "tip": "Engage your core to maintain a straight line from shoulder to ankle. Avoid sagging hips or excessive pike.",
            "joint": "Torso & Spine",
        },
    ]
    return tips[int(time.time()) % len(tips)]
