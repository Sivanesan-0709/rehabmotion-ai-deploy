"""
AI Exercise Form & Posture Intelligence Platform
Exercise Registry & Module Factory
"""

from typing import Dict, List, Optional
from exercises.base import BaseExercise, MovementPhase, FormIssue, RepetitionSummary
from exercises.squat import SquatExercise
from exercises.curl import BicepCurlExercise
from exercises.pushup import PushUpExercise
from exercises.lunge import LungeExercise
from exercises.shoulder_press import ShoulderPressExercise

AVAILABLE_EXERCISES = {
    "Squat": SquatExercise,
    "Bicep Curl": BicepCurlExercise,
    "Push-up": PushUpExercise,
    "Lunge": LungeExercise,
    "Shoulder Press": ShoulderPressExercise,
}


def get_exercise_engine(name: str) -> BaseExercise:
    """Instantiates exercise state machine by name."""
    cls = AVAILABLE_EXERCISES.get(name)
    if cls is None:
        raise ValueError(f"Exercise '{name}' not recognized. Available: {list(AVAILABLE_EXERCISES.keys())}")
    return cls()


def list_available_exercises() -> List[str]:
    return list(AVAILABLE_EXERCISES.keys())
