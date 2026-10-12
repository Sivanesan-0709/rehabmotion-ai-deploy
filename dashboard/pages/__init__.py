"""
AI Exercise Form & Posture Intelligence Platform
Dashboard Pages Module
"""

from dashboard.pages.home import render_home_page
from dashboard.pages.live_coach import render_live_coach_page
from dashboard.pages.motion_dna import render_motion_dna_page
from dashboard.pages.rep_quality import render_rep_quality_page
from dashboard.pages.analytics import render_analytics_page
from dashboard.pages.benchmarks import render_benchmarks_page
from dashboard.pages.history import render_history_page
from dashboard.pages.exercise_library import render_exercise_library_page
from dashboard.pages.workout_planner import render_workout_planner_page
from dashboard.pages.settings import render_settings_page
from dashboard.pages.architecture import render_architecture_page
from dashboard.pages.achievements import render_achievements_page
from dashboard.pages.ai_coach import render_ai_coach_page
from dashboard.pages.reports import render_reports_page
from dashboard.pages.leaderboard import render_leaderboard_page
from dashboard.pages.help_tour import render_help_tour_page
from dashboard.pages.data_studio import render_data_studio_page

__all__ = [
    "render_home_page",
    "render_live_coach_page",
    "render_motion_dna_page",
    "render_rep_quality_page",
    "render_analytics_page",
    "render_benchmarks_page",
    "render_history_page",
    "render_exercise_library_page",
    "render_workout_planner_page",
    "render_settings_page",
    "render_architecture_page",
    "render_achievements_page",
    "render_ai_coach_page",
    "render_reports_page",
    "render_leaderboard_page",
    "render_help_tour_page",
    "render_data_studio_page",
]
