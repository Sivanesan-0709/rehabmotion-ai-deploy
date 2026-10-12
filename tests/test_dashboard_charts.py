"""
Unit Tests for Dashboard Plotly Visualizations (Timeline & Radar Charts)
"""

import plotly.graph_objects as go
import pytest

from dashboard.components.radar_chart import create_motion_dna_radar_chart
from dashboard.components.timeline_chart import create_rep_quality_timeline_chart


def test_rep_quality_timeline_chart_empty():
    fig = create_rep_quality_timeline_chart(rep_numbers=[], scores=[])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_rep_quality_timeline_chart_single_rep():
    fig = create_rep_quality_timeline_chart(rep_numbers=[1], scores=[88.0], degradation_threshold=70.0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # 1 bar trace
    assert fig.layout.yaxis.title.text == "Form Score (0-100)"
    assert fig.layout.yaxis.title.font.color == "#94A3B8"


def test_rep_quality_timeline_chart_multiple_reps_with_trend():
    rep_numbers = [1, 2, 3, 4, 5, 6, 7]
    scores = [94.0, 91.0, 95.0, 82.0, 74.0, 68.0, 59.0]
    fig = create_rep_quality_timeline_chart(rep_numbers=rep_numbers, scores=scores, degradation_threshold=70.0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # bar trace + trend scatter trace
    assert fig.layout.yaxis.title.text == "Form Score (0-100)"


def test_motion_dna_radar_chart_baseline_only():
    scores = {
        "knee_rom": 92.0,
        "hip_rom": 88.0,
        "rep_duration": 85.0,
        "symmetry": 96.0,
        "smoothness": 91.0,
        "torso_stability": 89.0,
    }
    fig = create_motion_dna_radar_chart(fingerprint_scores=scores, exercise_name="Squat")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatterpolar"


def test_motion_dna_radar_chart_with_comparison():
    baseline = {"knee_rom": 90.0, "symmetry": 95.0, "smoothness": 90.0}
    comp = {"knee_rom": 75.0, "symmetry": 85.0, "smoothness": 80.0}
    fig = create_motion_dna_radar_chart(fingerprint_scores=baseline, comparison_scores=comp, exercise_name="Squat")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # baseline + current rep
