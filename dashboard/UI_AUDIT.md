# AI Exercise Intelligence Platform — Comprehensive UI/UX & Voice Audit

**Date:** 2026-10-08  
**Auditor:** Antigravity AI Senior Engineer  
**Status:** Pre-Overhaul Baseline Audit  

---

## 1. Existing Pages & Interactive Widget Audit

### Page 1: Live Workout Coach (`dashboard/pages/live_coach.py`)
- **Athlete Profile Selectbox:** Works (Updates active user in AppState).
- **Exercise Mode Radio (Auto/Manual):** Works (Toggles auto-detection flag).
- **Manual Exercise Selectbox:** Works (Updates state.exercise_name).
- **Pose Model Selectbox (MediaPipe/YOLO):** Works (Updates pose estimator backend).
- **Video Input Source Radio (Webcam/Synthetic/Upload):** Works (Initializes VideoStream).
- **File Uploader (Video):** Works (Saves temp video file for stream).
- **Voice Coach Checkbox:** Partial (Sets internal flag, but lacks real-time rep voice callouts in Streamlit loop).
- **Feedback Cooldown Slider:** Works (Adjusts hysteresis interval).
- **START WORKOUT SESSION Button:** Works (Initializes session in DB).
- **END & SAVE SESSION Button:** Works (Finalizes session and builds baseline).
- **Pause Stream Button:** Works (Terminates while loop).
- **Rep Quality Timeline Chart:** Works (Plotly visualization updated per frame).
- **Missing Elements:** Big on-screen rep counter synced with voice announcements, in-HUD mute button, speaking status chip, and post-session summary modal.

### Page 2: Motion DNA / Fingerprint (`dashboard/pages/motion_dna.py`)
- **User Profile Selectbox:** Works.
- **Exercise Selectbox:** Works.
- **Generate Calibrated Baseline Button:** Works (Builds synthetic profile if no history).
- **Radar Chart:** Works (Plotly polar chart).
- **Distribution Table:** Works (Pandas dataframe display).
- **Rep Selectbox Comparator:** Works (Calculates Z-score deviation breakdown).
- **Missing Elements:** Fatigue meter, injury-risk gauge, ideal-vs-you visualization, export baseline button.

### Page 3: Rep Quality Timeline (`dashboard/pages/rep_quality.py`)
- **User Selectbox:** Works.
- **Exercise Selectbox:** Works.
- **Timeline Chart:** Works (Plotly bar + moving average trendline).
- **Consistency Alert Banner:** Works (Triggers if degradation detected).
- **Repetition Matrix Table:** Works (Pandas dataframe).
- **Missing Elements:** CSV export button, date filter, filter by correct/flawed status.

### Page 4: Progress Analytics (`dashboard/pages/analytics.py`)
- **User Selectbox:** Works.
- **Historical Form Line Chart:** Works.
- **Exercise Distribution Pie Chart:** Works.
- **Bilateral Symmetry Histogram:** Works.
- **Smoothness vs ROM Scatter Plot:** Works.
- **Missing Elements:** Time-range filter (7d/30d/All), date range picker, export CSV report, empty state CTA to load demo data.

### Page 5: Academic Benchmarks (`dashboard/pages/benchmarks.py`)
- **Methodology Expander:** Works.
- **Model Comparison Table:** Works (Reads from `artifacts/model_comparison.csv`).
- **Benchmark Chart Image:** Works (Reads from `artifacts/model_comparison.png`).
- **Retrain Benchmark Models Button:** Works (Calls AcademicEvaluator).
- **Confusion Matrix Selectbox:** Works (Displays per-model confusion matrix image).
- **Missing Elements:** Metric filter tabs (Accuracy/F1/Latency/Params), export benchmark report.

### Page 6: Workout Session History (`dashboard/pages/history.py`)
- **User Selectbox:** Works.
- **Exercise Filter Selectbox:** Works.
- **Session List & Metrics Cards:** Works.
- **AI Session Insights Generator:** Works (Rule-based kinematic analysis).
- **Session Reps Dataframe:** Works.
- **Missing Elements:** Interactive session detail dialog/modal, delete session with confirmation, export session report.

### Page 7: Dataset Collection Studio (`dashboard/pages/data_studio.py`)
- **Target Exercise Selectbox:** Works.
- **Form Quality Label Selectbox:** Works.
- **Target Frames Slider:** Works.
- **Participant & Session ID Inputs:** Works.
- **START RECORDING DATA Button:** Works (Captures frames, extracts features, exports CSV).
- **PAUSE / STOP Button:** Partial (Passive button in UI).
- **Existing Dataset Viewer:** Works.
- **Missing Elements:** Class balance distribution chart, export zip package.

---

## 2. Voice Architecture Audit

### Standalone OpenCV Runner (`run.py --voice`)
- **Implementation:** Uses `MultimodalFeedbackManager` (`intelligence/feedback.py`).
- **TTS Engine:** Uses `VoiceSynthesizerThread` running `pyttsx3` asynchronously in a background daemon thread.
- **Behavior:** Speaks qualitative form issue messages (e.g., "Knee valgus detected", "Good rep! Rep 1").
- **Limitations:**
  - Lacks dedicated, high-priority spoken numbers ("One", "Two", "Three") immediately upon rep completion.
  - Lacks speaker volume / rate controls and voice selection in UI.
  - Lacks multi-backend fallbacks (pyttsx3 -> Windows SAPI -> Web Speech API -> Mock).
  - Web UI lacked an in-memory voice manager with stale rep queue dropping.

---

## 3. Plan of Action

1. **Build `dashboard/theme.py`:** Centralize design tokens, typography, dark glassmorphism CSS, KPI cards, status pills, and safe-render wrapper.
2. **Build `dashboard/demo/demo_data.py`:** Create isolated synthetic demo profiles, plans, notifications, and leaderboards.
3. **Build `dashboard/voice.py` & `dashboard/components/voice_controls.py`:** Implement multi-backend `VoiceCoach` with prioritized thread queue, stale-rep dropping, and user settings persistence.
4. **Implement New Pages:** Home Command Center, Exercise Library, Workout Planner, Settings, Architecture Viva Guide, Achievements, AI Coach Assistant, Reports, Leaderboard, Help/Tour.
5. **Upgrade Existing Pages:** Wrap with `safe_render`, add interactive dialogs, downloads, filters, and HUD voice synchronization.
6. **Add Automated Test Suite:** Unit tests for voice coach (`test_voice.py`) and UI components (`test_dashboard_ui.py`).
