# UI/UX & Feature Showcase — "No Dead Buttons" Verification Checklist

**Project:** AI Exercise Form & Posture Intelligence Platform  
**Target:** Academic Viva & Live Webcam Product Showcase  
**Status:** 100% Interactive Coverage (All widgets have active handlers)

---

## 1. Athlete Command Center (`dashboard/pages/home.py`)
- [x] **Active Athlete Profile Selectbox:** Switches active user in SQLite `AppState`.
- [x] **➕ Add Profile Button:** Opens interactive user creation dialog; saves to SQLite and refreshes.
- [x] **Top KPI Cards (Reps, Avg Form, Streak, Calories):** Real-time aggregation from SQLite with fallback to calibrated baseline.
- [x] **Quick Start Exercise Selectbox:** Selects target exercise.
- [x] **Quick Target Reps Input:** Configures workout goal.
- [x] **🚀 Start Coach Button:** Updates session state target and routes directly to Live Workout Coach.
- [x] **Weekly Training Bar Chart:** Interactive Plotly chart with daily volume and form score gradient.
- [x] **Quick Tools Navigation Buttons (🧬 View Motion DNA, 📋 Workout Planner, 📈 Rep Timeline, 🤖 AI Assistant):** Direct inter-page routing.
- [x] **Voice Coach Quick Controller:** Inline toggle and audio test button.
- [x] **Recent Workout Sessions Table:** Displays live database sessions or sample baseline with `DEMO DATA` badge.

---

## 2. Live AI Workout Coach (`dashboard/pages/live_coach.py`)
- [x] **Athlete Profile & Manual/Auto-Detect Radios:** Full state machine control.
- [x] **Video Source Radio (Webcam / Synthetic / Upload):** Switches OpenCV DirectShow hardware capture, synthetic kinetic generator, or video upload.
- [x] **▶ START SESSION Button:** Initiates SQLite session, dispatches opening voice cue ("Starting Squat session..."), activates recording.
- [x] **⏹ END SESSION Button:** Finalizes session metrics, logs summary to SQLite, opens celebratory post-session summary modal, speaks completion audio.
- [x] **🔄 Reset Counter Button:** Clears real-time rep counter to 0.
- [x] **Big Glow Rep Counter Display:** Synchronized visually with real-time pose state machine.
- [x] **In-HUD Voice Quick Toggle / Mute:** Real-time audio muting without stopping camera frame processing.
- [x] **Live Rep Quality Timeline Plotly Chart:** Real-time per-rep scores with Plotly 5.24+ `title=dict(text=...)` compatibility.

---

## 3. Exercise Library & Guide (`dashboard/pages/exercise_library.py`)
- [x] **Search Input & Muscle Group Filter:** Real-time client filtering of exercise cards.
- [x] **🔍 View Cues & Mistakes Modal:** Opens detailed anatomy breakdown, target angles, primary cues, and common mistakes.
- [x] **🏋️ Start Workout Button:** Sets selected exercise in `AppState` and jumps to Live Coach.
- [x] **📋 Add to Planner Button:** Appends exercise to daily routine in Workout Planner with confirmation toast.

---

## 4. Workout Routine Planner (`dashboard/pages/workout_planner.py`)
- [x] **Day-of-Week Tabs (Monday – Sunday):** Selects daily routine schedule.
- [x] **Exercise, Sets, Target Reps & Target Form Controls:** Add custom movements to plan.
- [x] **➕ Add Exercise to Plan Button:** Persists routine and updates UI.
- [x] **🤖 Generate AI Balanced Weekly Split Button:** Generates optimal push/pull/legs split with confetti feedback.
- [x] **🚀 Start Today's Routine Button:** Sets today's exercise and routes to Live Coach.
- [x] **📥 Download Workout Plan (CSV):** Exports active routine table as CSV.

---

## 5. Personalized Motion DNA (`dashboard/pages/motion_dna.py`)
- [x] **User Profile & Exercise Selectors:** Re-evaluates baseline from database.
- [x] **Biomechanical Radar Chart:** Multi-dimensional kinematic fingerprint (ROM, symmetry, tempo, stability, inflection).
- [x] **Baseline Distribution Table:** Mean, standard deviation, median, IQR, and observed range.
- [x] **Repetition Comparator Dropdown:** Compares any past rep against baseline with Z-score deviation matrix.
- [x] **⚡ Generate Live Baseline in DB Button:** Seeds verified calibration reps into SQLite for viva evaluation.
- [x] **🏋️ Start Live Coach to Calibrate Button:** Direct route to Live Coach.

---

## 6. Rep Quality Timeline (`dashboard/pages/rep_quality.py`)
- [x] **Athlete & Exercise Selectors:** Filters historical repetitions.
- [x] **Fatigue Consistency Degradation Alert:** Triggers warning banner if 4-rep sliding window drops >12%.
- [x] **Interactive Plotly Timeline Chart:** Visualizes score trend line.
- [x] **Repetition Breakdown Matrix:** ROM, Symmetry, Smoothness, Duration, and Issues.
- [x] **📥 Download Rep Telemetry (CSV):** Full dataset download.

---

## 7. Progress Analytics (`dashboard/pages/analytics.py`)
- [x] **KPI Summary Cards:** Total sessions, reps, avg score, training duration.
- [x] **Historical Form Progression Line Chart:** Multi-exercise trend over time.
- [x] **Exercise Distribution Donut Chart:** Breakdown of training volume.
- [x] **Bilateral Symmetry Histogram:** Left vs. Right balance frequency.
- [x] **Smoothness vs. Range of Motion Scatter Plot:** Kinematic stability correlation.

---

## 8. Academic Benchmarks (`dashboard/pages/benchmarks.py`)
- [x] **Session-Aware Data Leakage Protocol Explainer:** Clarifies academic evaluation methodology.
- [x] **Multi-Model Comparison Table:** Accuracy, Precision, Recall, F1, Latency for RF, SVM, MLP, GB, LSTM.
- [x] **Accuracy vs. Latency Tradeoff Plot:** Renders benchmark figure from `artifacts/`.
- [x] **Multi-Class Confusion Matrix Inspector:** Selectbox switching between models.
- [x] **Detailed Classification Reports Tabs:** Precision/Recall by flaw class.
- [x] **📥 Download Academic Benchmark Data (CSV):** Exports benchmark table.

---

## 9. Workout Session History (`dashboard/pages/history.py`)
- [x] **Session Selectbox:** Browse all past workouts.
- [x] **AI Form Intelligence Insights:** Rule-based narrative evaluation.
- [x] **Repetition Telemetry Table:** Complete per-rep kinematic log.
- [x] **📥 Download Session Rep Data (CSV):** Single session data export.
- [x] **🗑️ Delete Workout Session Button:** Permanent SQLite deletion.

---

## 10. Session Reports & Export (`dashboard/pages/reports.py`)
- [x] **Session Selector:** Choose session to generate audit.
- [x] **📄 Export CSV Rep Telemetry:** CSV format download.
- [x] **🧩 Export JSON Kinematics:** JSON structured telemetry download.
- [x] **🌐 Export Printable HTML Report:** Complete standalone HTML medical/athletic report.
- [x] **📧 Send Report to Coach Simulation:** Email input with validation toast and audio confirmation.

---

## 11. Achievements & Badges (`dashboard/pages/achievements.py`)
- [x] **Milestone Badges Grid:** Centurion, Form Master, Iron Spine, Consistency King, etc.
- [x] **Weekly Rep Goal & Form Floor Sliders:** Interactive goal configuration.
- [x] **💾 Save Goals Button:** Persists goals with Voice Coach confirmation.
- [x] **🎉 Celebrate Milestone Button:** Triggers `st.balloons()` and audio praise.
- [x] **Weekly Streak Consistency Bar Chart:** Adherence tracking.

---

## 12. AI Coach Assistant (`dashboard/pages/ai_coach.py`)
- [x] **Suggested Question Prompt Chips:** 4 instant chips for knee valgus, lumbar spine, pushup depth, symmetry.
- [x] **Natural Language Chat Stream:** User & AI message stream.
- [x] **🔊 Read Aloud Button:** Dispatches response summary to asynchronous Voice Coach.
- [x] **`st.chat_input` Handler:** Generates domain-aware biomechanical advice.
- [x] **🗑️ Clear Conversation Button:** Resets chat thread.

---

## 13. Community Leaderboard (`dashboard/pages/leaderboard.py`)
- [x] **Exercise & Timeframe Filters:** Dynamic ranking updates.
- [x] **Top 3 Podium Display (Gold, Silver, Bronze):** Highlighted cards.
- [x] **Complete Standings Table:** Displays active user position in rank 4.
- [x] **⚔️ Issue Challenge Controls:** Select athlete, set target reps, dispatch challenge toast and voice cue.

---

## 14. Architecture & Viva Guide (`dashboard/pages/architecture.py`)
- [x] **7-Stage Pipeline Visual Cards:** End-to-end architecture walkthrough.
- [x] **Biomechanical LaTeX Math Tab:** 3D joint angle dot product, BSI formula, Form Score penalty sum, Motion DNA normalization.
- [x] **Academic Viva Q&A Accordion:** Top 5 viva defense questions & answers.
- [x] **🚀 Run Full System Diagnostic Button:** Live tests SQLite, Voice engine, pose landmarker, and exercise FSMs.
- [x] **📥 Download Viva Architecture Whitepaper (.md):** Complete markdown whitepaper export.

---

## 15. Platform Settings (`dashboard/pages/settings.py`)
- [x] **📷 Test Hardware Camera Button:** Tests DirectShow camera availability and resolution.
- [x] **🔊 Voice Coach Settings:** Toggle TTS, rate slider (100–300 WPM), volume slider, voice selector dropdown.
- [x] **🔊 Play Audio Test Button:** Dispatches test phrase to background queue.
- [x] **Kinematic Sensitivity Sliders:** ROM tolerance, spine angle warning threshold, cadence buffer.
- [x] **💾 Save Configuration Button:** Persists to `config/user_settings.json`.
- [x] **🧹 Clear User Workout History & ⚠️ Factory Reset Buttons:** Interactive confirmation dialogs.

---

## 16. Help Center & Tour (`dashboard/pages/help_tour.py`)
- [x] **4-Step Visual Tour:** Camera placement, exercise selection, live HUD, Motion DNA analysis.
- [x] **🚀 Jump to Live Coach Button:** Direct route to Live Coach.
- [x] **🔊 Voice Speech & 🔇 Stop Audio Buttons:** Diagnostic audio queue control.
- [x] **📥 Load 10 Demo Sessions into DB Button:** Seeds SQLite database for viva demonstrations.
- [x] **🧹 Clear User Sessions Button:** Clears database records.
- [x] **Technical FAQ Accordions:** 4 expandable technical answers.
