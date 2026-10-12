"""
AI Exercise Form & Posture Intelligence Platform
Asynchronous Multi-Backend Voice Coach Engine & Observer Dispatcher
"""

import json
import os
import platform
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Word mapping for numbers 1-50
NUMBER_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
    11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen",
    16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty",
    21: "Twenty one", 22: "Twenty two", 23: "Twenty three", 24: "Twenty four", 25: "Twenty five",
    26: "Twenty six", 27: "Twenty seven", 28: "Twenty eight", 29: "Twenty nine", 30: "Thirty",
    35: "Thirty five", 40: "Forty", 45: "Forty five", 50: "Fifty",
}

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "config" / "user_settings.json"


def number_to_speech(num: int) -> str:
    """Converts integer rep numbers to clear spoken phonetic words."""
    if num in NUMBER_WORDS:
        return NUMBER_WORDS[num]
    return str(num)


@dataclass(order=True)
class VoiceUtterance:
    priority: int  # 1: Rep count (highest), 2: Set complete, 3: Form issue, 4: Encouragement/Milestone
    text: str = field(compare=False)
    rep_number: Optional[int] = field(default=None, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)


class VoiceCoach:
    """
    Asynchronous Thread-Safe Voice Coach with Priority Queue and Stale-Rep Dropping.
    Supports pyttsx3, Windows SAPI (PowerShell), and Mock fallbacks.
    """

    def __init__(self):
        self.settings: Dict[str, Any] = self._load_settings()
        self.enabled: bool = self.settings.get("voice_enabled", True)
        self.mode: str = self.settings.get("voice_mode", "Coach")  # "Count only", "Coach", "Silent"
        self.rate: int = self.settings.get("speech_rate", 180)
        self.volume: float = self.settings.get("speech_volume", 0.9)
        self.voice_index: int = self.settings.get("voice_index", 0)
        self.announce_interval: int = self.settings.get("announce_interval", 1)  # 1: every rep, 5: every 5 reps

        self.last_announced_rep: int = 0
        self.last_spoken_text: str = ""
        self.recent_log: List[Dict[str, Any]] = []
        self.is_currently_speaking: bool = False

        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=15)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._active_backend: str = "pyttsx3"

        self._start_worker()

    def _load_settings(self) -> Dict[str, Any]:
        default_settings = {
            "voice_enabled": True,
            "voice_mode": "Coach",
            "speech_rate": 180,
            "speech_volume": 0.9,
            "voice_index": 0,
            "announce_interval": 1,
            "camera_index": 0,
            "theme_accent": "Sky Blue",
        }
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    default_settings.update(data)
            except Exception:
                pass
        return default_settings

    def save_settings(self):
        """Persists voice and app preferences to config/user_settings.json."""
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass

    def get_available_voices(self) -> List[str]:
        """Returns the list of installed TTS voice names."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            if voices:
                return [v.name for v in voices]
        except Exception:
            pass
        return ["Default System Voice", "Voice 2", "Voice 3"]

    def _start_worker(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self):
        # Initialize COM on Windows for thread-safe SAPI
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        # Backend 1: pyttsx3
        engine = None
        backend = "mock"
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            voices = engine.getProperty("voices")
            if voices and len(voices) > self.voice_index:
                engine.setProperty("voice", voices[self.voice_index].id)
            backend = "pyttsx3"
        except Exception:
            engine = None
            if platform.system() == "Windows":
                backend = "sapi_powershell"

        self._active_backend = backend

        while not self._stop_event.is_set():
            try:
                utterance: VoiceUtterance = self._queue.get(timeout=0.2)
                if utterance is None:
                    break

                self.is_currently_speaking = True
                self.last_spoken_text = utterance.text

                # Log for UI indicator
                self.recent_log.append({
                    "text": utterance.text,
                    "priority": utterance.priority,
                    "time": time.strftime("%H:%M:%S"),
                })
                if len(self.recent_log) > 5:
                    self.recent_log.pop(0)

                # Execute Speech Synthesis
                if self.enabled and self.mode != "Silent":
                    if backend == "pyttsx3" and engine is not None:
                        try:
                            engine.say(utterance.text)
                            engine.runAndWait()
                        except Exception:
                            # Re-initialize engine on audio hardware reset
                            try:
                                import pyttsx3
                                engine = pyttsx3.init()
                                engine.setProperty("rate", self.rate)
                                engine.setProperty("volume", self.volume)
                                engine.say(utterance.text)
                                engine.runAndWait()
                            except Exception:
                                pass
                    elif backend == "sapi_powershell":
                        try:
                            clean_text = utterance.text.replace("'", "").replace('"', "")
                            cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_text}')"
                            subprocess.run(["powershell", "-Command", cmd], capture_output=True, timeout=5)
                        except Exception:
                            pass

                self.is_currently_speaking = False
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                self.is_currently_speaking = False

    def say(self, text: str, priority: int = 3, rep_number: Optional[int] = None):
        """
        Enqueues an utterance. If a new rep number arrives while an older rep number
        is still pending in the queue, drops the stale rep number to prevent speech lag.
        """
        if not self.enabled or self.mode == "Silent" or not text:
            return

        # Stale-Rep Dropping: If this is a rep announcement, purge any older pending rep announcements
        if priority == 1 and rep_number is not None:
            new_items = []
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item.priority == 1 and item.rep_number is not None and item.rep_number < rep_number:
                        # Drop stale older rep number
                        self._queue.task_done()
                        continue
                    new_items.append(item)
                except queue.Empty:
                    break
            # Put back non-stale items
            for it in new_items:
                self._queue.put(it)

        if not self._queue.full():
            self._queue.put(VoiceUtterance(priority=priority, text=text, rep_number=rep_number))

    # ==========================================================================
    # OBSERVER PATTERN: REAL-TIME REP & MILESTONE CALLOUTS
    # ==========================================================================
    def check_and_announce_rep(
        self,
        current_rep_count: int,
        form_score: float = 85.0,
        active_issue: Optional[str] = None,
        target_reps: int = 20,
    ):
        """
        Monitors rep counter state changes and triggers synchronized speech callouts.
        Guarantees:
        - Only announces when count strictly increases
        - Handles count jumps gracefully (announces only latest)
        - Milestone announcements at halfway, last 3 reps, and target completion
        """
        if current_rep_count <= self.last_announced_rep:
            return

        # Count increased!
        self.last_announced_rep = current_rep_count

        # Check frequency setting (every rep vs every 5th rep)
        should_speak = (self.announce_interval == 1) or (current_rep_count % self.announce_interval == 0) or (current_rep_count >= target_reps)
        if not should_speak:
            return

        spoken_num = number_to_speech(current_rep_count)

        if self.mode == "Count only":
            self.say(spoken_num, priority=1, rep_number=current_rep_count)

        elif self.mode == "Coach":
            # Coach mode: Spoken number + short coaching cue
            if current_rep_count >= target_reps:
                cue = f"Set complete. Target reached with {current_rep_count} reps. Excellent work!"
                self.say(f"{spoken_num}. {cue}", priority=1, rep_number=current_rep_count)
            elif active_issue and form_score < 75.0:
                short_issue = active_issue.split(" - ")[0] if " - " in active_issue else active_issue
                self.say(f"{spoken_num}. {short_issue}", priority=1, rep_number=current_rep_count)
            elif current_rep_count == (target_reps // 2) and target_reps > 4:
                self.say(f"{spoken_num}. Halfway there.", priority=1, rep_number=current_rep_count)
            elif current_rep_count == (target_reps - 3) and target_reps > 5:
                self.say(f"{spoken_num}. Three more reps.", priority=1, rep_number=current_rep_count)
            elif form_score >= 90.0:
                self.say(f"{spoken_num}. Perfect form.", priority=1, rep_number=current_rep_count)
            elif form_score >= 80.0:
                self.say(f"{spoken_num}. Good depth.", priority=1, rep_number=current_rep_count)
            else:
                self.say(spoken_num, priority=1, rep_number=current_rep_count)

    def announce_lifecycle(self, event: str):
        """Announces workout session lifecycle events (start, pause, resume, stop)."""
        messages = {
            "start": "Workout session started. Stand in full view of the camera.",
            "pause": "Workout paused.",
            "resume": "Resuming workout.",
            "stop": "Workout session complete. Great effort!",
            "camera_connected": "Camera stream connected.",
        }
        msg = messages.get(event)
        if msg and self.mode == "Coach":
            self.say(msg, priority=2)

    def reset_counter(self):
        """Silently resets rep tracking state on exercise change or manual reset."""
        self.last_announced_rep = 0

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        mode: Optional[str] = None,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        voice_index: Optional[int] = None,
        announce_interval: Optional[int] = None,
    ):
        """Updates internal settings and persists to disk."""
        if enabled is not None:
            self.enabled = enabled
            self.settings["voice_enabled"] = enabled
        if mode is not None:
            self.mode = mode
            self.settings["voice_mode"] = mode
        if rate is not None:
            self.rate = rate
            self.settings["speech_rate"] = rate
        if volume is not None:
            self.volume = volume
            self.settings["speech_volume"] = volume
        if voice_index is not None:
            self.voice_index = voice_index
            self.settings["voice_index"] = voice_index
        if announce_interval is not None:
            self.announce_interval = announce_interval
            self.settings["announce_interval"] = announce_interval

        self.save_settings()

    def speak_cue(self, text: str, priority: int = 2):
        """Dispatches a coaching or diagnostic cue to the voice queue."""
        self.say(text, priority=priority)

    def test_voice(self):
        """Emits a sample test phrase."""
        self.say("Voice Coach active. One, two, three. Form looks strong!", priority=1)

    def stop(self):
        """Flushes all queued audio utterances immediately."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    @property
    def available_voices(self) -> List[str]:
        return self.get_available_voices()

    @property
    def engine_type(self) -> str:
        return self._active_backend

    def close(self):
        """Stops background synthesis thread cleanly."""
        self._stop_event.set()
        self._queue.put(None)


# Global Voice Coach Singleton
_VOICE_COACH_INSTANCE: Optional[VoiceCoach] = None


def get_voice_coach() -> VoiceCoach:
    """Returns application singleton VoiceCoach instance cached across Streamlit reruns."""
    global _VOICE_COACH_INSTANCE
    if _VOICE_COACH_INSTANCE is None:
        _VOICE_COACH_INSTANCE = VoiceCoach()
    return _VOICE_COACH_INSTANCE
