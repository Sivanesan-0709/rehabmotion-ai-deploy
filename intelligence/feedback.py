"""
AI Exercise Form & Posture Intelligence Platform
Multimodal Feedback Engine & Asynchronous TTS Voice Coach
"""

import queue
import threading
import time
from typing import Dict, List, Optional


class VoiceSynthesizerThread(threading.Thread):
    """
    Background daemon thread handling text-to-speech synthesis
    to prevent blocking Computer Vision frame rendering.
    """

    def __init__(self, speech_rate: int = 175, speech_volume: float = 0.9):
        super().__init__(daemon=True)
        self.speech_rate = speech_rate
        self.speech_volume = speech_volume
        self.msg_queue: queue.Queue = queue.Queue(maxsize=10)
        self.is_running = True
        self._engine = None

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.speech_rate)
            self._engine.setProperty("volume", self.speech_volume)
        except Exception as e:
            self._engine = None

    def run(self):
        self._init_engine()
        while self.is_running:
            try:
                msg = self.msg_queue.get(timeout=0.2)
                if msg is None:
                    break
                if self._engine:
                    try:
                        self._engine.say(msg)
                        self._engine.runAndWait()
                    except Exception:
                        # Re-initialize engine if audio device resets
                        self._init_engine()
                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def speak(self, text: str):
        if not self.msg_queue.full():
            self.msg_queue.put(text)

    def stop(self):
        self.is_running = False
        self.msg_queue.put(None)


class MultimodalFeedbackManager:
    """
    Orchestrates visual cues, audio notifications, and cooldown hysteresis.
    """

    def __init__(
        self,
        voice_enabled: bool = False,
        cooldown_seconds: float = 3.0,
        speech_rate: int = 175,
        speech_volume: float = 0.9
    ):
        self.voice_enabled = voice_enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_spoken_time: float = 0.0
        self.last_spoken_message: str = ""
        self.active_banner_message: str = ""
        self.active_banner_expires_at: float = 0.0

        self._tts_worker: Optional[VoiceSynthesizerThread] = None
        if self.voice_enabled:
            self._start_tts_worker(speech_rate, speech_volume)

    def _start_tts_worker(self, rate: int, volume: float):
        if self._tts_worker is None or not self._tts_worker.is_alive():
            self._tts_worker = VoiceSynthesizerThread(speech_rate=rate, speech_volume=volume)
            self._tts_worker.start()

    def set_voice_enabled(self, enabled: bool):
        self.voice_enabled = enabled
        if enabled and self._tts_worker is None:
            self._start_tts_worker(175, 0.9)

    def emit_feedback(
        self,
        message: str,
        priority: str = "normal",  # "urgent", "normal", "info"
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Emits feedback if cooldown period has elapsed or if message is urgent.
        Returns True if feedback was triggered.
        """
        if timestamp is None:
            timestamp = time.time()

        is_urgent = priority == "urgent"
        time_since_last = timestamp - self.last_spoken_time

        # Check cooldown
        if not is_urgent and time_since_last < self.cooldown_seconds:
            return False

        # Prevent immediate repeating of the exact same message
        if message == self.last_spoken_message and time_since_last < (self.cooldown_seconds * 1.5):
            return False

        self.last_spoken_time = timestamp
        self.last_spoken_message = message
        self.active_banner_message = message
        self.active_banner_expires_at = timestamp + 3.0

        if self.voice_enabled and self._tts_worker:
            self._tts_worker.speak(message)

        return True

    def get_active_banner(self, current_time: float) -> Optional[str]:
        if current_time < self.active_banner_expires_at:
            return self.active_banner_message
        return None

    def close(self):
        if self._tts_worker:
            self._tts_worker.stop()
            self._tts_worker = None
