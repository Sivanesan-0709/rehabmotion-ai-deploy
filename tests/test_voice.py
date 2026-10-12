"""
Unit Tests for Voice Coach Engine & Observer Dispatcher
Tests priority queue, stale-rep dropping, number phonetic conversion, mute modes, and error resilience.
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from dashboard.voice import VoiceCoach, VoiceUtterance, number_to_speech


def test_number_to_speech_conversion():
    assert number_to_speech(1) == "One"
    assert number_to_speech(5) == "Five"
    assert number_to_speech(10) == "Ten"
    assert number_to_speech(25) == "Twenty five"
    assert number_to_speech(50) == "Fifty"
    assert number_to_speech(77) == "77"


def test_voice_coach_rep_announcements_in_order():
    coach = VoiceCoach()
    coach.enabled = True
    coach.mode = "Count only"
    coach.reset_counter()

    # Capture say calls
    with patch.object(coach, "say") as mock_say:
        for rep in range(1, 11):
            coach.check_and_announce_rep(rep)

        assert mock_say.call_count == 10
        # Verify first call is One and 10th call is Ten
        assert mock_say.call_args_list[0][0][0] == "One"
        assert mock_say.call_args_list[9][0][0] == "Ten"

    coach.close()


def test_voice_coach_no_duplicate_on_repeated_poll():
    coach = VoiceCoach()
    coach.enabled = True
    coach.reset_counter()

    with patch.object(coach, "say") as mock_say:
        # Polling rep 3 multiple times
        coach.check_and_announce_rep(3)
        coach.check_and_announce_rep(3)
        coach.check_and_announce_rep(3)

        assert mock_say.call_count == 1

    coach.close()


def test_voice_coach_count_jump():
    coach = VoiceCoach()
    coach.enabled = True
    coach.mode = "Count only"
    coach.reset_counter()

    with patch.object(coach, "say") as mock_say:
        coach.check_and_announce_rep(3)
        # Jumps directly from 3 to 5
        coach.check_and_announce_rep(5)

        assert mock_say.call_count == 2
        assert mock_say.call_args_list[0][0][0] == "Three"
        assert mock_say.call_args_list[1][0][0] == "Five"

    coach.close()


def test_voice_coach_reset_and_restart():
    coach = VoiceCoach()
    coach.enabled = True
    coach.mode = "Count only"
    coach.reset_counter()

    with patch.object(coach, "say") as mock_say:
        coach.check_and_announce_rep(5)
        assert mock_say.call_count == 1

        # Resetting counter
        coach.reset_counter()
        assert coach.last_announced_rep == 0

        # Announce rep 1 after reset
        coach.check_and_announce_rep(1)
        assert mock_say.call_count == 2
        assert mock_say.call_args_list[1][0][0] == "One"

    coach.close()


def test_voice_coach_silent_mode_produces_no_speech():
    coach = VoiceCoach()
    coach.enabled = True
    coach.mode = "Silent"
    coach.reset_counter()

    # Under silent mode, say() should do nothing
    coach.say("One", priority=1)
    assert coach._queue.empty()

    coach.close()


def test_voice_coach_stale_rep_dropping():
    coach = VoiceCoach()
    coach.enabled = True
    coach.mode = "Count only"

    # Enqueue rep 1, rep 2, rep 3
    coach.say("One", priority=1, rep_number=1)
    coach.say("Two", priority=1, rep_number=2)
    # Enqueue rep 4 - should drop 1, 2
    coach.say("Four", priority=1, rep_number=4)

    items = []
    while not coach._queue.empty():
        items.append(coach._queue.get())

    rep_nums = [it.rep_number for it in items if it.rep_number is not None]
    assert 1 not in rep_nums
    assert 2 not in rep_nums
    assert 4 in rep_nums

    coach.close()


def test_voice_coach_backend_resilience():
    coach = VoiceCoach()
    coach.enabled = True

    # Test that invalid inputs or synthesizer errors don't crash
    coach.say(None)
    coach.say("")
    coach.check_and_announce_rep(-1)

    coach.close()
