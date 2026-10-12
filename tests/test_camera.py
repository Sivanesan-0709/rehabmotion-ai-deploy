"""
Unit Tests for Camera Ingestion Layer & Hardware Adapter
Tests backend fallbacks, safe resolution negotiation, frame validation, and diagnostics using mocks.
"""

from unittest.mock import MagicMock, call, patch
import cv2
import numpy as np
import pytest

from cv.detector import VideoStream, diagnose_camera


def test_synthetic_stream():
    stream = VideoStream(is_synthetic=True, width=640, height=480, target_fps=30)
    assert stream.is_open is True
    assert stream.selected_backend_name == "SYNTHETIC"

    ret, frame, t_stamp = stream.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert t_stamp > 0.0

    stream.release()
    assert stream.is_open is False


def test_camera_successful_first_backend():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    mock_cap.read.return_value = (True, test_frame)
    mock_cap.get.side_effect = lambda prop: 1280.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (720.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30.0)

    with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
        stream = VideoStream(src=0, width=1280, height=720, target_fps=30, is_synthetic=False)
        assert stream.is_open is True
        assert stream.actual_width == 1280
        assert stream.actual_height == 720

        ret, frame, t_stamp = stream.read()
        assert ret is True
        assert frame is not None
        assert frame.shape == (720, 1280, 3)

        stream.release()
        mock_cap.release.assert_called()


def test_camera_backend_fallback_on_isopened_false():
    mock_cap_fail = MagicMock()
    mock_cap_fail.isOpened.return_value = False

    mock_cap_success = MagicMock()
    mock_cap_success.isOpened.return_value = True
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap_success.read.return_value = (True, test_frame)
    mock_cap_success.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 480.0

    # First backend fails isOpened, second backend succeeds
    with patch("cv2.VideoCapture", side_effect=[mock_cap_fail, mock_cap_success]):
        stream = VideoStream(src=0, width=1280, height=720, is_synthetic=False)
        assert stream.is_open is True
        assert stream.actual_width == 640
        assert stream.actual_height == 480
        mock_cap_fail.release.assert_called()
        stream.release()


def test_camera_resolution_setting_exception_handled_safely():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Simulate C++ exception on set(CAP_PROP_FRAME_HEIGHT)
    def fake_set(prop, val):
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            raise cv2.error("Simulated Unknown C++ exception from OpenCV code")
        return True

    mock_cap.set.side_effect = fake_set
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, test_frame)
    mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 480.0

    with patch("cv2.VideoCapture", return_value=mock_cap):
        stream = VideoStream(src=0, width=1280, height=720, is_synthetic=False)
        # Must not crash! Continues with native camera resolution
        assert stream.is_open is True
        assert stream.actual_width == 640
        assert stream.actual_height == 480
        stream.release()


def test_camera_invalid_first_frame_triggers_fallback():
    mock_cap_corrupt = MagicMock()
    mock_cap_corrupt.isOpened.return_value = True
    mock_cap_corrupt.read.return_value = (False, None)  # Corrupt first frame

    mock_cap_good = MagicMock()
    mock_cap_good.isOpened.return_value = True
    good_frame = np.ones((720, 1280, 3), dtype=np.uint8)
    mock_cap_good.read.return_value = (True, good_frame)
    mock_cap_good.get.side_effect = lambda prop: 1280.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 720.0

    with patch("cv2.VideoCapture", side_effect=[mock_cap_corrupt, mock_cap_good]):
        stream = VideoStream(src=0, is_synthetic=False)
        assert stream.is_open is True
        mock_cap_corrupt.release.assert_called()
        stream.release()


def test_camera_all_backends_fail():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        stream = VideoStream(src=99, is_synthetic=False)
        assert stream.is_open is False
        assert stream.cap is None
        ret, frame, _ = stream.read()
        assert ret is False
        assert frame is None


def test_camera_diagnostics():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, test_frame)
    mock_cap.get.side_effect = lambda prop: 1280.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 720.0

    with patch("cv2.VideoCapture", return_value=mock_cap):
        diag = diagnose_camera(0)
        assert diag["is_opened"] is True
        assert diag["device_index"] == 0
        assert "backend_selected" in diag
        assert diag["first_frame"] == "OK"
        assert diag["frame_shape"] == (720, 1280, 3)
