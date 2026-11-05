# regressors/peaks.py

from __future__ import annotations

import numpy as np

from sign_language_trf.preprocessing.motion_loader import MotionData
from sign_language_trf.preprocessing.markers_loader import MarkerData, MarkerName


def left_hand_move_onset(motion: MotionData, markers: MarkerData) -> np.ndarray:
    """Event times (seconds) of LEFT_HAND_MOVE_ONSET markers."""
    return markers.times_s(MarkerName.LEFT_HAND_MOVE_ONSET)


def left_hand_stop_onset(motion: MotionData, markers: MarkerData) -> np.ndarray:
    """Event times (seconds) of LEFT_HAND_STOP_ONSET markers."""
    return markers.times_s(MarkerName.LEFT_HAND_STOP_ONSET)


def right_hand_move_onset(motion: MotionData, markers: MarkerData) -> np.ndarray:
    """Event times (seconds) of RIGHT_HAND_MOVE_ONSET markers."""
    return markers.times_s(MarkerName.RIGHT_HAND_MOVE_ONSET)


def right_hand_stop_onset(motion: MotionData, markers: MarkerData) -> np.ndarray:
    """Event times (seconds) of RIGHT_HAND_STOP_ONSET markers."""
    return markers.times_s(MarkerName.RIGHT_HAND_STOP_ONSET)


def global_motion_peak(motion: MotionData, markers: MarkerData) -> np.ndarray:
    """Event times (seconds) of GLOBAL_MOTION_PEAK markers."""
    return markers.times_s(MarkerName.GLOBAL_MOTION_PEAK)
