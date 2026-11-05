# regressors/velocity.py

from __future__ import annotations

import numpy as np

from sign_language_trf.preprocessing.motion_loader import MotionData
from sign_language_trf.preprocessing.markers_loader import MarkerData


def left_hand_speed(
    motion: MotionData,
    markers: MarkerData,
    smooth: bool = False,
    savgol_window: int = 9,
    savgol_poly: int = 3,
) -> np.ndarray:
    spd = motion.speed("LeftHand")
    if smooth:
        from scipy.signal import savgol_filter
        spd = savgol_filter(spd, window_length=savgol_window, polyorder=savgol_poly)
        spd = np.maximum(spd, 0.0)  # clamp rare negative artefacts from smoothing edges
    return spd


def right_hand_speed(
    motion: MotionData,
    markers: MarkerData,
    smooth: bool = False,
    savgol_window: int = 9,
    savgol_poly: int = 3,
) -> np.ndarray:
    spd = motion.speed("RightHand")
    if smooth:
        from scipy.signal import savgol_filter
        spd = savgol_filter(spd, window_length=savgol_window, polyorder=savgol_poly)
        spd = np.maximum(spd, 0.0)
    return spd
