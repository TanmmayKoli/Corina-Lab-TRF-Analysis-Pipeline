# regressors/acceleration.py

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from sign_language_trf.preprocessing.motion_loader import MotionData
from sign_language_trf.preprocessing.markers_loader import MarkerData
from sign_language_trf.preprocessing.raw_rokoko_loader import RawRokokoData


# ===========================================================================
# Acceleration magnitude  (continuous)
# ===========================================================================

def left_hand_accel(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    return raw.accel_magnitude("LeftHand")


def right_hand_accel(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Right hand acceleration magnitude (m/s²) at raw.fs."""
    return raw.accel_magnitude("RightHand")


def left_forearm_accel(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Left forearm acceleration magnitude (m/s²) at raw.fs."""
    return raw.accel_magnitude("LeftForeArm")


def right_forearm_accel(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Right forearm acceleration magnitude (m/s²) at raw.fs."""
    return raw.accel_magnitude("RightForeArm")


def left_shoulder_accel(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Left shoulder acceleration magnitude (m/s²) at raw.fs."""
    return raw.accel_magnitude("LeftShoulder")


def right_shoulder_accel(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Right shoulder acceleration magnitude (m/s²) at raw.fs."""
    return raw.accel_magnitude("RightShoulder")


def combined_hand_accel(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    return raw.accel_magnitude("LeftHand") + raw.accel_magnitude("RightHand")


# ===========================================================================
# Upper-arm speed  (continuous)
# ===========================================================================
# LeftUpperArm / RightUpperArm are NOT in the processed CSV — only available
# here via the raw file.  Upper-arm speed captures proximal arm movement
# independently of the wrist/hand, useful for the body hierarchy experiment.

def left_upper_arm_speed(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Left upper arm (elbow-to-shoulder segment) speed (m/s) at raw.fs."""
    return raw.speed("LeftUpperArm")


def right_upper_arm_speed(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Right upper arm speed (m/s) at raw.fs."""
    return raw.speed("RightUpperArm")


# ===========================================================================
# Signing-space position  (continuous)
# ===========================================================================
# Y axis = vertical height.  In Rokoko sensor space, positive Y is typically
# upward.  Signing-space height captures where in the grammatical signing
# space the hands are located (forehead = pronouns/questions, chest = default,
# low = locatives/lowered-affect).
#
# These are raw absolute positions.  The manager z-scores continuous regressors
# so the absolute offset is removed — only relative change over time matters.

def right_hand_height(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Right hand vertical position (Y, metres) at raw.fs."""
    return raw.position("RightHand")[:, 1]   # Y column


def left_hand_height(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    """Left hand vertical position (Y, metres) at raw.fs."""
    return raw.position("LeftHand")[:, 1]


def right_hand_lateral(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    return raw.position("RightHand")[:, 0]


def right_hand_depth(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
) -> np.ndarray:
    return raw.position("RightHand")[:, 2]


# ===========================================================================
# Direction reversal events  (event)
# ===========================================================================
# A direction reversal occurs when the hand decelerates to near-zero speed
# then re-accelerates — this marks the transition between movement strokes
# (the sub-sign "hold" in movement-hold phonological structure).
#
# Detection strategy:
#   1. Compute smooth hand speed at native rate.
#   2. Find local minima of speed that fall below a threshold (nearly-still).
#   3. Require minima to be separated by at least min_gap_ms.
#   4. Return the times of those minima as event impulse times.
#
# This is more principled than zero-crossing of velocity components because
# ASL strokes are rarely perfectly aligned to a single axis.

def _direction_reversal_times(
    speed: np.ndarray,
    fs: float,
    speed_threshold_ms: float = 0.15,
    min_gap_ms: float = 100.0,
    smoothing_samples: int = 3,
) -> np.ndarray:
    # Light smoothing to reduce noise spikes
    if smoothing_samples > 1:
        kernel = np.ones(smoothing_samples) / smoothing_samples
        speed_sm = np.convolve(speed, kernel, mode="same")
    else:
        speed_sm = speed.copy()

    min_gap_samples = max(1, int(np.round(min_gap_ms / 1000.0 * fs)))

    # find_peaks on *negated* speed finds local minima
    minima, props = find_peaks(
        -speed_sm,
        height=-speed_threshold_ms,   # speed < threshold → -speed > -threshold
        distance=min_gap_samples,
    )
    return minima / fs


def right_hand_reversal(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
    speed_threshold_ms: float = 0.15,
    min_gap_ms: float = 100.0,
) -> np.ndarray:
    speed = raw.speed("RightHand")
    return _direction_reversal_times(speed, raw.fs, speed_threshold_ms, min_gap_ms)


def left_hand_reversal(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
    speed_threshold_ms: float = 0.15,
    min_gap_ms: float = 100.0,
) -> np.ndarray:
    """Event times (s) when the left hand undergoes a direction reversal."""
    speed = raw.speed("LeftHand")
    return _direction_reversal_times(speed, raw.fs, speed_threshold_ms, min_gap_ms)


# ===========================================================================
# Acceleration peaks  (event)
# ===========================================================================
# Local maxima of hand acceleration magnitude mark the moment of peak force
# application within each movement stroke — the neural correlate of motor
# command execution.

def _accel_peak_times(
    accel: np.ndarray,
    fs: float,
    min_height_ms2: float = 5.0,
    min_gap_ms: float = 80.0,
) -> np.ndarray:
    """Find local maxima of acceleration ≥ min_height_ms2 (m/s²)."""
    min_gap_samples = max(1, int(np.round(min_gap_ms / 1000.0 * fs)))
    peaks, _ = find_peaks(accel, height=min_height_ms2, distance=min_gap_samples)
    return peaks / fs


def right_hand_accel_peak(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
    min_height_ms2: float = 5.0,
    min_gap_ms: float = 80.0,
) -> np.ndarray:
    accel = raw.accel_magnitude("RightHand")
    return _accel_peak_times(accel, raw.fs, min_height_ms2, min_gap_ms)


def left_hand_accel_peak(
    motion: MotionData,
    markers: MarkerData,
    raw: RawRokokoData,
    min_height_ms2: float = 5.0,
    min_gap_ms: float = 80.0,
) -> np.ndarray:
    """Event times (s) of left hand acceleration peaks ≥ 5 m/s²."""
    accel = raw.accel_magnitude("LeftHand")
    return _accel_peak_times(accel, raw.fs, min_height_ms2, min_gap_ms)
