# regressors/stasis.py

from __future__ import annotations

import numpy as np

from sign_language_trf.preprocessing.motion_loader import MotionData
from sign_language_trf.preprocessing.markers_loader import MarkerData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bilateral_velocity_mask(
    motion: MotionData,
    threshold_ms: float = 0.05,
) -> np.ndarray:
    df = motion.df
    lv = np.sqrt(df["LeftHand_vx"]**2 + df["LeftHand_vy"]**2 + df["LeftHand_vz"]**2)
    rv = np.sqrt(df["RightHand_vx"]**2 + df["RightHand_vy"]**2 + df["RightHand_vz"]**2)
    return (lv.values < threshold_ms) & (rv.values < threshold_ms)


def _left_velocity_mask(
    motion: MotionData,
    threshold_ms: float = 0.05,
) -> np.ndarray:
    """Boolean mask: True where LEFT hand speed is below threshold (m/s)."""
    df = motion.df
    lv = np.sqrt(df["LeftHand_vx"]**2 + df["LeftHand_vy"]**2 + df["LeftHand_vz"]**2)
    return lv.values < threshold_ms


def _right_velocity_mask(
    motion: MotionData,
    threshold_ms: float = 0.05,
) -> np.ndarray:
    """Boolean mask: True where RIGHT hand speed is below threshold (m/s)."""
    df = motion.df
    rv = np.sqrt(df["RightHand_vx"]**2 + df["RightHand_vy"]**2 + df["RightHand_vz"]**2)
    return rv.values < threshold_ms


def _apply_min_duration(mask: np.ndarray, min_samples: int) -> np.ndarray:
    """Remove contiguous True runs shorter than min_samples."""
    result = np.zeros_like(mask, dtype=float)
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_samples:
                result[i:j] = 1.0
            i = j
        else:
            i += 1
    return result


def _transitions(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = np.diff(signal.astype(int), prepend=0)
    onsets  = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    return onsets, offsets


# ---------------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------------

def velocity_stasis(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    mask = _bilateral_velocity_mask(motion, threshold_ms)
    min_samples = max(1, int(np.round(min_duration_ms / 1000.0 * motion.fs)))
    return _apply_min_duration(mask, min_samples)


def velocity_stasis_onset(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    sig = velocity_stasis(motion, markers, threshold_ms, min_duration_ms)
    onsets, _ = _transitions(sig)
    return onsets / motion.fs


def velocity_stasis_offset(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    sig = velocity_stasis(motion, markers, threshold_ms, min_duration_ms)
    _, offsets = _transitions(sig)
    return offsets / motion.fs


# 0.10 m/s variants — 21 onset/offset events each, scientifically still strict
# but providing enough events for TRF estimation.

def velocity_stasis_onset_10(motion: MotionData, markers: MarkerData) -> np.ndarray:
    """Stasis onset events at 0.10 m/s bilateral threshold (21 events in 174s)."""
    return velocity_stasis_onset(motion, markers, threshold_ms=0.10)


def velocity_stasis_offset_10(motion: MotionData, markers: MarkerData) -> np.ndarray:
    """Stasis offset events at 0.10 m/s bilateral threshold (21 events in 174s)."""
    return velocity_stasis_offset(motion, markers, threshold_ms=0.10)


def stasis(
    motion: MotionData,
    markers: MarkerData,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    raw = motion.stasis  # 1.0 when both hands stopped, from stopped_rule
    min_samples = max(1, int(np.round(min_duration_ms / 1000.0 * motion.fs)))

    result = np.zeros_like(raw, dtype=float)
    n = len(raw)
    i = 0
    while i < n:
        if raw[i] == 1.0:
            # Find the end of this contiguous stasis run
            j = i
            while j < n and raw[j] == 1.0:
                j += 1
            run_len = j - i
            if run_len >= min_samples:
                result[i:j] = 1.0
            i = j
        else:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Per-hand stasis regressors  (UNILATERAL — left or right independently)
# ---------------------------------------------------------------------------
# Motivation: bilateral stasis requires both hands stopped simultaneously,
# yielding only 6 events in 174s (mostly within-sign holds of two-handed signs).
# Per-hand stasis tracks each hand independently, giving far more events and
# capturing the hold phase of one-handed signs — the majority of ASL vocabulary.
# ---------------------------------------------------------------------------

def left_hand_stasis(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    mask = _left_velocity_mask(motion, threshold_ms)
    min_samples = max(1, int(np.round(min_duration_ms / 1000.0 * motion.fs)))
    return _apply_min_duration(mask, min_samples)


def right_hand_stasis(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    mask = _right_velocity_mask(motion, threshold_ms)
    min_samples = max(1, int(np.round(min_duration_ms / 1000.0 * motion.fs)))
    return _apply_min_duration(mask, min_samples)


def left_hand_stasis_onset(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    """Event times (s) when left-hand stasis BEGINS (0→1 transitions)."""
    sig = left_hand_stasis(motion, markers, threshold_ms, min_duration_ms)
    onsets, _ = _transitions(sig)
    return onsets / motion.fs


def left_hand_stasis_offset(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    """Event times (s) when left-hand stasis ENDS (1→0 transitions)."""
    sig = left_hand_stasis(motion, markers, threshold_ms, min_duration_ms)
    _, offsets = _transitions(sig)
    return offsets / motion.fs


def right_hand_stasis_onset(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    """Event times (s) when right-hand stasis BEGINS (0→1 transitions)."""
    sig = right_hand_stasis(motion, markers, threshold_ms, min_duration_ms)
    onsets, _ = _transitions(sig)
    return onsets / motion.fs


def right_hand_stasis_offset(
    motion: MotionData,
    markers: MarkerData,
    threshold_ms: float = 0.05,
    min_duration_ms: float = 200.0,
) -> np.ndarray:
    """Event times (s) when right-hand stasis ENDS (1→0 transitions)."""
    sig = right_hand_stasis(motion, markers, threshold_ms, min_duration_ms)
    _, offsets = _transitions(sig)
    return offsets / motion.fs
