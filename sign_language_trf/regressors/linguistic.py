# regressors/linguistic.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sign_language_trf.preprocessing.motion_loader import MotionData
from sign_language_trf.preprocessing.markers_loader import MarkerData

if TYPE_CHECKING:
    from sign_language_trf.preprocessing.elan_loader import ELANData


# ---------------------------------------------------------------------------
# Sign onset / offset event regressors
# ---------------------------------------------------------------------------

def sign_onset(
    motion: MotionData,
    markers: MarkerData,
    elan: "ELANData",
) -> np.ndarray:
    return elan.onset_times_s(max_t=motion.duration_s)


def sign_offset(
    motion: MotionData,
    markers: MarkerData,
    elan: "ELANData",
) -> np.ndarray:
    return elan.offset_times_s(max_t=motion.duration_s)


# ---------------------------------------------------------------------------
# Continuous state regressors
# ---------------------------------------------------------------------------

def inter_sign_gap(
    motion: MotionData,
    markers: MarkerData,
    elan: "ELANData",
) -> np.ndarray:
    n_samples = len(motion.df)
    fs = motion.fs
    return elan.inter_sign_gap_signal(
        n_samples=n_samples,
        fs=fs,
        max_t=motion.duration_s,
    )


def sign_active(
    motion: MotionData,
    markers: MarkerData,
    elan: "ELANData",
) -> np.ndarray:
    n_samples = len(motion.df)
    fs = motion.fs
    return elan.sign_active_signal(
        n_samples=n_samples,
        fs=fs,
        max_t=motion.duration_s,
    )


# ---------------------------------------------------------------------------
# ELAN + Rokoko combined regressors
# ---------------------------------------------------------------------------

def sign_stroke_peak(
    motion: MotionData,
    markers: MarkerData,
    elan: "ELANData",
) -> np.ndarray:
    t     = motion.time_s
    speed = motion.left_hand_speed + motion.right_hand_speed
    df    = elan.annotations
    df    = df[df["onset_s"] < motion.duration_s].copy()

    event_times = []
    for _, row in df.iterrows():
        mask = (t >= row["onset_s"]) & (t <= min(row["offset_s"], motion.duration_s))
        if mask.sum() < 2:
            continue
        peak_idx = np.argmax(speed[mask])
        event_times.append(t[mask][peak_idx])

    return np.array(event_times)


def sign_transition_min(
    motion: MotionData,
    markers: MarkerData,
    elan: "ELANData",
) -> np.ndarray:
    t     = motion.time_s
    speed = motion.left_hand_speed + motion.right_hand_speed
    df    = elan.annotations
    df    = df[df["onset_s"] < motion.duration_s].copy().reset_index(drop=True)

    event_times = []
    for i in range(len(df) - 1):
        gap_start = df["offset_s"].iloc[i]
        gap_end   = min(df["onset_s"].iloc[i + 1], motion.duration_s)
        if gap_end - gap_start < 0.016:
            continue
        mask = (t >= gap_start) & (t <= gap_end)
        if mask.sum() < 2:
            continue
        min_idx = np.argmin(speed[mask])
        event_times.append(t[mask][min_idx])

    return np.array(event_times)


def sign_rate(
    motion: MotionData,
    markers: MarkerData,
    elan: "ELANData",
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    n_samples = len(motion.df)
    fs        = motion.fs

    impulse = np.zeros(n_samples, dtype=float)
    onsets  = elan.onset_times_s(max_t=motion.duration_s)
    indices = np.round(onsets * fs).astype(int)
    valid   = indices[(indices >= 0) & (indices < n_samples)]
    impulse[valid] = 1.0

    sigma_samples = 0.5 * fs
    rate = gaussian_filter1d(impulse, sigma=sigma_samples) * fs

    return rate
