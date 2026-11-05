# preprocessing/raw_rokoko_loader.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Segments to extract
# ---------------------------------------------------------------------------

UPPER_BODY_SEGMENTS = [
    "LeftShoulder",
    "RightShoulder",
    "LeftUpperArm",
    "RightUpperArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
]


def _segment_columns(segment: str) -> list[str]:
    """All expected column names for a single segment."""
    cols = []
    for axis in ("x", "y", "z"):
        cols += [
            f"{segment}_position_{axis}",
            f"{segment}_velocity_{axis}",
            f"{segment}_acceleration_{axis}",
        ]
    return cols


# ---------------------------------------------------------------------------
# RawRokokoData
# ---------------------------------------------------------------------------

@dataclass
class RawRokokoData:

    df: pd.DataFrame
    fs: float
    duration_s: float
    time_s: np.ndarray
    source_file: str
    participant: str = "unknown"
    session: str = "unknown"

    # ------------------------------------------------------------------
    # Convenience speed / acceleration magnitude helpers
    # ------------------------------------------------------------------

    def speed(self, segment: str) -> np.ndarray:
        v = self.df[[
            f"{segment}_velocity_x",
            f"{segment}_velocity_y",
            f"{segment}_velocity_z",
        ]].values
        return np.linalg.norm(v, axis=1)

    def accel_magnitude(self, segment: str) -> np.ndarray:
        """Scalar acceleration magnitude (m/s²) for `segment`, shape (n_samples,)."""
        a = self.df[[
            f"{segment}_acceleration_x",
            f"{segment}_acceleration_y",
            f"{segment}_acceleration_z",
        ]].values
        return np.linalg.norm(a, axis=1)

    def position(self, segment: str) -> np.ndarray:
        """3-D position (metres) for `segment`, shape (n_samples, 3) in x/y/z order."""
        return self.df[[
            f"{segment}_position_x",
            f"{segment}_position_y",
            f"{segment}_position_z",
        ]].values

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        segs = ", ".join(UPPER_BODY_SEGMENTS)
        lines = [
            "RawRokokoData",
            f"  Source      : {self.source_file}",
            f"  Participant : {self.participant}",
            f"  Session     : {self.session}",
            f"  Samples     : {len(self.df)}",
            f"  Duration    : {self.duration_s:.2f} s",
            f"  Sampling fs : {self.fs:.3f} Hz",
            f"  Segments    : {segs}",
            f"  Columns     : {len(self.df.columns)} extracted",
        ]
        return "\n".join(lines)

    @property
    def n_samples(self) -> int:
        return len(self.df)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_raw_rokoko(
    path: str | Path,
    participant: str = "unknown",
    session: str = "unknown",
) -> RawRokokoData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw Rokoko CSV not found: {path}")

    # --- Build list of columns to read --------------------------------
    # Always include Timestamp; add all segment columns present in file.
    # We probe headers first to avoid KeyError on optional columns.
    header_df = pd.read_csv(path, nrows=0)
    available = set(header_df.columns)

    keep = ["Timestamp"]
    for seg in UPPER_BODY_SEGMENTS:
        for col in _segment_columns(seg):
            if col in available:
                keep.append(col)
            else:
                import warnings
                warnings.warn(
                    f"Column '{col}' not found in {path.name} — skipping.",
                    UserWarning,
                    stacklevel=2,
                )

    df = pd.read_csv(path, usecols=keep)

    # --- Time axis ----------------------------------------------------
    # Timestamp is in milliseconds from start of recording.
    ts_ms = df["Timestamp"].values.astype(float)
    time_s = ts_ms / 1000.0

    # Sampling rate from median inter-sample interval
    dt_ms = np.median(np.diff(ts_ms))
    fs = 1000.0 / dt_ms

    df = df.drop(columns=["Timestamp"])
    df.insert(0, "time_s", time_s)
    df = df.reset_index(drop=True)

    duration_s = float(time_s[-1] - time_s[0])

    # Infer participant/session from filename if not provided
    stem = path.stem
    if participant == "unknown" and stem:
        participant = stem.split("_")[0] if "_" in stem else stem
    if session == "unknown" and "_" in stem:
        session = "_".join(stem.split("_")[1:])

    return RawRokokoData(
        df=df,
        fs=fs,
        duration_s=duration_s,
        time_s=time_s,
        source_file=str(path),
        participant=participant,
        session=session,
    )
