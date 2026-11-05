# preprocessing/motion_loader.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MotionData:

    df: pd.DataFrame
    fs: float
    duration_s: float
    joints: list[str]
    participant: str | None = None
    session: str | None = None

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @property
    def time_s(self) -> np.ndarray:
        """Continuous time axis in seconds."""
        return self.df["time_s"].values

    @property
    def time_ms(self) -> np.ndarray:
        """Continuous time axis in milliseconds."""
        return self.df["time_ms"].values

    # ------------------------------------------------------------------
    # Per-joint accessors
    # ------------------------------------------------------------------

    def velocity(self, joint: str) -> np.ndarray:
        """3-component velocity for a joint."""
        self._check_joint(joint)
        cols = [f"{joint}_vx", f"{joint}_vy", f"{joint}_vz"]
        return self.df[cols].values

    def speed(self, joint: str) -> np.ndarray:
        return np.linalg.norm(self.velocity(joint), axis=1)

    def stopped(self, joint: str) -> np.ndarray:
        self._check_joint(joint)
        return self.df[f"{joint}_stopped_rule"].values.astype(bool)

    # ------------------------------------------------------------------
    # Convenience properties for primary regressors
    # ------------------------------------------------------------------

    @property
    def left_hand_speed(self) -> np.ndarray:
        """Scalar speed of the left hand. Shape (N,)."""
        return self.speed("LeftHand")

    @property
    def right_hand_speed(self) -> np.ndarray:
        """Scalar speed of the right hand. Shape (N,)."""
        return self.speed("RightHand")

    @property
    def motion_energy(self) -> np.ndarray:
        return sum(self.speed(j) ** 2 for j in self.joints)

    @property
    def stasis(self) -> np.ndarray:
        both_stopped = self.stopped("LeftHand") & self.stopped("RightHand")
        return both_stopped.astype(float)

    @property
    def left_stasis(self) -> np.ndarray:
        """Stasis for left hand only. Shape (N,)."""
        return self.stopped("LeftHand").astype(float)

    @property
    def right_stasis(self) -> np.ndarray:
        """Stasis for right hand only. Shape (N,)."""
        return self.stopped("RightHand").astype(float)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the recording."""
        lines = [
            f"MotionData summary",
            f"  Participant : {self.participant or 'unknown'}",
            f"  Session     : {self.session or 'unknown'}",
            f"  Duration    : {self.duration_s:.1f} s",
            f"  Samples     : {len(self.df)}",
            f"  Fs          : {self.fs:.2f} Hz",
            f"  Joints      : {', '.join(self.joints)}",
            f"  Stasis %    : {self.stasis.mean() * 100:.1f}% of recording",
        ]
        return "\n".join(lines)

    def _check_joint(self, joint: str) -> None:
        if joint not in self.joints:
            raise ValueError(
                f"Joint '{joint}' not found. Available joints: {self.joints}"
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_motion(
    path: str | Path,
    participant: str | None = None,
    session: str | None = None,
) -> MotionData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")

    df = pd.read_csv(path)

    # --- Validate required columns ---
    for col in ("time_s", "time_ms"):
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in {path.name}. "
                f"Columns present: {list(df.columns)}"
            )

    # --- Infer joint names from columns ending in '_vx' ---
    joints = [col[:-3] for col in df.columns if col.endswith("_vx")]
    if not joints:
        raise ValueError(
            f"No velocity columns found in {path.name}. "
            "Expected columns like 'LeftHand_vx'."
        )

    # --- Validate that each joint has vx, vy, vz, stopped_rule ---
    for joint in joints:
        missing = [
            f"{joint}_{s}"
            for s in ("vx", "vy", "vz", "stopped_rule")
            if f"{joint}_{s}" not in df.columns
        ]
        if missing:
            raise ValueError(f"Incomplete data for joint '{joint}'. Missing: {missing}")

    # --- Infer sampling rate from median inter-sample interval ---
    dt = np.diff(df["time_s"].values)
    if len(dt) == 0:
        raise ValueError("Motion file contains fewer than 2 rows.")
    fs = float(1.0 / np.median(dt))

    duration_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0])

    # --- Infer participant / session from filename if not supplied ---
    if participant is None or session is None:
        parts = path.stem.split("_")
        # Convention: {Participant}_{Story}_{N}_processed
        # e.g. Tristan_Story_1_processed
        if participant is None and len(parts) >= 1:
            participant = parts[0]
        if session is None and len(parts) >= 3:
            session = "_".join(parts[1:3])  # e.g. 'Story_1'

    return MotionData(
        df=df,
        fs=round(fs, 4),
        duration_s=duration_s,
        joints=joints,
        participant=participant,
        session=session,
    )
