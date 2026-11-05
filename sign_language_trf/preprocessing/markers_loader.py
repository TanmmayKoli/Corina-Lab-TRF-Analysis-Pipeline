# preprocessing/markers_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical marker names (single source of truth)
# ---------------------------------------------------------------------------

class MarkerName:
    """String constants for all known marker types."""
    LEFT_HAND_MOVE_ONSET  = "LEFT_HAND_MOVE_ONSET"
    LEFT_HAND_STOP_ONSET  = "LEFT_HAND_STOP_ONSET"
    RIGHT_HAND_MOVE_ONSET = "RIGHT_HAND_MOVE_ONSET"
    RIGHT_HAND_STOP_ONSET = "RIGHT_HAND_STOP_ONSET"
    GLOBAL_MOTION_PEAK    = "GLOBAL_MOTION_PEAK"

    ALL = [
        LEFT_HAND_MOVE_ONSET,
        LEFT_HAND_STOP_ONSET,
        RIGHT_HAND_MOVE_ONSET,
        RIGHT_HAND_STOP_ONSET,
        GLOBAL_MOTION_PEAK,
    ]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MarkerData:
    """Container for motion-derived event markers."""

    df: pd.DataFrame

    # ------------------------------------------------------------------
    # Event time accessors
    # ------------------------------------------------------------------

    def times_ms(self, marker_name: str) -> np.ndarray:
        self._check_marker(marker_name)
        mask = self.df["marker_name"] == marker_name
        return self.df.loc[mask, "time_ms"].values.astype(float)

    def times_s(self, marker_name: str) -> np.ndarray:
        return self.times_ms(marker_name) / 1000.0

    def counts(self) -> dict[str, int]:
        """Return event counts per marker type."""
        return self.df["marker_name"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # Impulse train conversion
    # ------------------------------------------------------------------

    def to_impulse_train(
        self,
        marker_name: str,
        fs: float,
        duration_s: float,
        onset_offset_s: float = 0.0,
    ) -> np.ndarray:
        n_samples = int(np.round(duration_s * fs))
        impulse = np.zeros(n_samples, dtype=float)

        times = self.times_s(marker_name) + onset_offset_s
        indices = np.round(times * fs).astype(int)

        # Discard any events that fall outside the recording window
        valid = indices[(indices >= 0) & (indices < n_samples)]
        impulse[valid] = 1.0

        return impulse

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of event counts."""
        lines = ["MarkerData summary"]
        for name in MarkerName.ALL:
            n = len(self.times_ms(name)) if name in self.df["marker_name"].values else 0
            lines.append(f"  {name:<28}: {n} events")
        return "\n".join(lines)

    def _check_marker(self, marker_name: str) -> None:
        if marker_name not in MarkerName.ALL:
            raise ValueError(
                f"Unknown marker '{marker_name}'. "
                f"Valid options: {MarkerName.ALL}"
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_markers(path: str | Path) -> MarkerData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Markers file not found: {path}")

    df = pd.read_csv(path)

    required = ("time_ms", "marker_name", "port_code")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    # Warn about any unrecognized marker names (future-proofing)
    unknown = set(df["marker_name"].unique()) - set(MarkerName.ALL)
    if unknown:
        import warnings
        warnings.warn(
            f"Unrecognized marker names in {path.name}: {unknown}. "
            "These will be ignored by named accessors.",
            UserWarning,
            stacklevel=2,
        )

    return MarkerData(df=df)
