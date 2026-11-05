# regressors/manager.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy.interpolate import interp1d

from sign_language_trf.preprocessing.motion_loader import MotionData
from sign_language_trf.preprocessing.markers_loader import MarkerData


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressorSpec:
    """Metadata for a single registered regressor."""
    name: str
    kind: Literal["continuous", "event"]
    fn: Callable  # fn(motion, markers) -> np.ndarray

    def __repr__(self) -> str:
        return f"RegressorSpec(name={self.name!r}, kind={self.kind!r})"


# ---------------------------------------------------------------------------
# Global registry  (populated at module import via the bottom of this file)
# ---------------------------------------------------------------------------

REGISTRY: dict[str, RegressorSpec] = {}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class RegressorManager:

    def __init__(
        self,
        motion: MotionData,
        markers: MarkerData,
        target_fs: float = 64.0,
    ) -> None:
        self.motion = motion
        self.markers = markers
        self.target_fs = target_fs

        # Canonical output dimensions — all outputs align to this
        self._n_samples = int(np.floor(motion.duration_s * target_fs))
        self._t_target = np.arange(self._n_samples) / target_fs
        self._t_source = motion.time_s  # native motion time axis

        # Cache so repeated calls don't recompute
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of samples in the canonical timeline at target_fs."""
        return self._n_samples

    @property
    def duration_s(self) -> float:
        """Duration of the canonical timeline in seconds."""
        return self._n_samples / self.target_fs

    @property
    def available(self) -> list[str]:
        """All registered regressor names."""
        return list(REGISTRY.keys())

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def get(self, name: str) -> np.ndarray:
        if name not in REGISTRY:
            raise ValueError(
                f"Unknown regressor '{name}'. Registered: {self.available}"
            )
        if name not in self._cache:
            self._cache[name] = self._compute(name)
        return self._cache[name]

    def get_matrix(self, names: list[str]) -> tuple[np.ndarray, list[str]]:
        if not names:
            raise ValueError("names must be a non-empty list.")

        columns = []
        for name in names:
            col = self.get(name)
            kind = REGISTRY[name].kind
            if kind == "continuous":
                std = col.std()
                if std == 0:
                    import warnings
                    warnings.warn(
                        f"Regressor '{name}' has zero variance — "
                        "check that the data covers a moving segment.",
                        UserWarning, stacklevel=2,
                    )
                    std = 1.0
                col = (col - col.mean()) / std
            # kind == "event": leave as 0/1
            columns.append(col)

        return np.column_stack(columns), list(names)

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self, name: str) -> np.ndarray:
        """Compute, resample, and cache one regressor."""
        spec = REGISTRY[name]
        raw = spec.fn(self.motion, self.markers)

        if spec.kind == "continuous":
            return self._resample_continuous(raw)
        elif spec.kind == "event":
            return self._times_to_impulse(raw)
        else:
            raise ValueError(f"Unknown kind: {spec.kind!r}")

    def _resample_continuous(self, raw: np.ndarray) -> np.ndarray:
        interp = interp1d(
            self._t_source,
            raw,
            kind="linear",
            bounds_error=False,
            fill_value=(raw[0], raw[-1]),
        )
        return interp(self._t_target)

    def _times_to_impulse(self, event_times_s: np.ndarray) -> np.ndarray:
        impulse = np.zeros(self._n_samples, dtype=float)
        indices = np.round(event_times_s * self.target_fs).astype(int)
        valid = indices[(indices >= 0) & (indices < self._n_samples)]
        impulse[valid] = 1.0
        return impulse

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "RegressorManager",
            f"  Target fs   : {self.target_fs} Hz",
            f"  n_samples   : {self.n_samples}",
            f"  Duration    : {self.duration_s:.1f} s",
            f"  Source fs   : {self.motion.fs:.2f} Hz",
            f"  Participant : {self.motion.participant}",
            f"  Session     : {self.motion.session}",
            f"  Registered  : {len(REGISTRY)} regressors",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry population
# Imports happen here (bottom of file) to avoid circular imports.
# Each module's functions are imported and registered with their kind.
# ---------------------------------------------------------------------------

def _populate_registry() -> None:
    from sign_language_trf.regressors import velocity, motion_energy, stasis, peaks

    entries = [
        # --- Continuous regressors ---
        RegressorSpec("left_hand_speed",    "continuous", velocity.left_hand_speed),
        RegressorSpec("right_hand_speed",   "continuous", velocity.right_hand_speed),
        RegressorSpec("motion_energy",      "continuous", motion_energy.motion_energy),
        RegressorSpec("stasis",             "continuous", stasis.stasis),
        RegressorSpec("left_stasis",        "continuous",
                      lambda m, mk: m.left_stasis),
        RegressorSpec("right_stasis",       "continuous",
                      lambda m, mk: m.right_stasis),
        # Velocity-threshold stasis (explicit ±0.05 m/s criterion, strict)
        RegressorSpec("velocity_stasis",           "continuous", stasis.velocity_stasis),
        RegressorSpec("velocity_stasis_onset",     "event",      stasis.velocity_stasis_onset),
        RegressorSpec("velocity_stasis_offset",    "event",      stasis.velocity_stasis_offset),
        # Relaxed 0.10 m/s variants — 21 onset/offset events each
        RegressorSpec("velocity_stasis_onset_10",  "event",      stasis.velocity_stasis_onset_10),
        RegressorSpec("velocity_stasis_offset_10", "event",      stasis.velocity_stasis_offset_10),

        # --- Per-hand (unilateral) stasis regressors ---
        # Each hand tracked independently — far more events than bilateral stasis.
        # Captures hold phases of one-handed signs (majority of ASL vocabulary).
        RegressorSpec("left_hand_stasis",          "continuous", stasis.left_hand_stasis),
        RegressorSpec("right_hand_stasis",         "continuous", stasis.right_hand_stasis),
        RegressorSpec("left_hand_stasis_onset",    "event",      stasis.left_hand_stasis_onset),
        RegressorSpec("left_hand_stasis_offset",   "event",      stasis.left_hand_stasis_offset),
        RegressorSpec("right_hand_stasis_onset",   "event",      stasis.right_hand_stasis_onset),
        RegressorSpec("right_hand_stasis_offset",  "event",      stasis.right_hand_stasis_offset),

        # --- Event impulse train regressors (raw 0/1, not z-scored) ---
        RegressorSpec("left_hand_move_onset",  "event", peaks.left_hand_move_onset),
        RegressorSpec("left_hand_stop_onset",  "event", peaks.left_hand_stop_onset),
        RegressorSpec("right_hand_move_onset", "event", peaks.right_hand_move_onset),
        RegressorSpec("right_hand_stop_onset", "event", peaks.right_hand_stop_onset),
        RegressorSpec("global_motion_peak",    "event", peaks.global_motion_peak),
    ]

    for spec in entries:
        REGISTRY[spec.name] = spec


_populate_registry()


# ---------------------------------------------------------------------------
# Raw-Rokoko regressors (registered dynamically at runtime)
# ---------------------------------------------------------------------------

def register_raw_regressors(raw_data) -> None:
    from sign_language_trf.regressors import acceleration as acc

    r = raw_data  # closed over in lambdas below

    raw_entries = [
        # --- Hand acceleration magnitude (continuous) ---
        RegressorSpec("left_hand_accel",   "continuous",
                      lambda m, mk, _r=r: acc.left_hand_accel(m, mk, _r)),
        RegressorSpec("right_hand_accel",  "continuous",
                      lambda m, mk, _r=r: acc.right_hand_accel(m, mk, _r)),
        RegressorSpec("combined_hand_accel", "continuous",
                      lambda m, mk, _r=r: acc.combined_hand_accel(m, mk, _r)),

        # --- Signing-space position (continuous) ---
        RegressorSpec("right_hand_height",   "continuous",
                      lambda m, mk, _r=r: acc.right_hand_height(m, mk, _r)),
        RegressorSpec("left_hand_height",    "continuous",
                      lambda m, mk, _r=r: acc.left_hand_height(m, mk, _r)),
        RegressorSpec("right_hand_lateral",  "continuous",
                      lambda m, mk, _r=r: acc.right_hand_lateral(m, mk, _r)),
        RegressorSpec("right_hand_depth",    "continuous",
                      lambda m, mk, _r=r: acc.right_hand_depth(m, mk, _r)),

        # --- Direction reversal events (event) ---
        RegressorSpec("right_hand_reversal", "event",
                      lambda m, mk, _r=r: acc.right_hand_reversal(m, mk, _r)),
        RegressorSpec("left_hand_reversal",  "event",
                      lambda m, mk, _r=r: acc.left_hand_reversal(m, mk, _r)),

        # --- Acceleration peak events (event) ---
        RegressorSpec("right_hand_accel_peak", "event",
                      lambda m, mk, _r=r: acc.right_hand_accel_peak(m, mk, _r)),
        RegressorSpec("left_hand_accel_peak",  "event",
                      lambda m, mk, _r=r: acc.left_hand_accel_peak(m, mk, _r)),
    ]

    for spec in raw_entries:
        REGISTRY[spec.name] = spec

    n = len(raw_entries)
    print(f"Registered {n} raw-Rokoko regressors "
          f"(participant={raw_data.participant}, session={raw_data.session})")


# ---------------------------------------------------------------------------
# ELAN-based linguistic regressors (registered dynamically at runtime)
# ---------------------------------------------------------------------------

def register_elan_regressors(elan_data) -> None:
    from sign_language_trf.regressors import linguistic

    elan_entries = [
        # Event regressors: impulse at each sign boundary
        RegressorSpec(
            "sign_onset",
            "event",
            lambda m, mk, e=elan_data: linguistic.sign_onset(m, mk, e),
        ),
        RegressorSpec(
            "sign_offset",
            "event",
            lambda m, mk, e=elan_data: linguistic.sign_offset(m, mk, e),
        ),
        # Continuous state regressors: signing vs gap
        RegressorSpec(
            "inter_sign_gap",
            "continuous",
            lambda m, mk, e=elan_data: linguistic.inter_sign_gap(m, mk, e),
        ),
        RegressorSpec(
            "sign_active",
            "continuous",
            lambda m, mk, e=elan_data: linguistic.sign_active(m, mk, e),
        ),
        # ELAN + Rokoko combined: phonologically-grounded kinematic events
        RegressorSpec(
            "sign_stroke_peak",
            "event",
            lambda m, mk, e=elan_data: linguistic.sign_stroke_peak(m, mk, e),
        ),
        RegressorSpec(
            "sign_transition_min",
            "event",
            lambda m, mk, e=elan_data: linguistic.sign_transition_min(m, mk, e),
        ),
        RegressorSpec(
            "sign_rate",
            "continuous",
            lambda m, mk, e=elan_data: linguistic.sign_rate(m, mk, e),
        ),
    ]

    for spec in elan_entries:
        REGISTRY[spec.name] = spec

    print(f"Registered 4 ELAN regressors from '{elan_data.tier_id}' tier "
          f"({elan_data.n_annotations} annotations, "
          f"{elan_data.signing_onset_s:.1f}–{elan_data.signing_offset_s:.1f} s)")


# ---------------------------------------------------------------------------
# Video-derived regressors (registered dynamically at runtime)
# ---------------------------------------------------------------------------

def register_video_regressors(
    video_path: str,
    motion_time_s,
    downsample_factor: int = 4,
) -> None:
    from sign_language_trf.regressors.video import (
        compute_ivc, compute_ivc_regions, resample_ivc_to_motion
    )

    print("Computing IVC from video...", flush=True)

    # --- Global IVC ---
    ivc_global, fps = compute_ivc(video_path, downsample_factor=downsample_factor)
    ivc_global_motion = resample_ivc_to_motion(ivc_global, fps, motion_time_s)

    REGISTRY["ivc"] = RegressorSpec(
        "ivc", "continuous",
        lambda m, mk, _a=ivc_global_motion: _a,
    )

    # --- Spatial region IVCs ---
    region_results = compute_ivc_regions(
        video_path, downsample_factor=downsample_factor
    )

    region_name_map = {
        "face":            "ivc_face",
        "dominant_hand":   "ivc_dominant_hand",
        "nondominant_hand": "ivc_nondominant_hand",
    }

    for region_key, reg_name in region_name_map.items():
        ivc_arr, _ = region_results[region_key]
        ivc_motion = resample_ivc_to_motion(ivc_arr, fps, motion_time_s)
        _arr = ivc_motion  # capture for closure
        REGISTRY[reg_name] = RegressorSpec(
            reg_name, "continuous",
            lambda m, mk, _a=_arr: _a,
        )

    print(f"Registered video regressors: ivc, ivc_face, "
          f"ivc_dominant_hand, ivc_nondominant_hand")
