# sign_language_trf/config.py

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class TRFConfig:
    """TRF model hyperparameters."""
    tmin: float          # minimum lag in seconds (can be negative for pre-stimulus)
    tmax: float          # maximum lag in seconds
    backend: str         # "ridge" (default) | future: "boosting", "sparse"
    alpha: float         # ridge regularisation λ


@dataclass
class EvaluationConfig:
    """Evaluation and cross-validation strategy."""
    scheme: str          # "blocked" (only supported scheme; random splits are invalid for EEG)
    folds: str           # "leave_one_presentation_out" | "k_fold:<k>"


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Per-recording-session hardware and acquisition facts."""
    participant: str
    srate: int                      # EEG native sampling rate (Hz)
    common_rate: float              # shared analysis rate after decimation/resampling
    onset_code: int                 # EEG trigger code for video onset
    end_code: int                   # EEG trigger code for video end
    n_presentations: int            # number of times the stimulus was played
    presentation_duration_s: float  # measured from actual EEG brackets
    pulse_code: int                 # sync pulse trigger code (pending confirmation)
    pulse_interval_ms: float        # intended inter-pulse interval (ms)
    drop_channels: list[str]        # non-scalp channels to remove before analysis
    eeg_format: str                 # "eeglab" | future: "bdf", "fif"
    motion_onset_offset_ms: float   # estimated offset between motion time-0 and video onset
    eeg_path: Optional[str] = None          # path to EEG file (relative to project root)
    motion_path: Optional[str] = None       # path to motion CSV
    markers_path: Optional[str] = None      # path to markers CSV


@dataclass
class ExperimentConfig:
    """Per-experiment scientific question declaration."""
    name: str
    regressors: list[str]           # ordered list of regressor names from manager registry
    trf: TRFConfig
    evaluation: EvaluationConfig


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_session_config(path: str | Path) -> SessionConfig:
    """Load and validate a session YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Session config not found: {path}")
    with open(path) as f:
        d = yaml.safe_load(f)

    return SessionConfig(
        participant=str(d["participant"]),
        srate=int(d["srate"]),
        common_rate=float(d["common_rate"]),
        onset_code=int(d["onset_code"]),
        end_code=int(d["end_code"]),
        n_presentations=int(d["n_presentations"]),
        presentation_duration_s=float(d["presentation_duration_s"]),
        pulse_code=int(d["pulse_code"]),
        pulse_interval_ms=float(d["pulse_interval_ms"]),
        drop_channels=list(d.get("drop_channels", [])),
        eeg_format=str(d.get("eeg_format", "eeglab")),
        motion_onset_offset_ms=float(d.get("motion_onset_offset_ms", 0.0)),
        eeg_path=d.get("eeg_path"),
        motion_path=d.get("motion_path"),
        markers_path=d.get("markers_path"),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with open(path) as f:
        d = yaml.safe_load(f)

    trf = TRFConfig(
        tmin=float(d["trf"]["tmin"]),
        tmax=float(d["trf"]["tmax"]),
        backend=str(d["trf"]["backend"]),
        alpha=float(d["trf"]["alpha"]),
    )
    evaluation = EvaluationConfig(
        scheme=str(d["evaluation"]["scheme"]),
        folds=str(d["evaluation"]["folds"]),
    )
    return ExperimentConfig(
        name=str(d["name"]),
        regressors=list(d["regressors"]),
        trf=trf,
        evaluation=evaluation,
    )


# ---------------------------------------------------------------------------
# Reproducibility dump
# ---------------------------------------------------------------------------

def dump_configs(
    session: SessionConfig,
    experiment: ExperimentConfig,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_path = output_dir / f"session_{session.participant}.json"
    experiment_path = output_dir / f"experiment_{experiment.name}.json"

    with open(session_path, "w") as f:
        json.dump(asdict(session), f, indent=2)
    with open(experiment_path, "w") as f:
        json.dump(asdict(experiment), f, indent=2)
