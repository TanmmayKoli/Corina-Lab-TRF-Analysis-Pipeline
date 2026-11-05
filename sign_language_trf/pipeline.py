# sign_language_trf/pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(
    experiment: str,
    motion_path: str | Path,
    markers_path: str | Path,
    session_config_path: str | Path = "config/session.yaml",
    eeg_path: Optional[str | Path] = None,
    target_fs: Optional[float] = None,
    snr: float = 5.0,
    sim_n_channels: int = 4,
    sim_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run a complete TRF experiment end-to-end."""
    from sign_language_trf.config import load_session_config, load_experiment_config
    from sign_language_trf.preprocessing.motion_loader import load_motion
    from sign_language_trf.preprocessing.markers_loader import load_markers
    from sign_language_trf.regressors.manager import RegressorManager
    from sign_language_trf.models.forward_trf import make_trf_model
    from sign_language_trf.models.evaluation import per_channel_r, summarise_r
    from sign_language_trf.models.cross_validation import (
        blocked_folds, presentation_folds
    )

    # ── 1. Load configs ───────────────────────────────────────────────────
    session_cfg = load_session_config(session_config_path)
    exp_cfg     = load_experiment_config(
        Path("config/experiments") / f"{experiment}.yaml"
    )
    if verbose:
        _banner(f"EXPERIMENT: {experiment}")
        print(f"  Regressors : {exp_cfg.regressors}")
        print(f"  TRF window : {exp_cfg.trf.tmin*1000:.0f} → "
              f"{exp_cfg.trf.tmax*1000:.0f} ms")
        print(f"  α (ridge)  : {exp_cfg.trf.alpha}")

    fs = target_fs or float(session_cfg.common_rate)

    # ── 2. Load motion + markers, build regressor matrix ─────────────────
    if verbose:
        print(f"\n[ 1 ]  Loading motion + markers...")
    motion  = load_motion(motion_path)
    markers = load_markers(markers_path)
    mgr = RegressorManager(motion, markers, target_fs=fs)
    X, feature_names = mgr.get_matrix(exp_cfg.regressors)
    if verbose:
        print(f"       Regressor matrix : {X.shape}  features: {feature_names}")

    # ── 3. Get EEG (real or synthetic) ───────────────────────────────────
    mode = "synthetic"
    ch_names = [f"ch{i}" for i in range(sim_n_channels)]

    if eeg_path is not None:
        mode = "real"
        if verbose:
            print(f"\n[ 2 ]  Loading real EEG from {Path(eeg_path).name}...")
        eeg_presentations, ch_names = _load_real_eeg(
            eeg_path, session_cfg, verbose
        )
        n_channels = eeg_presentations[0].shape[1]
        if verbose:
            print(f"       n_presentations: {len(eeg_presentations)}, "
                  f"n_channels: {n_channels}")
    else:
        if verbose:
            print(f"\n[ 2 ]  Generating synthetic EEG (SNR={snr}, "
                  f"n_ch={sim_n_channels})...")
        eeg_presentations, _planted = _generate_synthetic_eeg(
            X, feature_names, exp_cfg, fs, snr, sim_n_channels, sim_seed,
            session_cfg.n_presentations, verbose
        )
        n_channels = sim_n_channels

    n_presentations = len(eeg_presentations)
    if verbose:
        print(f"       mode={mode}, n_presentations={n_presentations}")

    # ── 4. Cross-validation ───────────────────────────────────────────────
    # Leave-one-playback-out: same stimulus (X) repeated N times.
    # Each EEG presentation may differ slightly in length from X due to
    # rounding; we trim both X and EEG to the minimum of the two.
    #
    # For each fold k:
    #   test:  (X_trimmed, EEG[k])
    #   train: (X_trimmed × (N-1), concat(EEG[i] for i≠k))
    #          — tiling X to match the concatenated training EEG.
    if verbose:
        print(f"\n[ 3 ]  Running leave-one-playback-out CV "
              f"({n_presentations} folds)...")

    fold_results = []
    all_r = np.zeros((n_presentations, n_channels))

    for k in range(n_presentations):
        # Common length: min of X and EEG presentation k
        n_common = min(X.shape[0], eeg_presentations[k].shape[0])
        X_k  = X[:n_common]
        y_k  = eeg_presentations[k][:n_common]

        # Training: tile X to match concatenated train EEG
        train_segs = [eeg_presentations[i][:min(X.shape[0], eeg_presentations[i].shape[0])]
                      for i in range(n_presentations) if i != k]
        X_train = np.vstack([X[:s.shape[0]] for s in train_segs])
        y_train = np.vstack(train_segs)

        model = make_trf_model(exp_cfg.trf, session_cfg)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_k)
        r_fold  = per_channel_r(y_k, y_pred)
        kernel  = model.get_kernel()   # (n_channels, n_features, n_lags)

        all_r[k] = r_fold
        fold_results.append({
            "fold":   k,
            "r":      r_fold,
            "kernel": kernel,
            **({"y_pred": y_pred} if mode == "synthetic" else {}),
        })

        if verbose:
            print(f"       Fold {k}:  "
                  f"mean r = {r_fold.mean():.4f}  "
                  f"max r = {r_fold.max():.4f}")

    # ── 5. Aggregate ─────────────────────────────────────────────────────
    r_per_channel = all_r.mean(axis=0)
    kernels_mean  = np.mean([f["kernel"] for f in fold_results], axis=0)
    delays_s      = model.delays_s

    r_summary = summarise_r(r_per_channel, ch_names=ch_names)
    if verbose:
        print(f"\n[ 4 ]  Summary:")
        print(f"       mean r across channels : {r_summary['mean']:.4f}")
        print(f"       best channel           : {r_summary.get('best_ch_name', r_summary['best_ch'])} "
              f"(r = {r_summary['max']:.4f})")
        print(f"       worst channel          : {r_summary.get('worst_ch_name', r_summary['worst_ch'])} "
              f"(r = {r_summary['min']:.4f})")

    return {
        "experiment":        experiment,
        "feature_names":     feature_names,
        "delays_s":          delays_s,
        "fold_results":      fold_results,
        "r_per_channel":     r_per_channel,
        "r_summary":         r_summary,
        "kernels_mean":      kernels_mean,
        "ch_names":          ch_names,
        "session_config":    session_cfg,
        "experiment_config": exp_cfg,
        "mode":              mode,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_eeg(
    X, feature_names, exp_cfg, fs, snr, n_channels, seed, n_presentations, verbose
):
    from sign_language_trf.models.simulator import SimulatedEEG
    from sign_language_trf.models.forward_trf import RidgeTRF

    tmin = exp_cfg.trf.tmin
    tmax = exp_cfg.trf.tmax

    # Probe for n_lags
    _probe = RidgeTRF(tmin=tmin, tmax=tmax, sfreq=fs, alpha=exp_cfg.trf.alpha)
    _probe.fit(X[:100], np.zeros((100, n_channels)))
    n_lags = _probe.n_lags

    # Generate N independent noisy copies (same clean signal, different noise seeds)
    presentations = []
    planted_kernels = None
    for i in range(n_presentations):
        sim = SimulatedEEG(n_channels=n_channels, snr=snr, seed=seed + i,
                           tmin=tmin, tmax=tmax)
        noisy, _, kernels = sim.generate(X, feature_names, fs, n_lags)
        presentations.append(noisy)
        if planted_kernels is None:
            planted_kernels = kernels

    return presentations, planted_kernels


def _load_real_eeg(eeg_path, session_cfg, verbose):
    from sign_language_trf.preprocessing.eeg_loader import load_eeg
    from sign_language_trf.preprocessing.synchronization import (
        extract_presentation_segments, extract_segment_eeg
    )

    eeg = load_eeg(eeg_path, session_cfg, verbose=verbose)
    segs = extract_presentation_segments(
        eeg,
        onset_code=session_cfg.onset_code,
        offset_code=session_cfg.end_code,
        verbose=verbose,
    )

    presentations = [extract_segment_eeg(eeg, seg) for seg in segs]
    return presentations, eeg.ch_names


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
