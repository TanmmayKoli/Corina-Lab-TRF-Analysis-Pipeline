# sign_language_trf/models/permutation.py

from __future__ import annotations

import numpy as np


def run_permutation_test(
    X: np.ndarray,
    presentations: list[np.ndarray],
    exp_cfg,
    session_cfg,
    n_permutations: int = 100,
    min_shift_s: float = 30.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    fs = float(session_cfg.common_rate)
    n_time = X.shape[0]
    min_shift = int(min_shift_s * fs)
    max_shift = n_time - min_shift

    if max_shift <= min_shift:
        raise ValueError(
            f"Regressor too short ({n_time} samples = {n_time/fs:.1f}s) for "
            f"min_shift_s={min_shift_s}s. Reduce min_shift_s."
        )

    rng = np.random.default_rng(seed)

    # ── Observed r ────────────────────────────────────────────────────────
    if verbose:
        print("  Computing observed r (true alignment)...")
    observed_r = _run_cv(X, presentations, exp_cfg, session_cfg)

    if verbose:
        print(f"  Observed: mean r = {observed_r.mean():.4f}  "
              f"max r = {observed_r.max():.4f}")
        print(f"\n  Running {n_permutations} circular-shift permutations "
              f"(min shift: {min_shift_s:.0f}s)...")

    # ── Null distribution ─────────────────────────────────────────────────
    null_r = np.zeros((n_permutations, len(observed_r)))

    for i in range(n_permutations):
        shift = int(rng.integers(min_shift, max_shift + 1))
        X_shifted = np.roll(X, shift, axis=0)
        null_r[i] = _run_cv(X_shifted, presentations, exp_cfg, session_cfg)

        if verbose:
            pct_done = (i + 1) / n_permutations * 100
            print(f"  [{i+1:3d}/{n_permutations}]  shift = {shift/fs:5.0f}s  "
                  f"mean r = {null_r[i].mean():.4f}  ({pct_done:.0f}% done)")

    # ── p-values ──────────────────────────────────────────────────────────
    # One-tailed: does the true r exceed the null distribution?
    # +1 in numerator and denominator = standard conservative correction
    # (Phipson & Smyth 2010) — avoids p=0 which is uninterpretable.
    p_values = (np.sum(null_r >= observed_r[None, :], axis=0) + 1) / (n_permutations + 1)

    if verbose:
        n_sig_05 = (p_values < 0.05).sum()
        n_sig_01 = (p_values < 0.01).sum()
        print(f"\n  Done. Significant channels (p<0.05): {n_sig_05} / {len(p_values)}")
        print(f"         Significant channels (p<0.01): {n_sig_01} / {len(p_values)}")

    return {
        "observed_r":     observed_r,
        "null_r":         null_r,
        "p_values":       p_values,
        "significant_01": p_values < 0.01,
        "significant_05": p_values < 0.05,
        "n_permutations": n_permutations,
        "min_shift_s":    min_shift_s,
        "seed":           seed,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_cv(
    X: np.ndarray,
    presentations: list[np.ndarray],
    exp_cfg,
    session_cfg,
) -> np.ndarray:
    """Leave-one-playback-out CV on (X, presentations)."""
    from sign_language_trf.models.forward_trf import make_trf_model
    from sign_language_trf.models.evaluation import per_channel_r

    n_presentations = len(presentations)
    n_channels = presentations[0].shape[1]
    all_r = np.zeros((n_presentations, n_channels))

    for k in range(n_presentations):
        n_common = min(X.shape[0], presentations[k].shape[0])
        X_k = X[:n_common]
        y_k = presentations[k][:n_common]

        train_segs = [
            presentations[i][:min(X.shape[0], presentations[i].shape[0])]
            for i in range(n_presentations) if i != k
        ]
        X_train = np.vstack([X[:s.shape[0]] for s in train_segs])
        y_train = np.vstack(train_segs)

        model = make_trf_model(exp_cfg.trf, session_cfg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_k)
        all_r[k] = per_channel_r(y_k, y_pred)

    return all_r.mean(axis=0)
