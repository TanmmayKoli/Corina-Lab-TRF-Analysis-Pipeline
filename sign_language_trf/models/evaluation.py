# models/evaluation.py

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def per_channel_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    n_channels = y_true.shape[1]
    rs = np.zeros(n_channels)
    for ch in range(n_channels):
        r, _ = pearsonr(y_true[:, ch], y_pred[:, ch])
        rs[ch] = float(r)
    return rs


def summarise_r(
    r_values: np.ndarray,
    ch_names: list[str] | None = None,
) -> dict:
    best_idx  = int(np.argmax(r_values))
    worst_idx = int(np.argmin(r_values))
    summary = {
        "mean":       float(np.mean(r_values)),
        "median":     float(np.median(r_values)),
        "std":        float(np.std(r_values)),
        "min":        float(np.min(r_values)),
        "max":        float(np.max(r_values)),
        "best_ch":    best_idx,
        "worst_ch":   worst_idx,
    }
    if ch_names is not None:
        summary["best_ch_name"]  = ch_names[best_idx]
        summary["worst_ch_name"] = ch_names[worst_idx]
    return summary
