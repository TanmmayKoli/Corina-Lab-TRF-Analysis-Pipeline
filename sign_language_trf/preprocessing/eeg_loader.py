# preprocessing/eeg_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EEGData:

    raw: object             # mne.io.Raw (typed as object to avoid import at class level)
    sfreq: float
    sfreq_raw: float
    ch_names: list
    events: np.ndarray      # (n_events, 3) at sfreq
    event_id: dict          # label -> integer code (original trigger value)
    n_samples: int
    duration_s: float
    participant: Optional[str] = None
    session: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def times_for_code(self, code: int) -> np.ndarray:
        mask = self.events[:, 2] == code
        return self.events[mask, 0] / self.sfreq

    def samples_for_code(self, code: int) -> np.ndarray:
        """Return sample indices (at self.sfreq) for events matching `code`."""
        mask = self.events[:, 2] == code
        return self.events[mask, 0]

    def get_data(self) -> np.ndarray:
        data_v = self.raw.get_data()              # (n_channels, n_samples)
        return data_v.T * 1e6                      # → (n_samples, n_channels) µV


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_eeg(
    path: str | Path,
    session_config,                         # SessionConfig dataclass
    participant: Optional[str] = None,
    session_label: Optional[str] = None,
    preload: bool = True,
    verbose: bool = False,
    bandpass: bool = True,
    l_freq: float = 0.1,
    h_freq: float = 30.0,
) -> EEGData:
    import mne

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")

    # ── 1. Read raw ──────────────────────────────────────────────────────
    raw = mne.io.read_raw_eeglab(str(path), preload=preload, verbose=verbose)
    sfreq_raw = raw.info['sfreq']

    if verbose:
        print(f"[eeg_loader] Loaded: {path.name}")
        print(f"             Channels: {len(raw.ch_names)}, sfreq: {sfreq_raw} Hz, "
              f"duration: {raw.times[-1]:.1f}s")

    # ── 2. Drop non-scalp channels ───────────────────────────────────────
    channels_to_drop = [ch for ch in session_config.drop_channels
                        if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
        if verbose:
            print(f"             Dropped: {channels_to_drop}")

    # ── 3. Bandpass filter (before resampling) ───────────────────────────
    if bandpass:
        if not preload:
            raw.load_data(verbose=verbose)
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir',
                   fir_window='hamming', verbose=verbose)
        if verbose:
            print(f"             Bandpass filtered: {l_freq}–{h_freq} Hz")

    # ── 5. Extract events (before resampling, at original sfreq) ─────────
    # MNE reads EEGLAB annotations as string labels; event_id maps
    # label → MNE-internal remapped integer. We discard the remap and
    # build our own label→original_code dict so callers can use numeric codes.
    events_mne, event_id_mne = mne.events_from_annotations(raw, verbose=verbose)

    # Convert label→mne_index to label→original_numeric_code.
    # EEGLAB labels are the original trigger codes stored as strings ('255', etc.).
    # Normalise keys to plain Python str, values to plain Python int.
    try:
        event_id = {str(label): int(str(label)) for label in event_id_mne}
    except ValueError:
        # Non-numeric labels: keep the MNE-internal index as fallback.
        event_id = {str(k): v for k, v in event_id_mne.items()}
        if verbose:
            print("             Warning: non-numeric annotation labels; "
                  "event_id values are MNE-internal indices, not trigger codes.")

    # Remap events[:, 2] from MNE-internal indices to original numeric codes.
    remap = {v: event_id[str(k)] for k, v in event_id_mne.items()
             if str(k) in event_id and isinstance(event_id[str(k)], int)}
    events_orig = events_mne.copy()
    for mne_idx, numeric_code in remap.items():
        events_orig[events_mne[:, 2] == mne_idx, 2] = numeric_code

    # ── 6. Resample to common_rate ────────────────────────────────────────
    target_sfreq = float(session_config.common_rate)
    if sfreq_raw != target_sfreq:
        if not preload:
            raw.load_data(verbose=verbose)
        raw.resample(target_sfreq, npad='auto', verbose=verbose)

    # ── 7. Re-index events to decimated sample space ─────────────────────
    scale = target_sfreq / sfreq_raw
    events_decimated = events_orig.copy()
    events_decimated[:, 0] = np.round(events_orig[:, 0] * scale).astype(int)

    n_samples = len(raw.times)
    duration_s = raw.times[-1]

    participant_label = participant or getattr(session_config, 'participant', None)

    if verbose:
        print(f"             After resample: sfreq={target_sfreq} Hz, "
              f"n_samples={n_samples}")

    return EEGData(
        raw=raw,
        sfreq=target_sfreq,
        sfreq_raw=sfreq_raw,
        ch_names=list(raw.ch_names),
        events=events_decimated,
        event_id=event_id,
        n_samples=n_samples,
        duration_s=duration_s,
        participant=participant_label,
        session=session_label,
    )
