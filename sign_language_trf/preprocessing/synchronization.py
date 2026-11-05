# preprocessing/synchronization.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PresentationSegment:

    idx: int
    onset_s: float
    offset_s: float
    onset_sample: int
    offset_sample: int
    n_samples: int
    duration_s: float

    def eeg_slice(self) -> slice:
        """Convenience: Python slice for extracting this segment from an array."""
        return slice(self.onset_sample, self.offset_sample)


# ---------------------------------------------------------------------------
# Job A: Presentation segmentation
# ---------------------------------------------------------------------------

def extract_presentation_segments(
    eeg_data,                       # EEGData (duck-typed to avoid circular import)
    onset_code: int = 255,
    offset_code: int = 254,
    min_duration_s: float = 100.0,
    verbose: bool = True,
) -> list[PresentationSegment]:
    events   = eeg_data.events
    sfreq    = eeg_data.sfreq
    n_samples = eeg_data.n_samples

    onset_samples  = events[events[:, 2] == onset_code,  0]
    offset_samples = events[events[:, 2] == offset_code, 0]

    if len(onset_samples) == 0:
        raise ValueError(
            f"No events with code {onset_code} found. "
            f"Available codes: {sorted(set(events[:, 2]))}"
        )
    if len(offset_samples) == 0:
        raise ValueError(
            f"No events with code {offset_code} found. "
            f"Available codes: {sorted(set(events[:, 2]))}"
        )

    onset_samples  = np.sort(onset_samples)
    offset_samples = np.sort(offset_samples)

    # Match each onset to the first offset that follows it
    segments: list[PresentationSegment] = []
    for i, on in enumerate(onset_samples):
        following_offsets = offset_samples[offset_samples > on]
        if len(following_offsets) == 0:
            if verbose:
                print(f"  [sync] Warning: onset {i} at {on/sfreq:.2f}s has no "
                      f"following offset — skipped.")
            continue
        off = following_offsets[0]

        onset_s  = on  / sfreq
        offset_s = off / sfreq
        dur_s    = offset_s - onset_s

        if dur_s < min_duration_s:
            if verbose:
                print(f"  [sync] Warning: segment {i} duration {dur_s:.1f}s < "
                      f"min_duration_s={min_duration_s}s — skipped.")
            continue

        # Clamp to valid sample range
        on_clamped  = max(0, int(on))
        off_clamped = min(n_samples, int(off))

        segments.append(PresentationSegment(
            idx           = i,
            onset_s       = onset_s,
            offset_s      = offset_s,
            onset_sample  = on_clamped,
            offset_sample = off_clamped,
            n_samples     = off_clamped - on_clamped,
            duration_s    = dur_s,
        ))

    if len(segments) == 0:
        raise ValueError(
            "No valid presentation segments found after filtering. "
            "Check trigger codes and min_duration_s."
        )

    if verbose:
        _print_segment_table(segments)

    return segments


def fold_boundaries(
    segments: list[PresentationSegment],
) -> list[tuple[int, int]]:
    return [(seg.onset_sample, seg.offset_sample) for seg in segments]


def extract_segment_eeg(
    eeg_data,
    segment: PresentationSegment,
) -> np.ndarray:
    data = eeg_data.get_data()      # (n_total_samples, n_channels) µV
    return data[segment.eeg_slice()]


# ---------------------------------------------------------------------------
# Job B: Modality offset estimation (stub — see docstring)
# ---------------------------------------------------------------------------

def estimate_sync_offset(
    eeg_data,
    motion_data,
    pulse_code: int = 50,
    pulse_interval_ms: float = 1000.0,
) -> float:
    raise NotImplementedError(
        "estimate_sync_offset is not yet implemented. "
        "Current analyses use motion_onset_offset_ms=0 from session config. "
        "Implement this when sub-frame timing precision is required."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_segment_table(segments: list[PresentationSegment]) -> None:
    print("\n  Presentation segments:")
    print(f"  {'#':>2}  {'onset_s':>9}  {'offset_s':>9}  "
          f"{'dur_s':>7}  {'n_samples':>10}")
    print("  " + "-" * 48)
    for seg in segments:
        print(f"  {seg.idx:>2}  {seg.onset_s:>9.2f}  {seg.offset_s:>9.2f}  "
              f"{seg.duration_s:>7.2f}  {seg.n_samples:>10}")
    print()
