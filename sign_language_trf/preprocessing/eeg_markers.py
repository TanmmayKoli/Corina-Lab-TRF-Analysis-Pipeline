# preprocessing/eeg_markers.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sign_language_trf.preprocessing.eeg_loader import EEGData
from sign_language_trf.preprocessing.markers_loader import MarkerName


# ---------------------------------------------------------------------------
# Port code → marker name mapping  (single source of truth)
# ---------------------------------------------------------------------------

PORT_CODE_TO_NAME: dict[int, str] = {
    11: MarkerName.LEFT_HAND_MOVE_ONSET,
    12: MarkerName.LEFT_HAND_STOP_ONSET,
    13: MarkerName.RIGHT_HAND_MOVE_ONSET,
    14: MarkerName.RIGHT_HAND_STOP_ONSET,
    15: MarkerName.GLOBAL_MOTION_PEAK,
}

NAME_TO_PORT_CODE: dict[str, int] = {v: k for k, v in PORT_CODE_TO_NAME.items()}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EEGMarkerData:

    event_times_s: dict[str, np.ndarray]
    per_presentation: dict[str, list[np.ndarray]]
    n_presentations: int
    sfreq: float

    # ------------------------------------------------------------------
    # Interface matching MarkerData
    # ------------------------------------------------------------------

    def times_s(self, marker_name: str) -> np.ndarray:
        if marker_name not in self.event_times_s:
            raise ValueError(
                f"Unknown marker '{marker_name}'. "
                f"Available: {list(self.event_times_s.keys())}"
            )
        return self.event_times_s[marker_name]

    def times_ms(self, marker_name: str) -> np.ndarray:
        """Return averaged event times in milliseconds (motion timeline)."""
        return self.times_s(marker_name) * 1000.0

    def counts(self) -> dict[str, int]:
        """Return number of events per marker type (in the averaged set)."""
        return {name: len(times) for name, times in self.event_times_s.items()}

    def counts_per_presentation(self) -> dict[str, list[int]]:
        """Return raw per-presentation event counts for diagnostics."""
        return {
            name: [len(arr) for arr in arrays]
            for name, arrays in self.per_presentation.items()
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"EEGMarkerData  ({self.n_presentations} presentations, "
            f"averaged, sfreq={self.sfreq} Hz)"
        ]
        pp_counts = self.counts_per_presentation()
        for name in MarkerName.ALL:
            if name in self.event_times_s:
                per_pres = pp_counts[name]
                avg = np.mean(per_pres)
                lines.append(
                    f"  {name:<28}: {len(self.event_times_s[name]):3d} events "
                    f"(per pres: {per_pres}, mean={avg:.1f})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def extract_eeg_markers(
    eeg: EEGData,
    onset_code: int = 255,
    end_code: int = 254,
    motion_codes: Optional[list[int]] = None,
    average_across_presentations: bool = True,
) -> EEGMarkerData:
    if motion_codes is None:
        motion_codes = list(PORT_CODE_TO_NAME.keys())  # [11, 12, 13, 14, 15]

    # Locate all [onset → end] brackets
    onset_samples = eeg.samples_for_code(onset_code)
    end_samples   = eeg.samples_for_code(end_code)

    if len(onset_samples) == 0 or len(end_samples) == 0:
        raise ValueError(
            f"No code {onset_code} or {end_code} events found in EEG. "
            "Check onset_code / end_code parameters."
        )
    if len(onset_samples) != len(end_samples):
        raise ValueError(
            f"Mismatched brackets: {len(onset_samples)} onsets, "
            f"{len(end_samples)} ends. EEG may have incomplete presentations."
        )

    n_pres = len(onset_samples)
    per_presentation: dict[str, list[np.ndarray]] = {
        PORT_CODE_TO_NAME[c]: [] for c in motion_codes
    }

    for i, (start_samp, end_samp) in enumerate(zip(onset_samples, end_samples)):
        # Events within this bracket
        mask = (eeg.events[:, 0] > start_samp) & (eeg.events[:, 0] <= end_samp)
        pres_events = eeg.events[mask]

        for code in motion_codes:
            name = PORT_CODE_TO_NAME[code]
            code_mask = pres_events[:, 2] == code
            code_samples = pres_events[code_mask, 0]

            # Convert to seconds on the motion timeline
            # (t=0 = video onset = code 255 sample)
            times_s = (code_samples - start_samp).astype(float) / eeg.sfreq
            per_presentation[name].append(np.sort(times_s))

    # Build averaged (or single-presentation) canonical event times
    event_times_s: dict[str, np.ndarray] = {}

    for name, pres_arrays in per_presentation.items():
        if not average_across_presentations:
            # Use presentation 1 only
            event_times_s[name] = pres_arrays[0]
            continue

        # Average approach:
        # All presentations should have the same events (same video played 5×).
        # Minor count mismatches happen when events fall right on a bracket edge.
        # Strategy: use the presentation with the median count as reference,
        # then average aligned events across presentations.
        counts = [len(a) for a in pres_arrays]
        median_count = int(np.median(counts))

        # Filter to presentations that match the median count
        matching = [a for a in pres_arrays if len(a) == median_count]

        if len(matching) == 0:
            # Fallback: use the most common count
            from collections import Counter
            most_common_count = Counter(counts).most_common(1)[0][0]
            matching = [a for a in pres_arrays if len(a) == most_common_count]

        if len(matching) == 1:
            event_times_s[name] = matching[0]
        else:
            # Stack and average element-wise
            stacked = np.array(matching)           # (n_matching, n_events)
            event_times_s[name] = np.mean(stacked, axis=0)

    return EEGMarkerData(
        event_times_s=event_times_s,
        per_presentation=per_presentation,
        n_presentations=n_pres,
        sfreq=eeg.sfreq,
    )
