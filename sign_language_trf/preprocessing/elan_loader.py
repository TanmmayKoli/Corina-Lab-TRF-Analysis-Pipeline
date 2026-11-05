# preprocessing/elan_loader.py

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ELANData:

    annotations: pd.DataFrame
    tier_id: str
    source_file: str

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def n_annotations(self) -> int:
        return len(self.annotations)

    @property
    def signing_onset_s(self) -> float:
        """Onset of the first annotation."""
        return float(self.annotations["onset_s"].iloc[0])

    @property
    def signing_offset_s(self) -> float:
        """Offset of the last annotation."""
        return float(self.annotations["offset_s"].iloc[-1])

    # ------------------------------------------------------------------
    # Regressor helpers
    # ------------------------------------------------------------------

    def onset_times_s(self, max_t: Optional[float] = None) -> np.ndarray:
        onsets = self.annotations["onset_s"].values.copy()
        if max_t is not None:
            onsets = onsets[onsets < max_t]
        return onsets

    def offset_times_s(self, max_t: Optional[float] = None) -> np.ndarray:
        offsets = self.annotations["offset_s"].values.copy()
        if max_t is not None:
            offsets = offsets[offsets < max_t]
        return offsets

    def inter_sign_gap_signal(
        self,
        n_samples: int,
        fs: float,
        max_t: Optional[float] = None,
    ) -> np.ndarray:
        if max_t is None:
            max_t = n_samples / fs

        t = np.arange(n_samples) / fs  # time axis

        # Filter annotations within max_t
        df = self.annotations
        df = df[df["onset_s"] < max_t].copy()
        df["offset_s"] = df["offset_s"].clip(upper=max_t)

        # Start as "signing active" everywhere after first sign onset
        # (gap = 0 outside signing period, gap = 1 only within narrative)
        gap = np.zeros(n_samples, dtype=float)

        if df.empty:
            return gap

        # Mark the entire narrative window (first onset → last offset) as gap=1
        narrative_start = df["onset_s"].iloc[0]
        narrative_end = df["offset_s"].iloc[-1]
        narrative_mask = (t >= narrative_start) & (t <= narrative_end)
        gap[narrative_mask] = 1.0

        # Then zero out the samples that are WITHIN any annotation
        for _, row in df.iterrows():
            on_idx = int(np.round(row["onset_s"] * fs))
            off_idx = int(np.round(row["offset_s"] * fs))
            on_idx = max(0, min(on_idx, n_samples))
            off_idx = max(0, min(off_idx, n_samples))
            if on_idx < off_idx:
                gap[on_idx:off_idx] = 0.0

        return gap

    def sign_active_signal(
        self,
        n_samples: int,
        fs: float,
        max_t: Optional[float] = None,
    ) -> np.ndarray:
        return 1.0 - self.inter_sign_gap_signal(n_samples, fs, max_t)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        df = self.annotations
        mean_dur = df["duration_s"].mean()
        mean_gap = df["onset_s"].iloc[1:].values - df["offset_s"].iloc[:-1].values
        lines = [
            f"ELANData  tier='{self.tier_id}'  file='{Path(self.source_file).name}'",
            f"  Annotations : {self.n_annotations}",
            f"  Signing start : {self.signing_onset_s:.3f} s",
            f"  Signing end   : {self.signing_offset_s:.3f} s",
            f"  Narrative dur : {self.signing_offset_s - self.signing_onset_s:.1f} s",
            f"  Mean sign dur : {mean_dur * 1000:.0f} ms",
            f"  Mean gap dur  : {mean_gap.mean() * 1000:.0f} ms  "
            f"(min {mean_gap.min() * 1000:.0f}, max {mean_gap.max() * 1000:.0f} ms)",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ELANData(tier='{self.tier_id}', "
            f"n_annotations={self.n_annotations}, "
            f"signing={self.signing_onset_s:.1f}–{self.signing_offset_s:.1f} s)"
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def load_elan(
    path: str | Path,
    tier_id: str = "Gloss",
) -> ELANData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ELAN file not found: {path}")

    tree = ET.parse(str(path))
    root = tree.getroot()

    # ---------------------------------------------------------------
    # 1. Build time-slot lookup: ts_id → time in seconds
    # ---------------------------------------------------------------
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError("No <TIME_ORDER> element found in EAF file.")

    ts_to_ms: dict[str, float] = {}
    for ts in time_order.findall("TIME_SLOT"):
        ts_id = ts.attrib["TIME_SLOT_ID"]
        ts_val = float(ts.attrib["TIME_VALUE"])  # milliseconds
        ts_to_ms[ts_id] = ts_val

    ts_to_s: dict[str, float] = {k: v / 1000.0 for k, v in ts_to_ms.items()}

    # ---------------------------------------------------------------
    # 2. Find the requested tier
    # ---------------------------------------------------------------
    target_tier = None
    all_tiers = []
    for tier in root.findall("TIER"):
        tid = tier.attrib.get("TIER_ID", "")
        all_tiers.append(tid)
        if tid == tier_id:
            target_tier = tier

    if target_tier is None:
        raise ValueError(
            f"Tier '{tier_id}' not found in EAF file. "
            f"Available tiers: {all_tiers}"
        )

    # ---------------------------------------------------------------
    # 3. Extract ALIGNABLE_ANNOTATIONs
    # ---------------------------------------------------------------
    rows: list[dict] = []
    for ann in target_tier.findall("ANNOTATION"):
        align_ann = ann.find("ALIGNABLE_ANNOTATION")
        if align_ann is None:
            continue  # skip ref annotations (child tiers)

        ts1 = align_ann.attrib["TIME_SLOT_REF1"]
        ts2 = align_ann.attrib["TIME_SLOT_REF2"]
        ann_id = align_ann.attrib.get("ANNOTATION_ID", "")

        val_elem = align_ann.find("ANNOTATION_VALUE")
        gloss = val_elem.text.strip() if val_elem is not None and val_elem.text else ""

        onset_s = ts_to_s[ts1]
        offset_s = ts_to_s[ts2]

        rows.append({
            "annotation_id": ann_id,
            "onset_s": onset_s,
            "offset_s": offset_s,
            "duration_s": offset_s - onset_s,
            "gloss": gloss,
        })

    if not rows:
        raise ValueError(
            f"Tier '{tier_id}' found but contains no ALIGNABLE_ANNOTATIONs."
        )

    df = pd.DataFrame(rows).sort_values("onset_s").reset_index(drop=True)

    return ELANData(
        annotations=df,
        tier_id=tier_id,
        source_file=str(path),
    )
