# regressors/motion_energy.py

from __future__ import annotations

import numpy as np

from sign_language_trf.preprocessing.motion_loader import MotionData
from sign_language_trf.preprocessing.markers_loader import MarkerData

# Default: hands only (matches Tristan_Story_1_hands_only.csv)
_HANDS = ["LeftHand", "RightHand"]


def motion_energy(
    motion: MotionData,
    markers: MarkerData,
    joints: list[str] | None = None,
) -> np.ndarray:
    if joints is None:
        joints = _HANDS

    for j in joints:
        if j not in motion.joints:
            raise ValueError(
                f"Joint '{j}' not found in motion data. "
                f"Available joints: {motion.joints}"
            )

    return sum(motion.speed(j) ** 2 for j in joints)
