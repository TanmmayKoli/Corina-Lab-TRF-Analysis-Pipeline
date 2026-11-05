# regressors/video.py

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Region definitions  (fraction of frame: [top, bottom, left, right])
# ---------------------------------------------------------------------------

IVC_REGIONS = {
    # Signer's face — upper-center portion of frame
    "face": (0.00, 0.35, 0.25, 0.75),
    # Video is MIRRORED (selfie/camera-facing view):
    # Tristan's right (dominant) hand appears on RIGHT side of frame.
    # Confirmed by Rokoko: right hand mean speed 0.36 vs 0.21 m/s (75% dominant),
    # and right frame half has higher IVC mean (372k vs 232k).
    # Anatomical note: right side of frame = right visual field → LEFT hemisphere → O1.
    "dominant_hand":     (0.35, 1.00, 0.50, 1.00),   # lower-RIGHT (Tristan's right hand)
    "nondominant_hand":  (0.35, 1.00, 0.00, 0.50),   # lower-LEFT  (Tristan's left hand)
}


# ---------------------------------------------------------------------------
# Core IVC computation
# ---------------------------------------------------------------------------

def compute_ivc(
    video_path: str | Path,
    downsample_factor: int = 4,
    grayscale: bool = True,
) -> tuple[np.ndarray, float]:
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for video IVC. Install with:\n"
            "  pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ivc_values = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Spatially downsample
        if downsample_factor > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (w // downsample_factor, h // downsample_factor),
                interpolation=cv2.INTER_AREA,
            )

        frame = frame.astype(np.float32)

        if prev_frame is not None:
            diff = frame - prev_frame
            ivc_values.append(np.sum(diff ** 2))

        prev_frame = frame

    cap.release()

    ivc = np.array(ivc_values, dtype=np.float64)
    print(f"  IVC computed: {len(ivc)} frames at {fps:.3f} fps "
          f"= {len(ivc)/fps:.1f} s")

    return ivc, fps


def compute_ivc_regions(
    video_path: str | Path,
    regions: dict | None = None,
    downsample_factor: int = 4,
) -> dict[str, tuple[np.ndarray, float]]:
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required. pip install opencv-python")

    if regions is None:
        regions = IVC_REGIONS

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Accumulate per-region frame lists
    region_buffers: dict[str, list[float]] = {r: [] for r in regions}
    prev_frames: dict[str, np.ndarray | None] = {r: None for r in regions}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Grayscale + downsample full frame once
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if downsample_factor > 1:
            h, w = gray.shape
            gray = cv2.resize(
                gray,
                (w // downsample_factor, h // downsample_factor),
                interpolation=cv2.INTER_AREA,
            )

        h, w = gray.shape
        gray = gray.astype(np.float32)

        for rname, (top, bottom, left, right) in regions.items():
            r0, r1 = int(top * h), int(bottom * h)
            c0, c1 = int(left * w), int(right * w)
            crop = gray[r0:r1, c0:c1]

            prev = prev_frames[rname]
            if prev is not None:
                diff = crop - prev
                region_buffers[rname].append(float(np.sum(diff ** 2)))
            prev_frames[rname] = crop.copy()

    cap.release()

    result = {}
    for rname, vals in region_buffers.items():
        result[rname] = (np.array(vals, dtype=np.float64), fps)
        print(f"  IVC [{rname}]: {len(vals)} frames = {len(vals)/fps:.1f} s")

    return result


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def resample_ivc_to_motion(
    ivc: np.ndarray,
    video_fps: float,
    motion_time_s: np.ndarray,
) -> np.ndarray:
    n_frames = len(ivc)
    video_time_s = np.arange(n_frames) / video_fps

    interp = interp1d(
        video_time_s,
        ivc,
        kind="linear",
        bounds_error=False,
        fill_value=(ivc[0], ivc[-1]),
    )
    return interp(motion_time_s)
