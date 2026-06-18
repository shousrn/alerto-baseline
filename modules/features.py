# modules/features.py — Phase 2: Feature Extraction (three parallel branches)
# =============================================================================
# ALGORITHMS IMPLEMENTED:
#   Branch A — EAR    : Soukupová & Čech (2016) — Eye Aspect Ratio
#   Branch A — PERCLOS: Wierwille & Ellsworth (1994) via NHTSA standard
#   Branch B — MAR    : Rahman et al. (2015) — Mouth Aspect Ratio
#   Branch C — PnP    : Lepetit et al. (2009) — 3D Head Pose Estimation
#
# INPUT:  MediaPipe FaceLandmarker landmark list (Tasks API, 478 points)
# OUTPUT: 6-dimensional feature vector [ear, perclos, mar, pitch, yaw, roll]
# =============================================================================

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from collections import deque
from typing import Optional

from config import (
    RIGHT_EYE_IDX, LEFT_EYE_IDX, MOUTH_IDX,
    HEAD_POSE_ANCHOR_LM, FACE_3D_MODEL,
    EAR_CLOSED_THRESHOLD, PERCLOS_WINDOW_FRAMES,
    FEATURE_NAMES,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _px(lm: list, idx: int, h: int, w: int) -> np.ndarray:
    """Convert a single normalised MediaPipe landmark → pixel [x, y] array."""
    pt = lm[idx]
    return np.array([pt.x * w, pt.y * h], dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D numpy points."""
    return float(euclidean(a, b))


# ── Branch A: EAR (Eye Aspect Ratio) ─────────────────────────────────────────

# EAR formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 · ||p1-p4||)
# Source: Soukupová, T. & Čech, J. (2016). Real-time eye blink detection
#         using facial landmarks. 21st Computer Vision Winter Workshop (CVWW).
def compute_ear(lm: list, eye_idx: dict, h: int, w: int) -> float:
    """
    Eye Aspect Ratio — Soukupová & Čech (2016).

         ||p2 – p6|| + ||p3 – p5||
    EAR = ──────────────────────────
                 2 × ||p1 – p4||

    Parameters
    ----------
    lm      : MediaPipe landmark list (478 points)
    eye_idx : dict with keys p1…p6 → landmark indices
    h, w    : frame height and width in pixels

    Returns
    -------
    float — EAR value (~0.28–0.42 open; < 0.15 closed); 0.0 on degenerate input
    """
    p1 = _px(lm, eye_idx["p1"], h, w)
    p2 = _px(lm, eye_idx["p2"], h, w)
    p3 = _px(lm, eye_idx["p3"], h, w)
    p4 = _px(lm, eye_idx["p4"], h, w)
    p5 = _px(lm, eye_idx["p5"], h, w)
    p6 = _px(lm, eye_idx["p6"], h, w)

    vert  = _dist(p2, p6) + _dist(p3, p5)
    horiz = _dist(p1, p4)
    if horiz < 1e-6:
        return 0.0
    return vert / (2.0 * horiz)


def compute_avg_ear(lm: list, h: int, w: int) -> float:
    """Average EAR across both eyes — reduces unilateral blink noise."""
    r = compute_ear(lm, RIGHT_EYE_IDX, h, w)
    l = compute_ear(lm, LEFT_EYE_IDX,  h, w)
    return (r + l) / 2.0


# ── Branch A: PERCLOS (Percentage of Eyelid Closure) ─────────────────────────

# PERCLOS: percentage of frames in a sliding window where EAR < threshold.
# Source: Wierwille, W. W. & Ellsworth, L. A. (1994). Evaluation of driver
#         drowsiness by trained raters. Accident Analysis & Prevention, 26(5).
class PERCLOSBuffer:
    """
    PERCLOS sliding-window buffer — Wierwille & Ellsworth (1994) / NHTSA.

    PERCLOS = (frames where EAR < threshold) / window_size × 100 %

    Uses a fixed-length deque (circular FIFO) for O(1) insert and compute.

    Parameters
    ----------
    window_size : int   — frames in the analysis window (default 60 ≈ 2 s at 30 FPS)
    threshold   : float — EAR value below which an eye counts as "closed"
    """

    def __init__(self, window_size: int = PERCLOS_WINDOW_FRAMES,
                 threshold: float = EAR_CLOSED_THRESHOLD) -> None:
        self.window    = window_size
        self.threshold = threshold
        self._buf: deque = deque(maxlen=window_size)

    def update(self, ear: float) -> float:
        """Append 1 if ear < threshold else 0; return current PERCLOS percent (0–100)."""
        self._buf.append(1 if ear < self.threshold else 0)
        if not self._buf:
            return 0.0
        return (sum(self._buf) / len(self._buf)) * 100.0

    def reset(self) -> None:
        """Clear the sliding window."""
        self._buf.clear()

    @property
    def ready(self) -> bool:
        """True once the buffer holds a full window of samples."""
        return len(self._buf) == self.window


# ── Branch B: MAR (Mouth Aspect Ratio) ───────────────────────────────────────

# MAR formula: MAR = (v1 + v2 + v3) / (2 · h)
# Source: Rahman, A., Sirinukulwattana, K., & Togneri, R. (2015).
#         Video-based drowsiness detection using eye blink and yawning. ICCIT.
def compute_mar(lm: list, h: int, w: int) -> float:
    """
    Mouth Aspect Ratio — Rahman et al. (2015).

           v1 + v2 + v3
    MAR = ─────────────────────────
           2 × ||left_corner – right_corner||

    Where v1/v2/v3 are three vertical opening measurements across the mouth.

    Parameters
    ----------
    lm   : MediaPipe landmark list
    h, w : frame dimensions in pixels

    Returns
    -------
    float — MAR value (~0.10–0.25 resting; > 0.60 yawning); 0.0 on degenerate input
    """
    idx = MOUTH_IDX
    v1 = _dist(_px(lm, idx["upper_center"], h, w), _px(lm, idx["lower_center"], h, w))
    v2 = _dist(_px(lm, idx["upper_left"],   h, w), _px(lm, idx["lower_left"],   h, w))
    v3 = _dist(_px(lm, idx["upper_right"],  h, w), _px(lm, idx["lower_right"],  h, w))
    hz = _dist(_px(lm, idx["left_corner"],  h, w), _px(lm, idx["right_corner"], h, w))
    if hz < 1e-6:
        return 0.0
    return (v1 + v2 + v3) / (2.0 * hz)


# ── Branch C: Head Pose via PnP ───────────────────────────────────────────────

def _camera_matrix(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate pinhole camera intrinsic matrix.
    Focal length ≈ frame width (pixels); principal point at image centre.
    Distortion assumed zero — adequate for a consumer webcam at 0.5–1.5 m.

    K = [[W, 0, W/2],
         [0, W, H/2],
         [0, 0,   1]]
    """
    f      = float(w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    return K, D


# Perspective-n-Point recovery of head pose Euler angles.
# Source: Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009).
#         EPnP: An Accurate O(n) Solution to the PnP Problem.
#         International Journal of Computer Vision, 81(2), 155–166.
def compute_head_pose(
    lm: list, h: int, w: int
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate head-pose Euler angles via Perspective-n-Point (PnP).

    Method (Lepetit et al., 2009 — EPnP / Levenberg-Marquardt):
      1. Extract pixel coordinates of 6 stable anchor landmarks.
      2. cv2.solvePnP  → rotation vector (rvec) + translation vector.
      3. cv2.Rodrigues(rvec) → 3×3 rotation matrix R.
      4. cv2.RQDecomp3x3(R) → Euler angles (degrees).

    Parameters
    ----------
    lm   : MediaPipe landmark list
    h, w : frame dimensions in pixels

    Returns
    -------
    (pitch, yaw, roll) as Python floats (degrees), or (None, None, None) on failure.
    Sign convention (OpenCV):
        pitch > 0 → head tilted down;    pitch < 0 → head tilted up
        yaw   > 0 → head turned right;   yaw   < 0 → head turned left
    """
    if not lm:
        return None, None, None

    K, D = _camera_matrix(h, w)

    pts2d = np.array(
        [[lm[i].x * w, lm[i].y * h] for i in HEAD_POSE_ANCHOR_LM],
        dtype=np.float64,
    )

    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL, pts2d, K, D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(R)
    return float(angles[0]), float(angles[1]), float(angles[2])


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_feature_vector(
    ear: float,
    perclos: float,
    mar: float,
    pitch: Optional[float],
    yaw: Optional[float],
    roll: Optional[float],
) -> np.ndarray:
    """
    Build the 6-dimensional feature vector that feeds the SVM classifier.

    Column order MUST match config.FEATURE_NAMES and the training CSV:
      [ear_avg, perclos_pct, mar, pitch_deg, yaw_deg, roll_deg]

    Parameters
    ----------
    ear, perclos, mar   : float — ocular and oral feature values
    pitch, yaw, roll    : float or None — head-pose Euler angles
                          (None is coerced to 0.0 per Section 5.4.1)

    Returns
    -------
    np.ndarray of shape (6,) and dtype float64
    """
    p = float(pitch) if pitch is not None else 0.0
    y = float(yaw)   if yaw   is not None else 0.0
    r = float(roll)  if roll  is not None else 0.0
    return np.array([ear, perclos, mar, p, y, r], dtype=np.float64)
