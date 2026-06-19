# =============================================================================
# config.py — ALERTO Baseline: ALL fixed global thresholds and constants
# =============================================================================
# STUDY REFERENCE:
#   Jabbar, R., et al. (2018). Real-time driver drowsiness detection for
#   embedded system using model compression of deep neural networks.
#   Procedia Computer Science, 130, 736–743.
#
#   EAR algorithm: Soukupová & Čech (2016). Real-time eye blink detection
#   using facial landmarks. 21st CVWW.
#
#   Head pose: Lepetit, V., et al. (2009). EPnP: An Accurate O(n) Solution
#   to the PnP Problem. IJCV 81(2), 155–166.
#
# KEY DESIGN DECISION:
#   ALL thresholds below are GLOBALLY FIXED — the same for every user.
#   There is NO per-user calibration phase. This is the critical limitation
#   that the full ALERTO thesis resolves. Do NOT add calibration here.
#
# Course: COSC 304 — Introduction to Artificial Intelligence
# Group : BSCS 3-3, Group 7 — PUP CCIS, A.Y. 2025–2026
# =============================================================================

import numpy as np

# ── Eye Aspect Ratio (EAR) ─────────────────────────────────────────────────
EAR_CLOSED_THRESHOLD   = 0.25   # Global fixed; Soukupová & Čech (2016)
                                 # NOTE: Filipino resting EAR ≈ 0.316
                                 # → gap of only ~0.066 → higher false-positive rate
EAR_CONSECUTIVE_FRAMES = 50     # Sustained closed-eye frames → DROWSY alarm
                                 # @ 30 fps: 50 frames ≈ 1.67 seconds

# ── PERCLOS ────────────────────────────────────────────────────────────────
PERCLOS_WINDOW_FRAMES  = 60     # Rolling window (60 frames ≈ 2 s @ 30 fps)
PERCLOS_THRESHOLD      = 30.0   # > 30% eye-closure in window → DROWSY
                                 # Source: NHTSA (Wierwille & Ellsworth, 1994)

# ── Mouth Aspect Ratio (MAR) ───────────────────────────────────────────────
MAR_YAWN_THRESHOLD     = 0.60   # Global fixed; Rahman et al. (2015)
MAR_CONSECUTIVE_FRAMES = 30     # Sustained yawn frames → DROWSY alarm
                                 # @ 30 fps: 30 frames ≈ 1.0 second

# ── Head Pose (Spatial Branch — PnP) ──────────────────────────────────────
PITCH_THRESH_LOW   = -25.0   # Upward gaze limit
PITCH_THRESH_HIGH  =  30.0   # Downward gaze limit (seated driver typically 10–25°)
YAW_THRESHOLD      =  35.0   # Lateral head turn limit
DISTRACTION_FRAMES =  72     # Sustained off-axis frames → DISTRACTED alarm
                              # @ ~48 fps: 72 frames ≈ 1.5 seconds

# ── MediaPipe Landmark Indices (classic face_mesh API) ─────────────────────
# Right eye: p1=outer canthus, p4=inner canthus, p2/p3=upper lid, p5/p6=lower lid
RIGHT_EYE_IDX = {"p1": 33,  "p2": 160, "p3": 158,
                  "p4": 133, "p5": 153, "p6": 144}

# Left eye
LEFT_EYE_IDX  = {"p1": 362, "p2": 385, "p3": 387,
                  "p4": 263, "p5": 373, "p6": 380}

# Mouth: three vertical pairs + one horizontal width pair
MOUTH_IDX = {
    "upper_center": 13,   "lower_center": 14,
    "upper_left":   82,   "lower_left":   87,
    "upper_right":  312,  "lower_right":  317,
    "left_corner":  78,   "right_corner": 308,
}

# Head pose: 6 stable anchor landmarks for PnP
HEAD_POSE_ANCHOR_LM = [1, 152, 263, 33, 61, 291]
# [nose_tip, chin, left_eye_outer, right_eye_outer, left_mouth, right_mouth]

# Generic 3D face model in mm (anthropometric, nose tip = origin)
FACE_3D_MODEL = np.array([
    [  0.0,    0.0,    0.0],  # Landmark   1: Nose tip
    [  0.0, -330.0,  -65.0],  # Landmark 152: Chin
    [-165.0,  170.0, -135.0], # Landmark 263: Left eye outer
    [ 165.0,  170.0, -135.0], # Landmark  33: Right eye outer
    [-150.0, -150.0, -125.0], # Landmark  61: Left mouth corner
    [ 150.0, -150.0, -125.0], # Landmark 291: Right mouth corner
], dtype=np.float64)

# ── SVM Classifier ─────────────────────────────────────────────────────────
SVM_MODEL_PATH  = "data/svm_baseline_model.pkl"   # Trained model path
SVM_SCALER_PATH = "data/svm_baseline_scaler.pkl"  # StandardScaler path

# Label encoding — MUST match training order
LABEL_MAP  = {0: "ALERT", 1: "DROWSY", 2: "DISTRACTED"}
LABEL_INT  = {"ALERT": 0, "DROWSY": 1, "DISTRACTED": 2}

# Feature vector order — MUST match training script
# [ear_avg, perclos_pct, mar, pitch_deg, yaw_deg, roll_deg]
FEATURE_NAMES = ["ear_avg", "perclos_pct", "mar", "pitch_deg", "yaw_deg", "roll_deg"]

# ── System Performance ─────────────────────────────────────────────────────
TARGET_FPS        = 30
TARGET_LATENCY_MS = 50

# ── Display Colors (BGR for OpenCV) ───────────────────────────────────────
COLOR_ALERT      = (0, 255, 100)   # Green
COLOR_DROWSY     = (0, 165, 255)   # Amber
COLOR_DISTRACTED = (0,  60, 255)   # Red
COLOR_NONE       = (120, 120, 120) # Grey (no face)
