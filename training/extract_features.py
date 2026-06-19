"""
training/extract_features.py — Module 3.1: Feature Extraction Pipeline
========================================================================
Extracts the 6-dimensional feature vector from every NTHU-DDD image and
writes the result to data/extracted_features.csv.

Source corpus : NTHU Drowsy Driver Detection (NTHU-DDD) — Asian subjects.
No synthetic data is generated; all samples are derived from real images.

Label assignment (in priority order, with class-purity filter):
  1  DROWSY      — image comes from the `drowsy/` subdirectory AND the
                   recovered head pose is within normal range (|pitch| <= 20°
                   and |yaw| <= 30°). Drowsy frames with extreme pose are
                   SKIPPED to prevent inter-class label noise at the
                   Drowsy / Distracted decision boundary.
  2  DISTRACTED  — image comes from `notdrowsy/` and recovered head-pose
                   angles exceed the distraction thresholds
                   (|pitch| > 20° or |yaw| > 30°).
  0  ALERT       — image comes from `notdrowsy/` and head pose is within
                   the normal range.

PERCLOS at training time uses a per-frame static approximation (Section 5.8
of the documentation): 100 % if EAR < threshold, else 0 %. The true 60-frame
sliding-window PERCLOS is restored at inference time by PERCLOSBuffer.

CSV schema: ear_avg, perclos_pct, mar, pitch_deg, yaw_deg, roll_deg, label

Sample cap : 800 per class (2 400 total) with stratified 70/30 split at
             training time → 1 680 train / 720 test.

Usage:
    py training/extract_features.py

Course: COSC 304 — Introduction to Artificial Intelligence
Group : BSCS 3-3, Group 7 — PUP CCIS, A.Y. 2025–2026
"""

import os
import sys
import glob
import random

import cv2
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Allow imports from the project root when run from any working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    EAR_CLOSED_THRESHOLD,
    PITCH_THRESH_LOW,
    PITCH_THRESH_HIGH,
    YAW_THRESHOLD,
    FEATURE_NAMES,
    LABEL_INT,
)
from modules.features import compute_avg_ear, compute_mar, compute_head_pose

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH  = "dataset/face_landmarker.task"
NTHU_ROOT   = "dataset/NTHU-DDD/Multi class/train"
OUT_CSV     = "data/extracted_features.csv"
MAX_PER_CLASS = 800   # balanced cap per label (0, 1, 2)


# ---------------------------------------------------------------------------
# MediaPipe initialisation — IMAGE mode for static frames
# ---------------------------------------------------------------------------

def _build_landmarker() -> vision.FaceLandmarker:
    """Initialise MediaPipe FaceLandmarker in IMAGE mode for static images."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe model not found at '{MODEL_PATH}'. "
            "Download from https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
            "and place at the indicated path."
        )
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,   # IMAGE: stateless, no timestamps
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Per-image feature extraction
# ---------------------------------------------------------------------------

def _extract_one(image_path: str, landmarker: vision.FaceLandmarker) -> list | None:
    """
    Extract the 6D feature vector from a single image.

    Returns [ear_avg, perclos_pct, mar, pitch_deg, yaw_deg, roll_deg]
    or None if no face is detected or if solvePnP fails.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    h, w = frame.shape[:2]

    ear = compute_avg_ear(lm, h, w)
    mar = compute_mar(lm, h, w)
    pitch, yaw, roll = compute_head_pose(lm, h, w)

    if pitch is None:
        pitch, yaw, roll = 0.0, 0.0, 0.0

    # Section 5.8 static-image PERCLOS approximation:
    # 100 % if eye is classified as closed for this frame, else 0 %.
    perclos_pct = 100.0 if ear < EAR_CLOSED_THRESHOLD else 0.0

    return [ear, perclos_pct, mar, float(pitch), float(yaw), float(roll)]


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def _assign_label(source_dir: str, pitch: float, yaw: float) -> int:
    """
    Return integer label for a sample, in priority order:
      1 (DROWSY)      — image is from the 'drowsy/' subdirectory.
      2 (DISTRACTED)  — image is from 'notdrowsy/' with |pitch| > 25° or |yaw| > 35°.
      0 (ALERT)       — image is from 'notdrowsy/' with |pitch| < 15° and |yaw| < 20°.
     -1 (SKIP)        — ambiguous border frame; excluded from the training set.
    """
    if "drowsy" in source_dir.replace("notdrowsy", ""):
        return LABEL_INT["DROWSY"]

    abs_p, abs_y = abs(pitch), abs(yaw)
    if abs_p > 25.0 or abs_y > 35.0:
        return LABEL_INT["DISTRACTED"]
    if abs_p < 15.0 and abs_y < 20.0:
        return LABEL_INT["ALERT"]
    return -1


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full feature extraction pipeline and write data/extracted_features.csv."""
    print("\n" + "=" * 65)
    print("  ALERTO Baseline — Feature Extraction  (Module 3.1)")
    print("  Source corpus : NTHU-DDD (Asian subjects, real images)")
    print("=" * 65 + "\n")

    # -- Locate images -------------------------------------------------------
    notdrowsy_imgs = glob.glob(
        os.path.join(NTHU_ROOT, "notdrowsy", "**", "*.jpg"), recursive=True
    )
    drowsy_imgs = glob.glob(
        os.path.join(NTHU_ROOT, "drowsy", "**", "*.jpg"), recursive=True
    )

    if not notdrowsy_imgs and not drowsy_imgs:
        # Also try .png
        notdrowsy_imgs = glob.glob(
            os.path.join(NTHU_ROOT, "notdrowsy", "**", "*.png"), recursive=True
        )
        drowsy_imgs = glob.glob(
            os.path.join(NTHU_ROOT, "drowsy", "**", "*.png"), recursive=True
        )

    print(f"  Found {len(notdrowsy_imgs):,} notdrowsy images")
    print(f"  Found {len(drowsy_imgs):,} drowsy images")

    if not notdrowsy_imgs and not drowsy_imgs:
        print("[ERROR] No images found under", NTHU_ROOT)
        print("        Verify that NTHU-DDD is placed at dataset/NTHU-DDD/")
        sys.exit(1)

    # Shuffle for random sampling
    random.seed(42)
    random.shuffle(notdrowsy_imgs)
    random.shuffle(drowsy_imgs)

    # -- Build landmarker ----------------------------------------------------
    print("\n  Initialising MediaPipe FaceLandmarker (IMAGE mode)...")
    landmarker = _build_landmarker()
    print("  Landmarker ready.\n")

    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    records: list[list] = []

    all_images = (
        [(p, "drowsy")    for p in drowsy_imgs] +
        [(p, "notdrowsy") for p in notdrowsy_imgs]
    )
    random.shuffle(all_images)

    for img_path, src_dir in all_images:
        if all(v >= MAX_PER_CLASS for v in counts.values()):
            break

        features = _extract_one(img_path, landmarker)
        if features is None:
            continue

        ear, perclos_pct, mar, pitch, yaw, roll = features
        label = _assign_label(src_dir, pitch, yaw)

        if label == -1:
            continue   # Skip extreme-pose drowsy frames

        if counts[label] >= MAX_PER_CLASS:
            continue   # This class is full; try next image

        counts[label] += 1
        records.append([ear, perclos_pct, mar, pitch, yaw, roll, label])

        # Progress report every 100 records
        total = sum(counts.values())
        if total % 100 == 0:
            print(
                f"  Progress: {total:4d} total | "
                f"ALERT={counts[0]:3d} | "
                f"DROWSY={counts[1]:3d} | "
                f"DISTRACTED={counts[2]:3d}"
            )

    landmarker.close()

    # -- Write CSV -----------------------------------------------------------
    columns = FEATURE_NAMES + ["label"]
    df = pd.DataFrame(records, columns=columns)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    total = len(df)
    print(f"\n{'=' * 65}")
    print("  Extraction complete.")
    print(f"  ALERT      (label 0): {counts[0]} samples")
    print(f"  DROWSY     (label 1): {counts[1]} samples")
    print(f"  DISTRACTED (label 2): {counts[2]} samples")
    print(f"  Total: {total} samples written to {OUT_CSV}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
