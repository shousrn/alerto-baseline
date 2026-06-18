# ALERTO Baseline — Dataset Directory

This directory contains the data files required to run the ALERTO Baseline pipeline.

## Required Files

### 1. `face_landmarker.task` (MediaPipe model)

**Required location:** `dataset/face_landmarker.task`

Download with PowerShell:
```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile "dataset/face_landmarker.task"
```

Or download with curl:
```bash
curl -o dataset/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

**Size:** ~16 MB  
**Source:** [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

---

### 2. `NTHU-DDD/` — NTHU Drowsy Driver Detection Dataset

**Required location:** `dataset/NTHU-DDD/Multi class/train/`

Expected subdirectory structure:
```
dataset/
  NTHU-DDD/
    Multi class/
      train/
        notdrowsy/     ← Alert images (also serves as DISTRACTED source via head-pose filtering)
          <subject>/
            *.jpg
        drowsy/        ← Drowsy images
          <subject>/
            *.jpg
```

**Source:** Contact the NTHU research group via the dataset request form at  
https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

**License:** For academic and research use only. Do not redistribute.

**Notes:**
- Only the `train/` partition is used; `test/` is not read by any script.
- The `NTHU-DDD` corpus contains **Asian subjects**, which is intentional for
  this baseline study.

---

## What Is NOT Used

### `YawDD/` — Yawning Driver Detection Dataset

The `dataset/YawDD/` subdirectory **is NOT used by the baseline pipeline.**

A previous version of the codebase generated synthetic Caucasian feature
distributions to simulate YawDD data, but this was identified as a
methodological flaw and has been removed. All training data now comes
exclusively from real NTHU-DDD images.

---

## After Downloading

Run the training pipeline in this order:

```bash
py training/extract_features.py   # Extract features from NTHU-DDD images
py training/train_svm.py          # Train the SVM classifier
py evaluation/evaluate_baseline.py  # Evaluate on held-out test set
py main.py                        # Run live webcam inference
```
