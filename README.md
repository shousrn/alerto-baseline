# ALERTO Baseline

> **Adaptive Learning for Enhanced Real-Time Operations — Baseline System**  
> COSC 304 — Introduction to Artificial Intelligence  
> BSCS 3-3 Group 7 · PUP CCIS · A.Y. 2025–2026

---

## Overview

ALERTO Baseline is a real-time driver drowsiness and distraction detection
system. It implements a fixed, globally-normalised classification pipeline to
serve as the **controlled baseline** for a follow-up thesis that introduces
individualized calibration.

The system detects three driver states:
- **ALERT** — normal driving behaviour
- **DROWSY** — signs of microsleep or fatigue (low EAR / high PERCLOS / yawning)
- **DISTRACTED** — off-axis head pose exceeding safe thresholds

### AI algorithms implemented

| Phase | Algorithm | Reference |
|---|---|---|
| Phase 1 | MediaPipe FaceLandmarker (478-pt, VIDEO mode) | Google, 2023 |
| Phase 2A | Eye Aspect Ratio (EAR) | Soukupová & Čech, CVWW 2016 |
| Phase 2A | PERCLOS (60-frame sliding window) | Wierwille & Ellsworth, 1994 |
| Phase 2B | Mouth Aspect Ratio (MAR) | Rahman et al., ICCIT 2015 |
| Phase 2C | Head Pose via PnP (Lepetit/Rodrigues) | Lepetit et al., IJCV 2009 |
| Phase 3 | SVM (RBF kernel, global StandardScaler) | Jabbar et al., 2018 |

---

## Prerequisites

- Python 3.11+
- A webcam (USB or built-in)
- Windows / Linux / macOS

---

## Setup

### 1. Clone and enter the repository

```bash
git clone <repo-url>
cd AI_Project
```

### 2. Create a virtual environment

```bash
py -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux / macOS
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model

```powershell
# Windows PowerShell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile "dataset/face_landmarker.task"
```

```bash
# Linux / macOS
curl -o dataset/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

### 5. Obtain the NTHU-DDD dataset

Request access from the NTHU research group:  
https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

Place the dataset at:
```
dataset/
  NTHU-DDD/
    Multi class/
      train/
        notdrowsy/
        drowsy/
```

See [`dataset/README.md`](dataset/README.md) for full details.

---

## Training the SVM model

Run these three steps in order (takes 2–5 minutes depending on image count):

```bash
# Step 1: Extract features from NTHU-DDD images
py training/extract_features.py

# Step 2: Train the SVM classifier
py training/train_svm.py

# Step 3: Evaluate on the held-out test set
py evaluation/evaluate_baseline.py
```

After training, the following artifacts are produced:
- `data/extracted_features.csv` — 6D feature vectors with labels
- `data/svm_baseline_model.pkl` — trained SVC
- `data/svm_baseline_scaler.pkl` — fitted StandardScaler
- `outputs/training_report.txt` — classification report + CV accuracy
- `outputs/confusion_matrix.png` — training-phase confusion matrix
- `outputs/evaluation_report.txt` — held-out test set report
- `outputs/evaluation_confusion_matrix.png` — evaluation confusion matrix

---

## Running the system

### Primary method — Web interface (Flask + React)

This is the main deliverable. Run both commands in separate terminals:

**Terminal 1 — Start the backend (detection + API):**

```bash
py app.py
```

**Terminal 2 — Start the React frontend:**

```bash
cd frontend
npm install      # only needed first time
npm run dev
```

Open your browser to **http://localhost:5173**.

The React dev server (`npm run dev`) proxies API calls to Flask on port 5000. Both must be running at the same time.

To serve the frontend directly from Flask (single command, no separate terminal):

```bash
cd frontend && npm run build   # build once
cd ..
py app.py                      # visit http://localhost:5000
```

---


## Project structure

```
AI_Project/
├── main.py                    # Standalone OpenCV entry point
├── app.py                     # Flask web backend
├── config.py                  # All constants and thresholds (single source of truth)
├── requirements.txt
├── README.md
│
├── modules/
│   ├── features.py            # Phase 2: EAR, PERCLOS, MAR, Head Pose
│   ├── classifier.py          # Phase 3: SVMClassifier + RuleBasedClassifier
│   └── display.py             # Phase 4: OpenCV HUD overlay
│
├── training/
│   ├── extract_features.py    # Feature extraction from NTHU-DDD images
│   └── train_svm.py           # SVM training pipeline
│
├── evaluation/
│   └── evaluate_baseline.py   # Hold-out test set evaluation
│
├── data/                      # Generated CSV and pickle artifacts
├── outputs/                   # Reports, confusion matrix PNGs
├── assets/
│   └── alert.wav              # 800 Hz alert tone (1 second, 44.1 kHz)
├── dataset/
│   ├── README.md              # Dataset download instructions
│   ├── face_landmarker.task   # MediaPipe model (download separately)
│   └── NTHU-DDD/              # Training images (download separately)
│
├── frontend/                  # React + Vite + TailwindCSS web UI
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── Landing.jsx
│       │   ├── DriverView.jsx
│       │   ├── SystemDashboard.jsx
│       │   └── DeepAnalytics.jsx
│       └── hooks/
│           └── useMetrics.js
│
└── tests/
    └── test_features.py       # Unit tests (pytest)
```

---

## Running unit tests

```bash
py -m pytest tests/ -v --cov=modules --cov-report=term-missing
```

---

## Known limitations (intentional — baseline design)

These limitations are documented constraints, not bugs. They serve as the
measurement baseline for the ALERTO thesis:

1. **Fixed EAR threshold (0.25)** — not adapted per user. Asian subjects with
   naturally lower resting EAR values may generate more false drowsiness alerts.
2. **Global StandardScaler** — fitted on pooled NTHU-DDD training data.
   No per-user normalization is applied.
3. **No calibration phase** — the system cannot adapt to individual facial
   morphology, lighting, or camera position.
4. **Head-pose only for distraction** — pupil tracking and gaze estimation
   are outside the scope of this baseline.

---

## Citation

If you build upon this work, cite the underlying algorithms:

- Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. *21st CVWW*.
- Wierwille, W. W., & Ellsworth, L. A. (1994). Evaluation of driver drowsiness by trained raters. *Accident Analysis & Prevention, 26*(5).
- Rahman, A., Sirinukulwattana, K., & Togneri, R. (2015). Video-based drowsiness detection. *ICCIT*.
- Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). EPnP. *IJCV, 81*(2).
- Jabbar, R., et al. (2018). Real-time driver drowsiness detection. *Procedia CS, 130*.
- Kim, D., et al. (2023). Real-time driver monitoring system. *Scientific Reports, 13*(1).
