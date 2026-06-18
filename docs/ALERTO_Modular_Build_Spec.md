# ALERTO Baseline — Modular Phase-by-Phase Build Specification

**Intended Audience.** This document is written to be consumed by an AI coding agent inside the Antigravity IDE, one module at a time. Each module is a self-contained task specification. The team should hand the agent one module per session, review the resulting diff, run the verification command, commit, and only then move to the next module.

**Relationship to Other Documents.**
- The **AI Project Documentation** is the immutable specification of system behavior.
- The **Implementation and Refactoring Plan** is the human-facing project schedule.
- This **Modular Build Specification** is the agent-facing task queue. It refers back to the documentation for behavioral details and to the implementation plan for sequencing.

**Module Numbering.** Modules are numbered `M.S` where `M` is the architectural phase from the documentation and `S` is the sub-task index. Modules 0.x are scaffolding, 1.x is Phase 1, 2.x is Phase 2, 3.x is Phase 3, 4.x is Phase 4 (backend), 5.x is the React frontend, 6.x is testing, 7.x is documentation and deployment.

---

## Agent Operating Guidance (apply to every module)

The following constraints apply across every module. Mention them in the first agent prompt of every session if the agent does not retain context across sessions.

1. **Python style.** PEP 8. Use type hints on every function signature. Four-space indentation. Maximum line length 100 characters.
2. **Docstrings.** Every module-level file, class, and public function must carry a docstring. The docstring of a function that implements a published formula must cite the source in the form `<formula> — Author (Year)`, for example `EAR = (||p2-p6|| + ||p3-p5||) / (2 · ||p1-p4||) — Soukupová & Čech (2016)`.
3. **Imports.** Group imports as: standard library, third-party, project-local, in that order, separated by a blank line.
4. **Error handling.** Use specific exception types. Provide actionable error messages (state what was attempted, what failed, and what to check).
5. **Avoid magic numbers in code bodies.** All threshold values, frame counts, and landmark indices must be referenced through symbols imported from `config.py`.
6. **No new dependencies without justification.** Any addition to `requirements.txt` must be justified in the commit message.
7. **Idempotency.** Refactors must preserve the public interface of any function or class that is used by another module. If an interface must change, update every caller in the same commit.
8. **Test parity.** Whenever a function's behavior is added or modified, the unit test file `tests/test_features.py` must be updated in the same commit. Do not leave the tests behind.
9. **No commented-out code.** Deleted code must be deleted, not commented out. Source-control history is the audit trail.
10. **Citations as comments.** When implementing a function that traces to a paper, place a brief one-line comment immediately above the function definition citing the paper. This satisfies the professor's requirement that "what was implemented" is clearly distinguished from "what library was used".

---

## Module 0.1 — Project Scaffolding and Dependency Reconciliation

| Property | Value |
|---|---|
| **Files** | `requirements.txt`, `README.md`, `dataset/README.md` (create), `assets/` (create directory) |
| **Status** | Refactor |
| **Depends on** | None |
| **Documentation reference** | Sections 5.7, 4 (scope) |
| **Estimated effort** | 0.5 person-day |

### Specification

Bring the project's dependency declaration and top-level documentation into alignment with the documentation that has just been finalized.

1. Edit `requirements.txt` to contain the following lines (preserve any other lines already present that are not contradicted):

   ```
   opencv-python>=4.9.0.80,<5.0
   mediapipe>=0.10.14,<0.11
   numpy>=1.26.0,<2.0
   scipy>=1.13.0,<2.0
   scikit-learn>=1.4.2,<2.0
   pandas>=2.2.0,<3.0
   pytest>=8.2.0,<9.0
   pytest-cov>=5.0.0,<6.0
   flask>=3.0.0,<4.0
   flask-cors>=4.0.0,<5.0
   simpleaudio>=1.0.4,<2.0
   matplotlib>=3.8.0,<4.0
   ```

2. Create the directory `assets/` at the project root if it does not exist. This directory will contain `alert.wav` (produced in Module 4.2).

3. Create the file `dataset/README.md` with content that explains where `face_landmarker.task` is sourced from, where NTHU-DDD is downloaded from, and a note that the `dataset/YawDD/` subdirectory is no longer used by the baseline pipeline.

4. Replace the project root `README.md` with the structure listed in Implementation Plan Task R0.7 (do not produce a stub; produce the full README).

### Acceptance

- `pip install -r requirements.txt` completes in a clean virtual environment without warning.
- `dataset/README.md` is present and documents the data sources.
- A new reader can follow `README.md` and reach a working webcam preview within thirty minutes.

### Verification command

```bash
python -c "import cv2, mediapipe, numpy, scipy, sklearn, pandas, flask, simpleaudio, matplotlib; print('OK')"
```

---

## Module 1.1 — Phase 1: MediaPipe FaceLandmarker Initialization

| Property | Value |
|---|---|
| **Files** | `modules/features.py` (audit), `main.py` (audit), `app.py` (audit) |
| **Status** | Validate, repair if drift detected |
| **Depends on** | Module 0.1 |
| **Documentation reference** | Section 5.2 |
| **Estimated effort** | 1 person-day |

### Specification

The MediaPipe Tasks API is the only landmarker the project may use; the legacy `mp.solutions.face_mesh` interface was removed in MediaPipe ≥ 0.10.30. Audit every site where a landmarker is constructed and confirm or correct the following.

1. The landmarker is constructed exactly as:

   ```python
   from mediapipe.tasks import python as mp_python
   from mediapipe.tasks.python import vision

   base_options = mp_python.BaseOptions(
       model_asset_path="dataset/face_landmarker.task"
   )
   options = vision.FaceLandmarkerOptions(
       base_options=base_options,
       running_mode=vision.RunningMode.VIDEO,
       num_faces=1,
       min_face_detection_confidence=0.5,
       min_face_presence_confidence=0.5,
       min_tracking_confidence=0.5,
   )
   landmarker = vision.FaceLandmarker.create_from_options(options)
   ```

2. The running mode is `VIDEO` in `main.py` and `app.py`. (It is `IMAGE` in `training/extract_features.py`; do not change that script in this module.)

3. Every frame fed to `landmarker.detect_for_video` is:
   - In RGB color order (apply `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`).
   - Wrapped as `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)`.
   - Paired with a monotonically increasing integer millisecond timestamp derived from `int(time.time() * 1000)`. Maintain a `last_timestamp_ms` variable and increment it explicitly if `time.time()` produces two adjacent calls with the same millisecond value.

4. Wrap landmarker creation in a `try` block. On `FileNotFoundError`, emit:
   > "MediaPipe model file 'dataset/face_landmarker.task' not found. Download from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task and place at the indicated path."

### Acceptance

- `grep -rn "mp.solutions" --include="*.py" .` returns no matches in the runtime path.
- `main.py` launches against a live webcam and renders the face-mesh dots within five seconds.
- Deleting `dataset/face_landmarker.task` and running `main.py` produces the actionable error message above.

### Verification command

```bash
python -c "from mediapipe.tasks.python import vision; print(vision.RunningMode.VIDEO)"
python main.py  # close after visual verification
```

---

## Module 2.1 — Phase 2: Eye Aspect Ratio (EAR)

| Property | Value |
|---|---|
| **Files** | `modules/features.py`, `config.py` |
| **Status** | Validate, repair if drift detected |
| **Depends on** | Module 1.1 |
| **Documentation reference** | Section 5.3.1 |
| **Estimated effort** | 0.5 person-day |

### Specification

Confirm the EAR implementation against Soukupová & Čech (2016).

1. In `config.py`, confirm that the landmark indices are exactly as documented:

   ```python
   RIGHT_EYE_IDX = {"p1": 33, "p2": 160, "p3": 158, "p4": 133, "p5": 153, "p6": 144}
   LEFT_EYE_IDX  = {"p1": 362, "p2": 385, "p3": 387, "p4": 263, "p5": 373, "p6": 380}
   EAR_CLOSED_THRESHOLD = 0.25
   ```

2. In `modules/features.py`, confirm that `compute_ear(landmarks, eye_idx, h, w)` implements

   ```
   EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)
   ```

   Return `0.0` (not raise) when the horizontal denominator is below `1e-6`.

3. Confirm that `compute_avg_ear(landmarks, h, w)` returns the arithmetic mean of the two per-eye values.

4. Place this citation comment immediately above `compute_ear`:

   ```python
   # EAR formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 · ||p1-p4||)
   # Source: Soukupová, T. & Čech, J. (2016). Real-time eye blink detection using facial landmarks. CVWW.
   ```

### Acceptance

- All four tests in `tests/test_features.py::TestComputeEAR` pass.
- A live test with eyes open produces values in the interval [0.28, 0.42].
- A live test with eyes closed produces values below 0.15.

### Verification command

```bash
python -m pytest tests/test_features.py::TestComputeEAR -v
```

---

## Module 2.2 — Phase 2: PERCLOS Sliding-Window Buffer

| Property | Value |
|---|---|
| **Files** | `modules/features.py`, `config.py` |
| **Status** | Validate |
| **Depends on** | Module 2.1 |
| **Documentation reference** | Section 5.3.1 |
| **Estimated effort** | 0.5 person-day |

### Specification

Confirm the PERCLOS implementation against Wierwille & Ellsworth (1994).

1. In `config.py`:

   ```python
   PERCLOS_WINDOW_FRAMES = 60   # ≈ 2 seconds at 30 FPS
   PERCLOS_THRESHOLD     = 30.0 # percent
   ```

2. In `modules/features.py`, confirm the `PERCLOSBuffer` class has this signature and behavior:

   ```python
   class PERCLOSBuffer:
       def __init__(self, window_size: int = PERCLOS_WINDOW_FRAMES,
                          threshold: float = EAR_CLOSED_THRESHOLD) -> None: ...
       def update(self, ear: float) -> float:
           """Append 1 if ear < threshold else 0; return current PERCLOS percent."""
       def reset(self) -> None: ...
       @property
       def ready(self) -> bool:
           """True once the buffer has reached its full window."""
   ```

3. The internal storage must be `collections.deque(maxlen=window_size)`. Do not use a plain list; the deque's `maxlen` provides automatic eviction, which is the algorithmic requirement.

4. Place this citation comment immediately above the class:

   ```python
   # PERCLOS: percentage of frames in a sliding window where EAR < threshold.
   # Source: Wierwille, W. W. & Ellsworth, L. A. (1994). Evaluation of driver drowsiness by trained raters. Accident Analysis & Prevention.
   ```

### Acceptance

- All five tests in `tests/test_features.py::TestPERCLOSBuffer` pass.
- Manual test: feed 60 EAR values of 0.10 in sequence and confirm the buffer reports 100.0%.
- Manual test: feed 30 values of 0.10 followed by 30 values of 0.40 and confirm the buffer reports 50.0%.

### Verification command

```bash
python -m pytest tests/test_features.py::TestPERCLOSBuffer -v
```

---

## Module 2.3 — Phase 2: Mouth Aspect Ratio (MAR)

| Property | Value |
|---|---|
| **Files** | `modules/features.py`, `config.py` |
| **Status** | Validate |
| **Depends on** | Module 2.1 |
| **Documentation reference** | Section 5.3.2 |
| **Estimated effort** | 0.5 person-day |

### Specification

Confirm the MAR implementation against Rahman et al. (2015).

1. In `config.py`:

   ```python
   MOUTH_IDX = {
       "upper_center": 13, "lower_center": 14,
       "upper_left":   82, "lower_left":   87,
       "upper_right": 312, "lower_right":  317,
       "left_corner":  78, "right_corner": 308,
   }
   MAR_YAWN_THRESHOLD     = 0.60
   MAR_CONSECUTIVE_FRAMES = 30
   ```

2. In `modules/features.py`, confirm that `compute_mar(landmarks, h, w)` implements

   ```
   MAR = (v1 + v2 + v3) / (2 · h)
   ```

   where `v1`, `v2`, `v3` are the three vertical inner-lip distances at the center, left, and right of the inner mouth, and `h` is the horizontal distance between the lip corners. Return `0.0` when the horizontal denominator is below `1e-6`.

3. Place this citation comment immediately above the function:

   ```python
   # MAR formula: MAR = (v1 + v2 + v3) / (2 · h)
   # Source: Rahman, A., Sirinukulwattana, K., & Togneri, R. (2015). Video-based drowsiness detection using eye blink and yawning. ICCIT.
   ```

### Acceptance

- All three tests in `tests/test_features.py::TestComputeMAR` pass.
- A live test with mouth closed produces values in the interval [0.10, 0.25].
- A live test with mouth open in a yawn shape produces values exceeding 0.60.

### Verification command

```bash
python -m pytest tests/test_features.py::TestComputeMAR -v
```

---

## Module 2.4 — Phase 2: Head Pose via Perspective-n-Point

| Property | Value |
|---|---|
| **Files** | `modules/features.py`, `config.py` |
| **Status** | Validate |
| **Depends on** | Module 2.1 |
| **Documentation reference** | Section 5.3.3 |
| **Estimated effort** | 1 person-day |

### Specification

Confirm the head-pose implementation against Lepetit, Moreno-Noguer & Fua (2009).

1. In `config.py`, confirm:

   ```python
   import numpy as np

   HEAD_POSE_ANCHOR_LM = [1, 152, 263, 33, 61, 291]

   FACE_3D_MODEL = np.array([
       [   0.0,    0.0,    0.0],   # Nose tip       (LM 1)
       [   0.0, -330.0,  -65.0],   # Chin           (LM 152)
       [-165.0,  170.0, -135.0],   # Left eye outer (LM 263)
       [ 165.0,  170.0, -135.0],   # Right eye outer(LM 33)
       [-150.0, -150.0, -125.0],   # Left mouth     (LM 61)
       [ 150.0, -150.0, -125.0],   # Right mouth    (LM 291)
   ], dtype=np.float64)

   PITCH_THRESH_LOW  = -20.0
   PITCH_THRESH_HIGH =  20.0
   YAW_THRESHOLD     =  30.0
   DISTRACTION_FRAMES = 45
   ```

2. In `modules/features.py`, confirm that `compute_head_pose(landmarks, h, w)` follows this sequence:

   ```python
   # 1. Build camera intrinsic matrix K (focal length = frame width, principal point = frame center).
   # 2. Extract 2D pixel coordinates for the six anchor landmarks.
   # 3. Call cv2.solvePnP(FACE_3D_MODEL, points_2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE).
   # 4. Convert rotation vector to matrix via cv2.Rodrigues.
   # 5. Decompose rotation matrix to Euler angles via cv2.RQDecomp3x3.
   # 6. Return (pitch, yaw, roll) as Python floats, or (None, None, None) on failure.
   ```

3. Place this citation comment immediately above the function:

   ```python
   # Perspective-n-Point recovery of head pose Euler angles.
   # Source: Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). EPnP: An Accurate O(n) Solution to the PnP Problem. IJCV.
   ```

### Acceptance

- Sign-convention sanity check on a live camera: facing forward yields angles within ±5° of zero. Tilting the head down increases Pitch in the positive direction (per OpenCV convention; document the convention if the sign appears inverted on your hardware). Turning the head right (driver's left as seen by the camera) increases Yaw.

### Verification command

```bash
python -c "from modules.features import compute_head_pose; print('Function importable')"
python main.py  # close after visual verification of head-pose angles in the HUD
```

---

## Module 2.5 — Phase 2: Six-Dimensional Feature Vector Assembly

| Property | Value |
|---|---|
| **Files** | `modules/features.py` |
| **Status** | Validate |
| **Depends on** | Modules 2.1, 2.2, 2.3, 2.4 |
| **Documentation reference** | Section 5.4.1 |
| **Estimated effort** | 0.25 person-day |

### Specification

Confirm `extract_feature_vector` produces a NumPy array with the column order fixed across training and inference.

1. The function signature is exactly:

   ```python
   def extract_feature_vector(
       ear: float, perclos: float, mar: float,
       pitch: float | None, yaw: float | None, roll: float | None,
   ) -> np.ndarray:
       """Return [ear, perclos, mar, pitch, yaw, roll] as float64 array of shape (6,).
       Coerce any None among (pitch, yaw, roll) to 0.0."""
   ```

2. The constant `FEATURE_NAMES` in `config.py` must be exactly:

   ```python
   FEATURE_NAMES = ["ear_avg", "perclos_pct", "mar", "pitch_deg", "yaw_deg", "roll_deg"]
   ```

   This is the source of truth for the CSV header in Module 3.1 and for the scaler column order.

### Acceptance

- All three tests in `tests/test_features.py::TestExtractFeatureVector` pass.

### Verification command

```bash
python -m pytest tests/test_features.py::TestExtractFeatureVector -v
```

---

## Module 3.1 — Phase 3 Training: Feature Extraction Pipeline (CRITICAL REFACTOR)

| Property | Value |
|---|---|
| **Files** | `training/extract_features.py` (REWRITE), `data/extracted_features.csv` (REGENERATE) |
| **Status** | **Rewrite** — this is the most important refactor in the project |
| **Depends on** | Modules 2.1 through 2.5 |
| **Documentation reference** | Section 5.4.2 and 5.8 |
| **Estimated effort** | 1 person-day |

### Specification

Replace the existing `training/extract_features.py` with a clean implementation that uses NTHU-DDD exclusively and constructs all three class labels from real frames. **The existing synthetic Western-data generation block must be deleted in its entirety; do not keep it behind a flag.**

The new pipeline:

1. Initializes MediaPipe FaceLandmarker in `RunningMode.IMAGE` (note: `IMAGE`, not `VIDEO`, because we are processing independent static images).

2. Iterates over every `.jpg` and `.png` file under `dataset/NTHU-DDD/Multi class/train/notdrowsy/` and `dataset/NTHU-DDD/Multi class/train/drowsy/`, recursively.

3. For each image:
   - Reads with `cv2.imread`, converts BGR → RGB.
   - Runs `landmarker.detect(mp.Image(...))`. Skips the frame if no face is detected.
   - Computes `ear_avg`, `mar`, and `(pitch, yaw, roll)` using the functions audited in Modules 2.1–2.4.
   - Computes the per-frame PERCLOS approximation: `perclos_pct = 100.0 if ear_avg < 0.25 else 0.0`. (This is documented in Section 5.8 of the documentation as the static-image approximation; the true temporal PERCLOS is restored at inference time.)
   - Assigns the label as follows, in this order:
     - If the source directory contains `drowsy/`, assign label `1` (DROWSY).
     - Else if `abs(pitch) > 20.0` or `abs(yaw) > 30.0`, assign label `2` (DISTRACTED).
     - Else assign label `0` (ALERT).

4. Maintains three per-class counters. Stops sampling from any class once it has accumulated 800 samples.

5. Writes the resulting CSV to `data/extracted_features.csv` with the column header

   ```
   ear_avg,perclos_pct,mar,pitch_deg,yaw_deg,roll_deg,label
   ```

   and no leading index column (`index=False` in pandas).

6. Prints a summary to standard output:

   ```
   Extraction complete.
   ALERT      (label 0): NNN samples
   DROWSY     (label 1): NNN samples
   DISTRACTED (label 2): NNN samples
   Total: NNNN samples written to data/extracted_features.csv
   ```

### What must NOT appear in the new file

- Any `import numpy as np` followed by `np.random.normal(...)` for synthetic feature generation.
- Any reference to "YawDD", "Western", or "Caucasian" in identifiers or comments.
- Any `dataset` column in the CSV.

### Acceptance

- `data/extracted_features.csv` contains between 2,000 and 2,500 rows.
- Each of the three class counts is between 600 and 900.
- No row contains NaN.
- The file passes:

  ```bash
  python -c "
  import pandas as pd
  df = pd.read_csv('data/extracted_features.csv')
  assert list(df.columns) == ['ear_avg','perclos_pct','mar','pitch_deg','yaw_deg','roll_deg','label']
  assert df['label'].isin([0,1,2]).all()
  assert not df.isnull().any().any()
  print('CSV schema OK,', len(df), 'rows,', df['label'].value_counts().to_dict())
  "
  ```

### Verification command

```bash
python training/extract_features.py
python -c "import pandas as pd; df = pd.read_csv('data/extracted_features.csv'); print(df['label'].value_counts())"
```

---

## Module 3.2 — Phase 3 Training: SVM Training

| Property | Value |
|---|---|
| **Files** | `training/train_svm.py` (refactor), `data/svm_baseline_model.pkl`, `data/svm_baseline_scaler.pkl`, `outputs/confusion_matrix.png`, `outputs/training_report.txt` |
| **Status** | Refactor |
| **Depends on** | Module 3.1 |
| **Documentation reference** | Section 5.4 and 5.8 |
| **Estimated effort** | 0.5 person-day |

### Specification

Refactor `training/train_svm.py` to train a clean SVM on the regenerated dataset and produce honest, reproducible artifacts.

The pipeline:

1. Loads `data/extracted_features.csv` and separates `X` (six feature columns) from `y` (label column).

2. Performs a stratified 70/30 train-test split:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.30, stratify=y, random_state=42
   )
   ```

3. Fits `StandardScaler` on `X_train` only:

   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled  = scaler.transform(X_test)
   ```

4. Trains the SVM with documented hyperparameters:

   ```python
   model = SVC(kernel='rbf', C=1.0, gamma='scale',
               probability=True, random_state=42)
   model.fit(X_train_scaled, y_train)
   ```

5. Runs 5-fold stratified cross-validation on `X_train_scaled` and prints the mean and standard deviation of accuracy:

   ```python
   from sklearn.model_selection import cross_val_score
   cv_scores = cross_val_score(model, X_train_scaled, y_train,
                               cv=5, scoring='accuracy')
   print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
   ```

6. Evaluates on the held-out test set and produces a classification report:

   ```python
   y_pred = model.predict(X_test_scaled)
   report = classification_report(
       y_test, y_pred,
       target_names=["ALERT", "DROWSY", "DISTRACTED"],
       digits=4,
   )
   print(report)
   ```

7. Saves all artifacts:

   - `data/svm_baseline_model.pkl` (the trained SVC)
   - `data/svm_baseline_scaler.pkl` (the fitted StandardScaler)
   - `outputs/training_report.txt` (the printed report, plus the CV summary)
   - `outputs/confusion_matrix.png` (use `sklearn.metrics.ConfusionMatrixDisplay`)

### What must NOT appear in the new file

- Any code that splits the dataset by demographic or by `dataset` column.
- Any commentary referring to the Other-Race Effect.
- Any reference to a "YawDD" or "Western" subset.

### Acceptance

- The two `.pkl` files exist and load successfully with `pickle.load`.
- The classification report shows non-degenerate precision and recall (i.e., no class with precision or recall of 0.0).
- The confusion matrix is approximately diagonal-dominant (each diagonal element is the largest in its row).
- The 5-fold cross-validation accuracy is reported as `mean ± std`.

### Verification command

```bash
python training/train_svm.py
python -c "
import pickle
with open('data/svm_baseline_model.pkl','rb') as f: m = pickle.load(f)
with open('data/svm_baseline_scaler.pkl','rb') as f: s = pickle.load(f)
print('Loaded:', type(m).__name__, type(s).__name__)
print('n_support per class:', m.n_support_)
"
```

---

## Module 3.3 — Phase 3 Training: Evaluation Script Simplification

| Property | Value |
|---|---|
| **Files** | `evaluation/evaluate_baseline.py` (rewrite), `outputs/evaluation_report.txt` |
| **Status** | Rewrite |
| **Depends on** | Module 3.2 |
| **Documentation reference** | Section 5.9 |
| **Estimated effort** | 0.5 person-day |

### Specification

Replace `evaluation/evaluate_baseline.py` with a script that reports standard classification metrics on the held-out test partition, without any bias-narrative analysis.

The script:

1. Loads `data/extracted_features.csv`, performs the same 70/30 stratified split with `random_state=42` to obtain the same held-out test partition that `train_svm.py` used.

2. Loads `data/svm_baseline_model.pkl` and `data/svm_baseline_scaler.pkl`.

3. Applies the scaler to the test features and produces predictions.

4. Computes and writes to `outputs/evaluation_report.txt`:

   ```
   ALERTO BASELINE — Held-Out Test Set Evaluation
   ==============================================
   Test partition size:    <n_test>
   Overall accuracy:       <accuracy>
   
   Per-class metrics:
   <classification_report output>
   
   Confusion matrix:
   <3x3 matrix>
   
   Generated: <timestamp>
   ```

5. Writes the confusion-matrix figure as `outputs/evaluation_confusion_matrix.png` (so it does not collide with the training-stage one from Module 3.2).

### What must NOT appear in the new file

- Any block separating the test partition into subgroups by demographic.
- Any three-panel bar chart comparing FPR across subgroups.
- Any EAR-distribution histogram overlay.
- The string "Other-Race Effect", "ORE", "Asian", "Western", or "Caucasian".

### Acceptance

- `outputs/evaluation_report.txt` is produced.
- `outputs/evaluation_confusion_matrix.png` is produced.
- Neither output references demographic subgroups.

### Verification command

```bash
python evaluation/evaluate_baseline.py
grep -i "ORE\|other-race\|caucasian\|western" outputs/evaluation_report.txt
# The grep should return nothing.
```

---

## Module 3.4 — Phase 3 Inference: SVMClassifier with Hysteresis

| Property | Value |
|---|---|
| **Files** | `modules/classifier.py` |
| **Status** | Validate |
| **Depends on** | Module 3.2 |
| **Documentation reference** | Section 5.4.4, 5.4.5 |
| **Estimated effort** | 0.5 person-day |

### Specification

Confirm the `SVMClassifier` class operates as documented.

1. The class signature:

   ```python
   class SVMClassifier:
       def __init__(self) -> None: ...
       def load(self, model_path: str = SVM_MODEL_PATH,
                      scaler_path: str = SVM_SCALER_PATH) -> None: ...
       @property
       def is_loaded(self) -> bool: ...
       def predict(self, feature_vec: np.ndarray,
                         ear: float, mar: float,
                         pitch: float | None, yaw: float | None,
                  ) -> tuple[str, dict[str, float] | None]: ...
   ```

2. `predict` must:
   - Apply the scaler before invoking the model. This is a load-bearing requirement; skipping the scaler produces silently incorrect predictions.
   - Decode the integer label via `LABEL_MAP` (from `config.py`).
   - Attempt to return Platt-calibrated probabilities; if the underlying model raises `AttributeError`, return `proba=None` instead of crashing.
   - Pass the raw state through `_apply_hysteresis` before returning.

3. `_apply_hysteresis` must:
   - Maintain three counters `_ear_cnt`, `_mar_cnt`, `_pose_cnt`.
   - Increment each counter when its raw geometric condition holds and reset to zero otherwise.
   - Downgrade a raw `DROWSY` to `ALERT` unless `_ear_cnt >= 50` or `_mar_cnt >= 30`.
   - Downgrade a raw `DISTRACTED` to `ALERT` unless `_pose_cnt >= 45`.

### Acceptance

- All three tests in `tests/test_features.py::TestSVMClassifier` pass.
- All five tests in `tests/test_features.py::TestRuleBasedClassifier` pass (covers the fallback in Module 3.5).
- Manual test: with `data/svm_baseline_model.pkl` deleted, `SVMClassifier.load()` raises `FileNotFoundError` with an actionable message.

### Verification command

```bash
python -m pytest tests/test_features.py::TestSVMClassifier tests/test_features.py::TestRuleBasedClassifier -v
```

---

## Module 3.5 — Phase 3 Inference: Rule-Based Fallback

| Property | Value |
|---|---|
| **Files** | `modules/classifier.py` |
| **Status** | Validate |
| **Depends on** | Module 3.4 |
| **Documentation reference** | Section 5.4.6 |
| **Estimated effort** | 0.25 person-day |

### Specification

Confirm the `RuleBasedClassifier` class operates as documented.

1. The class signature:

   ```python
   class RuleBasedClassifier:
       def __init__(self) -> None: ...
       def classify(self, ear: float, perclos: float, mar: float,
                          pitch: float | None, yaw: float | None,
                   ) -> str: ...
   ```

2. The priority ordering must be `DROWSY ≻ DISTRACTED ≻ ALERT`.

3. The class uses the same thresholds as the SVM hysteresis layer (Module 3.4), sourced from `config.py`.

4. The classifier is selectable via the `--mode rule` CLI flag in `main.py`.

### Acceptance

- `python main.py --mode rule` runs without loading any pickle file.
- The five tests in `tests/test_features.py::TestRuleBasedClassifier` pass.

### Verification command

```bash
python main.py --mode rule  # close after visual verification
```

---

## Module 4.1 — Phase 4 Output: OpenCV Heads-Up Display

| Property | Value |
|---|---|
| **Files** | `modules/display.py`, `main.py` |
| **Status** | Refactor (add FPS counter; verify citation footer) |
| **Depends on** | Modules 2.5, 3.4 |
| **Documentation reference** | Section 5.5.1 |
| **Estimated effort** | 1 person-day |

### Specification

`draw_overlay(frame, state, metrics, fps, latency_ms, mode, proba)` must render on every frame:

1. A state label in state-specific color: `(0, 255, 100)` for ALERT, `(0, 165, 255)` for DROWSY, `(0, 60, 255)` for DISTRACTED. Color is BGR per OpenCV convention.

2. A metric panel listing each of the six feature values formatted to three decimal places, paired with its global threshold:

   ```
   EAR     0.317  thr 0.250
   PERCLOS  8.2%  thr 30.0%
   MAR     0.143  thr 0.600
   Pitch  -1.4°   thr ±20.0°
   Yaw     0.6°   thr ±30.0°
   Roll    0.2°
   ```

3. If `proba` is not `None`, render three horizontal bar segments representing the ALERT, DROWSY, and DISTRACTED probabilities, with the numeric percentage to the right.

4. An FPS counter in the upper-right corner showing the current frames-per-second to one decimal place.

5. A footer block in small font listing the originating literature:

   ```
   EAR Soukupová & Čech 2016 · PERCLOS Wierwille & Ellsworth 1994
   MAR Rahman et al 2015     · PnP Lepetit et al 2009
   SVM-RBF Jabbar et al 2018 · Pipeline Kim et al 2023
   ```

`draw_no_face(frame)` must render a centered "NO FACE DETECTED" message in red.

### Acceptance

- A live webcam test shows the HUD with all six metrics, three probability bars, the FPS counter, and the citation footer.
- Covering the camera switches the display to the "NO FACE DETECTED" message.

### Verification command

```bash
python main.py  # visual verification only
```

---

## Module 4.2 — Phase 4 Output: Audio Alert

| Property | Value |
|---|---|
| **Files** | `assets/alert.wav` (create), `main.py` (integrate), `app.py` (integrate) |
| **Status** | Create |
| **Depends on** | Module 0.1 |
| **Documentation reference** | Section 5.5.4 |
| **Estimated effort** | 0.5 person-day |

### Specification

1. Create `assets/alert.wav` as a one-second 800 Hz tone at 16-bit, 44.1 kHz, mono. A one-time generation script is acceptable:

   ```python
   import numpy as np, wave
   sr = 44100; dur = 1.0; freq = 800
   t = np.linspace(0, dur, int(sr * dur), endpoint=False)
   sig = (np.sin(2 * np.pi * freq * t) * 0.4 * 32767).astype(np.int16)
   with wave.open("assets/alert.wav", "w") as f:
       f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
       f.writeframes(sig.tobytes())
   ```

2. In `main.py`, load the WAV once at startup:

   ```python
   import simpleaudio as sa
   wave_obj = sa.WaveObject.from_wave_file("assets/alert.wav")
   ```

3. Implement a `play_alert(state)` helper with a private cooldown of 2 seconds:

   ```python
   _last_alert_at = 0.0

   def play_alert(state: str) -> None:
       global _last_alert_at
       now = time.time()
       if now - _last_alert_at >= 2.0:
           wave_obj.play()
           _last_alert_at = now
   ```

4. In the main loop, call `play_alert(state)` whenever the current state is `DROWSY` or `DISTRACTED` *and* differs from the previous frame's state.

### Acceptance

- `assets/alert.wav` exists and is approximately 86 KB (1 second × 44.1 kHz × 2 bytes).
- Triggering a DROWSY state during a live test produces an audible tone, not repeated more than once every two seconds.

### Verification command

```bash
python -c "import wave; w = wave.open('assets/alert.wav','r'); print(w.getframerate(), w.getnframes())"
```

---

## Module 4.3 — Phase 4 Output: Standalone CLI

| Property | Value |
|---|---|
| **Files** | `main.py` |
| **Status** | Validate |
| **Depends on** | Modules 4.1, 4.2, 3.4, 3.5 |
| **Documentation reference** | Section 5.5.1 |
| **Estimated effort** | 0.5 person-day |

### Specification

Confirm that `main.py` exposes the documented CLI surface:

```
usage: main.py [-h] [--source SOURCE] [--mode {svm,rule}]
               [--width WIDTH] [--height HEIGHT] [--model MODEL]

options:
  -h, --help          show this help message and exit
  --source SOURCE     camera index (int) or video file path. Default 0.
  --mode {svm,rule}   classifier to use. Default svm.
  --width WIDTH       capture width. Default 640.
  --height HEIGHT     capture height. Default 480.
  --model MODEL       override SVM model path.
```

Confirm:
- Pressing `q` in the OpenCV window exits cleanly.
- The camera is released and the landmarker is closed in a `finally` block.
- The chosen mode prints to standard output at startup.

### Acceptance

- `python main.py --help` prints the documented usage block.
- `python main.py --mode rule` runs without a trained model present.

### Verification command

```bash
python main.py --help
```

---

## Module 4.4 — Phase 4 Output: Flask Backend

| Property | Value |
|---|---|
| **Files** | `app.py` |
| **Status** | Validate / minor extension |
| **Depends on** | Modules 2.x, 3.4, 4.1 |
| **Documentation reference** | Section 5.5.2 |
| **Estimated effort** | 1 person-day |

### Specification

Confirm `app.py` provides:

1. A background camera-capture thread that runs the four-phase pipeline continuously and updates two thread-safe globals: `latest_frame_bytes` (the encoded JPEG) and `latest_metrics` (a dictionary).

2. The `/metrics` endpoint returns JSON with the schema:

   ```json
   {
     "state": "ALERT",
     "ear": 0.317, "perclos": 8.2, "mar": 0.143,
     "pitch": -1.4, "yaw": 0.6, "roll": 0.2,
     "fps": 30, "latency_ms": 12,
     "ear_threshold": 0.25, "mar_threshold": 0.60,
     "perclos_threshold": 30.0,
     "pitch_threshold": 20.0, "yaw_threshold": 30.0,
     "svm_proba": {"ALERT": 0.91, "DROWSY": 0.06, "DISTRACTED": 0.03},
     "ear_consec": 0, "mar_consec": 0, "pose_consec": 0,
     "error_message": null
   }
   ```

3. The `/video_feed` endpoint streams `multipart/x-mixed-replace; boundary=frame` MJPEG with face-mesh dots rendered on the frame but **no textual HUD overlays**. The HUD is reserved for the standalone OpenCV mode; the web frontend renders its own UI.

4. The `/` and `/<path>` routes serve the React build from `frontend/dist/`, with fallback to `index.html` for unmatched paths.

5. On startup, if loading the SVM raises any exception, the server still launches but `/metrics` returns `"state": "ERROR"` and `"error_message": "<reason>"`.

6. The `/metrics` payload includes:
   - `fps` computed as `1000.0 / latency_ms` from a 60-iteration rolling buffer.
   - `latency_ms` measured per loop iteration with `time.perf_counter`.

### Acceptance

- `curl http://localhost:5000/metrics` returns valid JSON with every documented key present.
- `curl -I http://localhost:5000/video_feed` returns `Content-Type: multipart/x-mixed-replace; boundary=frame`.
- Deleting the SVM model file and restarting the server produces `"state": "ERROR"` with a descriptive message rather than a 500.

### Verification command

```bash
# In one terminal:
python app.py
# In another:
curl -s http://localhost:5000/metrics | python -m json.tool
```

---

## Module 5.1 — Frontend: useMetrics Hook

| Property | Value |
|---|---|
| **Files** | `frontend/src/hooks/useMetrics.js` |
| **Status** | Validate |
| **Depends on** | Module 4.4 |
| **Documentation reference** | Section 5.5.3 |
| **Estimated effort** | 0.25 person-day |

### Specification

Confirm the `useMetrics` hook:

- Polls `/metrics` every 300 ms.
- Returns the latest `metrics` object plus an `error` flag if the fetch fails.
- Cleans up the polling interval on component unmount.
- Renders a stale-data indicator if no successful fetch has occurred in the last two seconds.

### Acceptance

- The hook is exported as a default export.
- Stopping the Flask backend while the React app is open causes the UI to indicate stale data within two seconds, without crashing.

### Verification command

```bash
cd frontend && npm run dev
# Open browser to http://localhost:5173 and inspect Network tab for the 300ms polling.
```

---

## Module 5.2 — Frontend: Landing Page

| Property | Value |
|---|---|
| **Files** | `frontend/src/components/Landing.jsx` |
| **Status** | Validate against Mock-Up |
| **Depends on** | Module 0.1 |
| **Documentation reference** | Appendix A.1 |
| **Estimated effort** | 0.5 person-day |

### Specification

Match Mock-Up Figures 1 (light mode) and 2 (dark mode). The component contains:

- ALERTO title in large Space Grotesk weight 700, with a cyan-tinted ALERTO/Baseline badge.
- Tagline: "Adaptive driver monitoring system with real-time biometric tracking and spatial classification."
- Two primary action buttons: "Calibrate System" (which in the baseline routes to the System Dashboard) and "Skip to Dashboard". **Both routes lead to the dashboard; there is no separate calibration screen.** Update the button label if necessary to reflect that calibration is not part of the baseline; "Start Monitoring" and "Driver HUD" are acceptable substitutes.
- Bottom indicator row: "VISION CORE · NEURAL NET".
- Dark mode toggle support.

### Acceptance

- The Landing page matches Mock-Up Figures 1 and 2 to a casual visual inspection.
- Clicking each CTA navigates to the correct destination.

### Verification command

```bash
cd frontend && npm run dev
# Manual inspection.
```

---

## Module 5.3 — Frontend: Driver State View

| Property | Value |
|---|---|
| **Files** | `frontend/src/components/DriverView.jsx` |
| **Status** | Validate against Mock-Up |
| **Depends on** | Module 5.1 |
| **Documentation reference** | Appendix A.2 |
| **Estimated effort** | 0.5 person-day |

### Specification

Match Mock-Up Figures 6, 7, and 8 (Driver State View — Alert, Drowsy, Distracted respectively). The component:

- Reads `state` from `useMetrics`.
- Renders a full-screen, centered display with the appropriate iconography:
  - **Alert**: green shield + check-mark glyph, "Alert", "Drive safely".
  - **Drowsy**: amber closed-eyelid glyph with `Zz` motif, "Drowsy", "Pull over and rest".
  - **Distracted**: red hazard triangle, "Distracted", "Eyes on the road".
- Animates a pulse ring around the central icon in the appropriate state color.
- Displays the current time in the lower portion of the central circle.

### Acceptance

- The three state variants match Mock-Up Figures 6–8 to a casual visual inspection.
- Forcing a state transition via the backend produces the corresponding visual change in under one second.

### Verification command

```bash
cd frontend && npm run dev
# Manual inspection after triggering each state.
```

---

## Module 5.4 — Frontend: System Live Monitoring Dashboard

| Property | Value |
|---|---|
| **Files** | `frontend/src/components/SystemDashboard.jsx` |
| **Status** | Validate against Mock-Up |
| **Depends on** | Module 5.1 |
| **Documentation reference** | Appendix A.3 |
| **Estimated effort** | 1 person-day |

### Specification

Match Mock-Up Figures 8, 9, and 10 (Dashboard — Alert, Drowsy, Distracted). The component:

- Renders a two-column layout. Left column (≈ 60% width): live MJPEG via `<img src="/video_feed" />` with a "Live · Front camera" label and a state badge in the upper-right.
- Right column (≈ 40% width): four stacked metric cards:
  1. **Ocular** — large EAR value, PERCLOS percentage, EAR_baseline reference. Card border turns amber when EAR is within 0.05 of the threshold.
  2. **Oral** — large MAR value, MAR_baseline reference. Card highlights when MAR exceeds the yawn threshold.
  3. **Spatial** — three columns for Pitch, Yaw, Roll. Card highlights when any angle breaches its distraction threshold.
  4. **Edge Performance** — FPS and latency values, with a "Healthy" badge when both are within target.
- The "Deep analytics" button on the upper-right opens the Deep Analytics drawer.
- The bottom strip shows "468 facial landmarks · {fps} FPS · {latency}ms latency".

**Important.** The component must source all threshold values from the `/metrics` payload. No threshold may be hard-coded in JSX. The string `0.25`, `0.60`, `30.0`, etc., must not appear as literals in the JS source.

### Acceptance

- The three dashboard variants match Mock-Up Figures 8–10 to a casual visual inspection.
- `grep -E "0\.25|0\.60|30\.0|20\.0" frontend/src/components/SystemDashboard.jsx` returns no matches in JSX expressions.

### Verification command

```bash
cd frontend && npm run dev
grep -E "0\.25|0\.60" frontend/src/components/SystemDashboard.jsx
```

---

## Module 5.5 — Frontend: Deep Analytics Drawer

| Property | Value |
|---|---|
| **Files** | `frontend/src/components/DeepAnalytics.jsx` |
| **Status** | Validate against Mock-Up |
| **Depends on** | Module 5.1 |
| **Documentation reference** | Appendix A.4 |
| **Estimated effort** | 1 person-day |

### Specification

Match Mock-Up Figures 11, 12, and 13 (Deep Analytics — Alert, Drowsy, Distracted). The drawer:

- Slides in from the right edge of the dashboard with a CSS transition.
- Displays the state badge in the header.
- Renders FPS and latency as sparkline charts over the last 60 readings (use a small custom SVG, no chart library).
- Renders three sub-panels:
  - **Ocular**: PERCLOS progress bar, EAR_LIVE vs. BASELINE comparison.
  - **Oral**: MAR progress bar, yawn count over the last five minutes, BASELINE value.
  - **Spatial**: radar crosshair showing the Pitch/Yaw position with a small dot. NORMAL or BREACH badge.
- Renders an event log with timestamps for the last 20 state transitions.

### Acceptance

- The three variants match Mock-Up Figures 11–13 to a casual visual inspection.
- The drawer can be opened and closed without resetting accumulated state.

### Verification command

```bash
cd frontend && npm run dev
# Manual inspection.
```

---

## Module 5.6 — Frontend: App Shell and Calibration Purge

| Property | Value |
|---|---|
| **Files** | `frontend/src/App.jsx`, plus any residual `Calibration*.jsx` files |
| **Status** | Refactor (purge calibration) |
| **Depends on** | Modules 5.2 through 5.5 |
| **Documentation reference** | Appendix A (opening note) |
| **Estimated effort** | 0.5 person-day |

### Specification

The frontend must contain no calibration UI. Calibration is the ALERTO thesis novelty and is out of scope for the baseline.

1. Search the entire `frontend/src/` tree for: `calibrat`, `baseline locked`, `individualized`, `personalized`, `countdown`. Remove or replace every match.

2. Delete any file named `Calibration*.jsx` if it exists.

3. In `App.jsx`, confirm the tab bar contains only: Home (Landing), Driver, System. There is no Calibration tab.

4. Any route that previously pointed to `/calibration` must be removed.

### Acceptance

- `grep -ri "calibrat" frontend/src` returns no matches (or only matches inside comments that are now correctly negated, e.g., "// calibration screen omitted: belongs to thesis").

### Verification command

```bash
grep -ri "calibrat" frontend/src
```

---

## Module 6.1 — Testing: Unit Test Suite

| Property | Value |
|---|---|
| **Files** | `tests/test_features.py` |
| **Status** | Augment as needed |
| **Depends on** | Modules 2.x, 3.4, 3.5, 4.2 |
| **Documentation reference** | Section 5.9 |
| **Estimated effort** | 0.5 person-day |

### Specification

The existing 24 tests must continue to pass after every refactor. Add tests for any new functions introduced during the modules above, in particular:

1. A test for the `play_alert` cooldown: calling it twice within two seconds should result in only one playback. Use `monkeypatch` on `time.time` and assert on a mocked `wave_obj.play`.

2. A test for the new label-assignment logic in the refactored `extract_features.py` (Module 3.1): given synthetic `(ear, pitch, yaw)` triples and a path string, the labeller returns the expected integer label.

### Acceptance

- `pytest tests/ -v --cov=modules --cov-report=term-missing` shows ≥ 80% coverage on the `modules/` package.
- No test is skipped or marked xfail.

### Verification command

```bash
python -m pytest tests/ -v --cov=modules --cov-report=html
```

---

## Module 6.2 — Testing: Integration Scenarios

| Property | Value |
|---|---|
| **Files** | `outputs/logs/integration_test_log.md` (create) |
| **Status** | Create |
| **Depends on** | All preceding modules |
| **Documentation reference** | Section 5.9 |
| **Estimated effort** | 0.5 person-day |

### Specification

Execute the twelve integration scenarios listed in Implementation Plan Task S6.2. Record each scenario's observed outcome in `outputs/logs/integration_test_log.md` with the following columns: Scenario, Expected, Observed, Pass/Fail, Notes.

### Acceptance

- All twelve scenarios are documented with explicit Pass/Fail markers.
- Any scenario marked Fail has a follow-up issue raised in the project tracker.

### Verification command

```bash
cat outputs/logs/integration_test_log.md
```

---

## Module 7.1 — Documentation: Project README

| Property | Value |
|---|---|
| **Files** | `README.md` |
| **Status** | Already addressed in Module 0.1; revisit at end |
| **Depends on** | All preceding modules |
| **Documentation reference** | — |
| **Estimated effort** | 0.25 person-day |

### Specification

After all preceding modules are complete, revisit the README to incorporate any lessons learned. Confirm that every command in the README still produces the documented outcome on a freshly cloned copy of the repository.

### Acceptance

- A teammate who has not touched the project can clone the repository, follow the README, and reach a working webcam preview within thirty minutes.

### Verification command

Manual.

---

## Module 7.2 — Documentation: Submission Bundle

| Property | Value |
|---|---|
| **Files** | `outputs/submission_bundle/` (create) |
| **Status** | Create |
| **Depends on** | All preceding modules |
| **Documentation reference** | Implementation Plan Task S7.5 |
| **Estimated effort** | 0.5 person-day |

### Specification

Construct the deliverable bundle as specified in Implementation Plan Task S7.5. The bundle excludes `node_modules/`, `venv/`, `dataset/NTHU-DDD/`, `.git/`, and `__pycache__/`.

### Acceptance

- The ZIP file extracts to a runnable repository on a fresh machine, modulo dataset and `face_landmarker.task` download per the included instructions.

### Verification command

```bash
# Extract the ZIP into a temporary location and run python -m pytest from there.
```

---

## Recommended Module Sequence for a Single Build Session

When delivering modules to the agent, recommended ordering:

1. **First session**: 0.1 → 1.1 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
2. **Second session**: 3.1 (the critical refactor) → 3.2 → 3.3 → 3.4 → 3.5
3. **Third session**: 4.1 → 4.2 → 4.3 → 4.4
4. **Fourth session**: 5.1 → 5.2 → 5.3 → 5.4 → 5.5 → 5.6
5. **Fifth session**: 6.1 → 6.2 → 7.1 → 7.2

Each session corresponds to roughly one to two person-days of agent-assisted work. Commit between modules so that any module's diff can be reviewed in isolation.

## Additional Notes for Antigravity Workflow

1. **Hand the agent one module at a time.** Pasting more than one module in a single prompt diffuses the agent's focus and produces compromise solutions. The verification command at the end of each module is your gate to the next.

2. **Anchor every session in the documentation.** Begin each session by pointing the agent at `ALERTO_AI_Project_Documentation.md` and the specific section number referenced in the module being assigned. This grounds the agent's decisions in the immutable behavioral specification.

3. **Reject solutions that introduce new abstractions.** The baseline architecture is intentionally flat. If the agent proposes a "ClassifierFactory" or "FeaturePipeline" abstraction layer that is not in the documentation, reject it. The defensibility of the project depends on the code matching the documentation line for line.

4. **Run the verification command before accepting the diff.** Antigravity will sometimes generate code that lints clean but does not satisfy the documented behavior. The verification command is the authoritative test, not the agent's self-report.

5. **Keep `config.py` as the single source of truth for constants.** If the agent inlines a threshold value anywhere outside `config.py`, that is grounds for a revision request.

6. **Commit messages should reference the module number.** For example: `M3.1: rewrite extract_features.py for NTHU-DDD-only training corpus`. This creates a clean audit trail between the modular specification and the version-control history.

---

*End of modular build specification.*
