# ALERTO Baseline — Implementation and Refactoring Plan

**Document Purpose.** This plan is the operational counterpart to the AI Project Documentation. It audits the present state of the codebase, identifies the structural misalignments between the existing implementation and the documentation, and prescribes a sequenced set of sprints to bring the project to defense-ready status.

**Source of Truth.** The AI Project Documentation produced in the preceding step is the single source of truth for what the system should do. Every sprint below refers back to a specific section of that document and implements or validates the behavior described there.

**Working Assumption.** The current codebase contains roughly seventy percent of the eventual deliverable. However, an estimated twenty percent of the existing implementation is misaligned with the replication framing and must be refactored before any additional work is added; an estimated ten percent of the deliverable is still missing entirely. The plan below addresses these proportions in order.

---

## Part I: Current Status Audit

### 1.1 File-Level Inventory

The following table catalogs every file currently in the project tree, classifies its status against the documentation, and identifies the specific action required.

| File / Module | Status | Required Action |
|---|---|---|
| `config.py` | ✓ Complete | Verify the threshold and landmark-index values are consistent with documentation Section 5.3 and Section 5.4.5. No code change expected. |
| `main.py` | ✓ Complete | Verify CLI flags (`--mode`, `--source`, `--width`, `--height`) match documentation Section 5.5.1. Minor audit only. |
| `app.py` | ✓ Complete | Verify `/metrics` JSON shape matches documentation Section 5.5.2 Table 4. Audit for missing fields. |
| `modules/features.py` | ✓ Complete | Verify EAR/MAR/PERCLOS/PnP implementations match documentation Sections 5.3.1–5.3.3 line by line. |
| `modules/classifier.py` | ✓ Complete | Verify hysteresis thresholds match documentation Section 5.4.5 Table 3. |
| `modules/display.py` | ✓ Complete | Verify HUD content matches documentation Section 5.5.1. |
| `training/extract_features.py` | ⚠ **Requires substantial refactor** | Remove all synthetic Western-data generation. Reconfigure to label NTHU-DDD frames into three classes using head-pose-driven reassignment. See Sprint 0, Task R0.1. |
| `training/train_svm.py` | ⚠ Requires minor refactor | Remove any logging that anchors training metrics to the bias narrative. Add classification report and confusion matrix output to `outputs/`. See Sprint 0, Task R0.3. |
| `evaluation/evaluate_baseline.py` | ⚠ Requires substantial refactor | Remove the Asian-vs-Western comparative analysis (which belongs to the thesis, not the AI Project). Replace with standard hold-out test evaluation reporting overall and per-class metrics. See Sprint 0, Task R0.4. |
| `data/extracted_features.csv` | ⚠ Requires regeneration | Regenerate after `extract_features.py` refactor. Expected sample count: ≈ 2,400 (≈ 800 per class, all from NTHU-DDD). |
| `data/svm_baseline_model.pkl` | ⚠ Requires retraining | Retrain on the regenerated CSV. See Sprint 0, Task R0.3. |
| `data/svm_baseline_scaler.pkl` | ⚠ Requires regeneration | Re-fit the `StandardScaler` on the regenerated training partition. See Sprint 0, Task R0.3. |
| `frontend/src/App.jsx` | ✓ Complete | Verify navigation structure matches documentation Appendix A (no calibration route). |
| `frontend/src/components/Landing.jsx` | ✓ Complete | Verify match with Mock-Up Figures 1–2. |
| `frontend/src/components/SystemDashboard.jsx` | ✓ Complete | Verify match with Mock-Up Figures 8–10. |
| `frontend/src/components/DriverView.jsx` | ✓ Complete | Verify match with Mock-Up Figures 6–7 and the Distracted equivalent. |
| `frontend/src/components/DeepAnalytics.jsx` | ✓ Complete | Verify match with Mock-Up Figures 11–13. |
| `frontend/src/hooks/useMetrics.js` | ✓ Complete | Verify polling interval (300 ms) matches documentation. |
| `tests/test_features.py` | ✓ Complete | Twenty-four tests should still pass after the feature module audit. |
| `requirements.txt` | ⚠ Requires augmentation | Add `simpleaudio` for the audio alert. Pin versions for reproducibility. |
| `README.md` | ⚠ Requires rewrite | Replace with a concise project README consistent with the documentation. |
| `docs/ALERTO_BASELINE_DOCUMENTATION.md` | ⚠ Requires replacement | Replace with the new AI Project Documentation. |
| `dataset/face_landmarker.task` | ✓ Complete | Verify file size (3.6 MB) and integrity. |
| `dataset/NTHU-DDD/` | ✓ Available | No action; used as input for `extract_features.py`. |
| `dataset/YawDD/` | ⚠ **No longer used** | Mark as unused in README; do not delete the directory in case other materials inside are still relevant. |
| `outputs/phase3_ore_evidence.png` | ✗ **Should be deleted** | The Other-Race-Effect bias visualization belongs to the thesis, not the AI Project. |
| `outputs/phase3_ear_distribution.png` | ✗ **Should be deleted** | Same as above. |
| `outputs/phase3_confusion_matrix.png` | ⚠ Regenerate | Keep the concept; regenerate from the clean retrained model as `outputs/confusion_matrix.png`. |
| `outputs/phase3_evaluation_report.txt` | ⚠ Regenerate | Regenerate as `outputs/evaluation_report.txt` with the bias commentary removed. |
| `interface/index.html` (legacy) | ✗ Should be removed | Superseded by React frontend; retain only if explicitly needed for fallback. |
| Audio alert WAV file | ✗ **Missing** | Create or source a short alert tone; place under `assets/alert.wav`. |

> *Table 1.* File-level inventory and required actions.

### 1.2 Critical Misalignments Between Code and Documentation

Three structural misalignments warrant explicit acknowledgement before Sprint 0 begins.

**Misalignment 1: Synthetic Western data injected into training.** The current `training/extract_features.py` generates 999 synthetic samples from published Caucasian-subject distributions and concatenates them with real NTHU-DDD samples before training. The documentation states that the training corpus is sourced exclusively from NTHU-DDD. The synthetic data must be removed not merely because it conflicts with the documentation but because, as a methodological matter, training a system on data that was statistically *invented* to match the parameters of a paper that the project does not have direct access to is not a defensible replication strategy. The synthetic data manufactures a demographic skew that, while convenient for narrating an Other-Race-Effect finding, cannot survive the most basic methodological challenge during the defense. The corrected pipeline trains exclusively on real NTHU-DDD frames; any limitations that emerge from this training corpus are honest empirical observations rather than pre-engineered outcomes.

**Misalignment 2: The Distracted class has no real training data in the current pipeline.** Because NTHU-DDD's `train/` partition is organized into `drowsy/` and `notdrowsy/` subdirectories only, the current code obtains all Distracted samples from the synthetic Western generation block. Removing that block eliminates the Distracted class entirely, which is unacceptable. The corrected pipeline must construct the Distracted class from real NTHU-DDD frames by reassigning the label of any `notdrowsy/` frame whose recovered head pose exceeds the distraction thresholds (|Pitch| > 20° or |Yaw| > 30°). This procedure is internally consistent: it uses the same head-pose threshold at training time as at inference time, and it produces Distracted training samples that come from real Asian-driver imagery rather than synthetic distributions.

**Misalignment 3: Evaluation script reports thesis-level findings within the AI Project deliverable.** The current `evaluation/evaluate_baseline.py` produces a three-panel bar chart, an EAR distribution overlay, and a text report that explicitly compare classification accuracy and false-alarm rates between an Asian subgroup (NTHU-DDD) and a Western subgroup (the synthetic YawDD samples). This is a thesis-level investigation embedded in an AI Project artifact. The corrected evaluation script reports standard hold-out classification metrics only: overall accuracy, per-class precision, per-class recall, per-class F1-score, and the confusion matrix. Any demographic analysis is deferred to the thesis.

### 1.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Removing synthetic Western data reveals that the model has insufficient training signal on its own. | Low | Medium | Class-balanced sampling from NTHU-DDD typically yields 600–1,000 usable frames per class after face-detection filtering. If a class is under-represented after extraction, augment by mirroring or by sampling additional NTHU-DDD subdirectories. |
| Reclassifying `notdrowsy/` frames as Distracted by head-pose threshold produces a class boundary that the classifier cannot learn because the same features that define the label also define the input. | Medium | Medium | The boundary is deliberately analytic, not learned. The classifier will rapidly memorize the threshold, which is fine: it provides the *fusion* between facial geometry and pose, and the hysteresis layer supplies the temporal robustness that a hard threshold cannot. |
| The retrained model produces lower headline accuracy than the previous (synthetic-padded) model. | High | Low | Lower headline accuracy on honest data is acceptable and arguably preferable; it can be defended as the natural performance of a single-corpus baseline. The defense narrative does not require a particular accuracy floor. |
| Sprint 0 takes longer than budgeted and pushes Sprint 7 (defense prep) into the deadline. | Medium | High | Sprint 0 is sequenced first precisely so that any over-run is detected early and absorbed into Sprints 1–5, which contain optional polish items that can be deferred. |
| The audio alert WAV file is missing and cannot be sourced quickly. | Low | Low | Use a programmatically generated tone via `numpy` and `simpleaudio` at runtime as a fallback. |

> *Table 2.* Risk register.

### 1.4 Sprint Schedule at a Glance

The plan is structured as eight sprints. Effort is expressed in person-days at one developer; with three members working in parallel on independent tasks, calendar time will be shorter.

| Sprint | Name | Effort (person-days) | Documentation Section | Calendar Priority |
|---|---|---|---|---|
| 0 | Refactoring and Cleanup | 3 | All | **Highest — must complete first** |
| 1 | Phase 1 Validation: Acquisition and Landmark Extraction | 1 | 5.2 | High |
| 2 | Phase 2 Validation: Geometric Feature Computation | 2 | 5.3 | High |
| 3 | Phase 3 Completion: Fusion and Classification | 3 | 5.4 | High |
| 4 | Phase 4 Completion: Output and Visualization | 3 | 5.5 | High |
| 5 | Frontend Polish and Alignment | 3 | 5.5.3 and Appendix A | Medium |
| 6 | Integration Testing and Performance Validation | 3 | 5.9 | High |
| 7 | Documentation Polish and Defense Preparation | 3 | — | High |
| | **Total** | **21 person-days** | | |

> *Table 3.* Sprint schedule overview.

---

## Part II: Sprint 0 — Refactoring and Cleanup

**Goal.** Bring the existing codebase into structural alignment with the AI Project Documentation before any further functionality is added. Every other sprint depends on Sprint 0 having completed cleanly.

**Sprint 0 Exit Criteria.** (a) No synthetic data appears anywhere in the training pipeline. (b) A regenerated `data/extracted_features.csv` exists containing ≈ 2,400 samples sourced exclusively from NTHU-DDD, stratified across three labels. (c) A retrained `data/svm_baseline_model.pkl` and `data/svm_baseline_scaler.pkl` exist. (d) The evaluation script reports standard classification metrics only. (e) `requirements.txt` includes every runtime dependency. (f) All 24 existing unit tests still pass.

### Task R0.1 — Refactor `training/extract_features.py`

**Owner.** Bacolor (lead developer of the training pipeline).

**Effort.** 1 person-day.

**File.** `training/extract_features.py`.

**Required deletions.**

1. Delete every line associated with the synthetic Western-data generation block, including all `np.random.normal(...)` calls used to fabricate ALERT, DROWSY, and DISTRACTED Western samples.
2. Delete the `dataset` column from the CSV schema; the regenerated CSV does not need to track sub-corpus provenance because there is only one corpus.
3. Delete any console output that references "YawDD" or "Western" samples.

**Required additions and modifications.**

1. Iterate over the NTHU-DDD `Multi class/train/notdrowsy/**/*.jpg` and `Multi class/train/drowsy/**/*.jpg` directories using `glob.glob` with `recursive=True`.
2. For each image, initialize MediaPipe FaceLandmarker in `RunningMode.IMAGE`, run inference, and obtain the landmark list. Skip the frame if no face is detected.
3. For each detected face, compute the binocular average EAR, the MAR, and the three head-pose Euler angles using the functions in `modules/features.py`. For PERCLOS at training time, retain the per-frame approximation `100.0 if EAR < 0.25 else 0.0` (documented in Section 5.8 of the documentation as the static-image approximation).
4. Assign the label according to the following rules, in the order shown:
   - If the source directory is `drowsy/`, assign label 1 (DROWSY).
   - Otherwise (source is `notdrowsy/`):
     - If `abs(pitch) > 20.0` or `abs(yaw) > 30.0`, assign label 2 (DISTRACTED).
     - Otherwise, assign label 0 (ALERT).
5. Maintain three per-class counters. Once any class reaches 800 samples, skip further frames from that class to keep the corpus balanced.
6. After iteration completes, write `data/extracted_features.csv` with columns exactly equal to `["ear_avg", "perclos_pct", "mar", "pitch_deg", "yaw_deg", "roll_deg", "label"]`. The leading index column is suppressed (`index=False`).
7. Print the final class distribution to standard output.

**Acceptance criteria.**

- The refactored script runs to completion without warning.
- The resulting CSV contains between 2,000 and 2,500 rows.
- No row contains a `NaN` value.
- The class distribution is approximately balanced, with no class falling below 600 samples or exceeding 900.

### Task R0.2 — Remove `dataset/YawDD/` from the Training Path

**Owner.** Bacolor.

**Effort.** 0.25 person-day.

**Required actions.**

1. Confirm that no code anywhere in the project references the `dataset/YawDD/` directory after Task R0.1.
2. Add a note to `dataset/README.md` (create the file if absent) stating that the YawDD subdirectory is no longer used by the baseline pipeline.
3. Do *not* delete the directory; preserve it in place so that any thesis-stage experiments that revisit Western-driver data can resume from existing assets.

**Acceptance criteria.**

- `grep -r "YawDD" --include="*.py" .` returns no matches.

### Task R0.3 — Retrain the SVM on the Cleaned Corpus

**Owner.** Bacolor.

**Effort.** 0.5 person-day.

**File.** `training/train_svm.py`.

**Required deletions.**

1. Remove any `print` statements or comments that reference Western or Asian sub-populations.
2. Remove any code that splits the dataset into demographic subsets before evaluation.

**Required additions and modifications.**

1. Confirm that the script loads `data/extracted_features.csv` and reads exactly the seven columns produced by R0.1.
2. Perform a stratified 70/30 train-test split using `sklearn.model_selection.train_test_split(stratify=y, random_state=42)`.
3. Fit a `StandardScaler` on the training partition only. Persist to `data/svm_baseline_scaler.pkl` via `pickle.dump`.
4. Train an `SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)`. Persist to `data/svm_baseline_model.pkl`.
5. Execute `sklearn.model_selection.cross_val_score` with 5 folds on the scaled training partition; print the mean and standard deviation of the cross-validation accuracy.
6. Evaluate on the held-out test partition and print `sklearn.metrics.classification_report(y_true, y_pred, target_names=["ALERT", "DROWSY", "DISTRACTED"])`.
7. Save a confusion matrix figure to `outputs/confusion_matrix.png` using `sklearn.metrics.ConfusionMatrixDisplay`.
8. Save the textual classification report to `outputs/training_report.txt`.

**Acceptance criteria.**

- The script runs to completion.
- The 5-fold cross-validation accuracy is reported.
- `outputs/confusion_matrix.png` and `outputs/training_report.txt` are produced.
- The trained model and scaler pickles are written successfully.

### Task R0.4 — Simplify `evaluation/evaluate_baseline.py`

**Owner.** Caole.

**Effort.** 0.5 person-day.

**File.** `evaluation/evaluate_baseline.py`.

**Required deletions.**

1. Delete the three-panel ORE-evidence bar chart generation.
2. Delete the EAR distribution histogram overlay.
3. Delete the Asian-versus-Western comparative text report.

**Required additions and modifications.**

1. Replace the deleted blocks with a single straightforward evaluation routine that loads the test partition (`data/extracted_features.csv` partitioned with the same random seed as `train_svm.py`), loads the trained model and scaler, applies the scaler, and produces:
   - Overall accuracy.
   - Per-class precision, recall, and F1-score.
   - The three-by-three confusion matrix (numerical and as a PNG figure).
   - A short prose summary appended to `outputs/evaluation_report.txt`.

**Acceptance criteria.**

- The script runs to completion.
- The output report contains no language about racial subgroups.
- The reported metrics are consistent with those reported by `train_svm.py` on the same held-out partition.

### Task R0.5 — Augment `requirements.txt`

**Owner.** Caole.

**Effort.** 0.25 person-day.

**File.** `requirements.txt`.

**Required modifications.**

1. Add the line `simpleaudio>=1.0.4` for audio alert playback in standalone mode.
2. Pin every existing dependency to the version range `>= <observed>, < <observed_major+1>` so that an unexpected library update cannot silently break the project.
3. Verify that `pytest` and `pytest-cov` are still present; verify that `flask` and `flask-cors` are pinned.

**Acceptance criteria.**

- A fresh `pip install -r requirements.txt` in a clean virtual environment produces no errors.
- `pip list` after install shows all dependencies present.

### Task R0.6 — Clean the `outputs/` Directory

**Owner.** Soriano.

**Effort.** 0.25 person-day.

**Required actions.**

1. Delete `outputs/phase3_ore_evidence.png` and `outputs/phase3_ear_distribution.png`.
2. Rename `outputs/phase3_confusion_matrix.png` to `outputs/confusion_matrix.png` (or simply allow R0.3 to overwrite it).
3. Rename `outputs/phase3_evaluation_report.txt` to `outputs/training_report.txt`.
4. Ensure the `outputs/logs/` and `outputs/screenshots/` subdirectories exist for later use by Sprint 6 and Sprint 7.

**Acceptance criteria.**

- No file in `outputs/` has the `phase3_` prefix.
- No file in `outputs/` references the Other-Race Effect.

### Task R0.7 — Rewrite `README.md`

**Owner.** Soriano.

**Effort.** 0.5 person-day.

**File.** `README.md` (project root).

**Required content (high level).**

1. Project identity (title, members, course, adviser, institution, academic year).
2. One-paragraph project summary copied or adapted from Section I of the documentation.
3. Prerequisites (Python 3.12+, Node 20+, NTHU-DDD download, `face_landmarker.task` location).
4. Installation steps for the Python virtual environment and the Node frontend.
5. Training-pipeline invocation (`py training/extract_features.py` then `py training/train_svm.py`).
6. Standalone-mode invocation (`py main.py` with CLI flags).
7. Web-mode invocation (`py app.py` in one terminal, `npm run dev` in another).
8. Test invocation (`py -m pytest tests/ -v`).
9. Citation block listing Kim et al. (2023) and Jabbar et al. (2018) as the source frameworks.
10. License and contact information.

**Acceptance criteria.**

- The README contains no reference to the Other-Race Effect, to bias engineering, or to thesis-level findings.
- A fresh reader can clone the repository, follow the README, and reach a working webcam preview within thirty minutes.

---

## Part III: Sprint 1 — Phase 1 Validation: Acquisition and Landmark Extraction

**Goal.** Confirm that the existing implementation of Phase 1 (documentation Section 5.2) is correct, robust, and consistent with the Tasks API contract.

**Owner.** Bacolor.

**Effort.** 1 person-day.

**Tasks.**

1. Open `modules/features.py` and confirm that the `_px` helper correctly converts normalized landmark coordinates to integer pixel coordinates by multiplication with frame dimensions.
2. In `main.py` and `app.py`, confirm that the MediaPipe FaceLandmarker is initialized through `mediapipe.tasks.python.vision.FaceLandmarker.create_from_options` (the Tasks API) and not through `mediapipe.solutions.face_mesh.FaceMesh` (the legacy API). If the legacy API is still in use anywhere, replace it with the Tasks API.
3. Confirm that the running mode is set to `mediapipe.tasks.python.vision.RunningMode.VIDEO` in both `main.py` and `app.py`.
4. Confirm that `detect_for_video` is invoked with a monotonically increasing integer millisecond timestamp derived from `time.time() * 1000`.
5. Add a `try`/`except` guard around the landmarker creation that produces an actionable error message if `dataset/face_landmarker.task` is missing.
6. Verify that the BGR-to-RGB conversion is applied to every frame before it is passed to the landmarker. The MediaPipe Tasks API requires RGB input; passing BGR silently degrades landmark accuracy.
7. Run `main.py` with a live webcam for thirty seconds and visually confirm that the face mesh dots overlay the face correctly under three lighting conditions: daylight near a window, ordinary indoor lighting, and a darker setting at around 100 lux.

**Acceptance criteria.**

- No reference to `mp.solutions` remains in the runtime path.
- The system displays the face mesh on a live webcam.
- A friendly error message appears if `face_landmarker.task` is absent.

---

## Part IV: Sprint 2 — Phase 2 Validation: Geometric Feature Computation

**Goal.** Confirm that the existing implementation of Phase 2 (documentation Section 5.3) is mathematically correct against the published formulas and that the landmark indices match the documentation exactly.

**Owner.** Bacolor.

**Effort.** 2 person-days.

### Task S2.1 — EAR Audit

1. Open `modules/features.py` and confirm that the `compute_ear` function implements the formula
$$EAR = \frac{\lVert P_2 - P_6 \rVert + \lVert P_3 - P_5 \rVert}{2 \cdot \lVert P_1 - P_4 \rVert}.$$
2. Confirm that `RIGHT_EYE_IDX` and `LEFT_EYE_IDX` in `config.py` contain exactly the index sets specified in documentation Section 5.3.1:
   - Right: {P1=33, P2=160, P3=158, P4=133, P5=153, P6=144}
   - Left:  {P1=362, P2=385, P3=387, P4=263, P5=373, P6=380}
3. Confirm that `compute_avg_ear` returns the arithmetic mean of the two per-eye values.
4. Run `py -m pytest tests/test_features.py::TestComputeEAR -v` and confirm that all four EAR tests pass.

### Task S2.2 — PERCLOS Audit

1. Confirm that `PERCLOSBuffer` is initialized with `window_size=60` and `threshold=0.25`.
2. Confirm that the internal storage is a `collections.deque(maxlen=60)`, providing O(1) amortized insertion and automatic eviction.
3. Confirm that `update` appends `1` if EAR is below threshold and `0` otherwise.
4. Confirm that the returned percentage is `(sum / len) * 100.0`.
5. Run `py -m pytest tests/test_features.py::TestPERCLOSBuffer -v` and confirm that all five tests pass.

### Task S2.3 — MAR Audit

1. Confirm that `compute_mar` implements the formula
$$MAR = \frac{v_1 + v_2 + v_3}{2 \cdot h}.$$
2. Confirm that `MOUTH_IDX` in `config.py` contains exactly the index set specified in documentation Section 5.3.2:
   - {upper_center=13, lower_center=14, upper_left=82, lower_left=87, upper_right=312, lower_right=317, left_corner=78, right_corner=308}
3. Run `py -m pytest tests/test_features.py::TestComputeMAR -v` and confirm that all three tests pass.

### Task S2.4 — Head Pose Audit

1. Confirm that `compute_head_pose` constructs the camera intrinsic matrix `K` with focal length equal to frame width, principal point at the frame center, and zero distortion.
2. Confirm that `FACE_3D_MODEL` in `config.py` matches the six anchor coordinates listed in documentation Section 5.3.3.
3. Confirm that `HEAD_POSE_ANCHOR_LM` in `config.py` is `[1, 152, 263, 33, 61, 291]`.
4. Confirm that the function uses `cv2.solvePnP(..., flags=cv2.SOLVEPNP_ITERATIVE)`, followed by `cv2.Rodrigues`, followed by `cv2.RQDecomp3x3`.
5. Confirm that the function returns `(pitch, yaw, roll)` as Python floats, or `(None, None, None)` if the PnP solver fails.
6. Conduct a manual sanity check: hold the head straight and observe values near (0, 0, 0); tilt down and observe positive Pitch; turn left and observe positive Yaw. If the signs are inverted, document the OpenCV convention in a code comment but do not modify the function.

### Task S2.5 — Feature Vector Assembly Audit

1. Confirm that `extract_feature_vector(ear, perclos, mar, pitch, yaw, roll)` returns a NumPy array of shape `(6,)` with `dtype=float64` and elements in the order `[ear, perclos, mar, pitch, yaw, roll]`.
2. Confirm that any `None` value for pitch, yaw, or roll is coerced to `0.0` before vector construction.
3. Run `py -m pytest tests/test_features.py::TestExtractFeatureVector -v` and confirm that all three tests pass.

**Sprint 2 Acceptance Criteria.**

- All 15 feature-related unit tests pass.
- A live webcam test displays plausible numerical values for each metric: EAR roughly 0.30 at rest, MAR roughly 0.15 at rest, PERCLOS at 0 when alert, and head-pose angles within ±5° when facing forward.

---

## Part V: Sprint 3 — Phase 3 Completion: Fusion and Classification

**Goal.** Confirm and where necessary correct the implementation of the Support Vector Machine classifier and the hysteresis layer (documentation Section 5.4).

**Owner.** Bacolor with support from Soriano.

**Effort.** 3 person-days.

**Prerequisite.** Sprint 0 must have completed; `data/svm_baseline_model.pkl` and `data/svm_baseline_scaler.pkl` must exist and have been generated from the cleaned corpus.

### Task S3.1 — SVMClassifier Validation

1. Open `modules/classifier.py` and confirm that `SVMClassifier.load` reads from the paths defined in `config.py` (`SVM_MODEL_PATH`, `SVM_SCALER_PATH`) and raises `FileNotFoundError` with an actionable message if either file is absent.
2. Confirm that `predict` calls `self.scaler.transform(feature_vec.reshape(1, -1))` *before* calling `self.model.predict`. Skipping the scaler transform is a common silent bug that produces wildly incorrect predictions; verify by inspection.
3. Confirm that `predict` returns a tuple `(state: str, proba: dict)` where `proba` maps each of the three state names to a float probability.
4. Add a guard: if `self.model.predict_proba` raises `AttributeError` (because the model was trained without `probability=True`), return `proba=None` instead of crashing.

### Task S3.2 — Hysteresis Layer Validation

1. Confirm that `_apply_hysteresis` increments `_ear_cnt` when `ear < EAR_CLOSED_THRESHOLD` and resets it otherwise.
2. Confirm the same pattern for `_mar_cnt` and `_pose_cnt`.
3. Confirm that the threshold values used by the hysteresis layer match those in `config.py` (and therefore those in documentation Section 5.4.5, Table 3):
   - `EAR_CONSECUTIVE_FRAMES = 50`
   - `MAR_CONSECUTIVE_FRAMES = 30`
   - `DISTRACTION_FRAMES = 45`
4. Confirm the suppression logic: a raw DROWSY prediction is downgraded to ALERT unless either `_ear_cnt >= 50` or `_mar_cnt >= 30`. A raw DISTRACTED prediction is downgraded to ALERT unless `_pose_cnt >= 45`.
5. Run `py -m pytest tests/test_features.py::TestSVMClassifier tests/test_features.py::TestRuleBasedClassifier -v` and confirm that all eight classifier tests pass.

### Task S3.3 — Rule-Based Fallback Validation

1. Confirm that `RuleBasedClassifier.classify` uses the same threshold values as the hysteresis layer.
2. Confirm the priority ordering: a DROWSY-triggering condition wins over a DISTRACTED-triggering condition.
3. Confirm that the fallback is selectable via the `--mode rule` CLI flag in `main.py`.

### Task S3.4 — Stress Test the Full Phase 3

Construct a simple integration test (not a unit test, run manually) that exercises the classifier end-to-end:

1. Cover the camera with a hand for two seconds and confirm that the system either remains in ALERT (no face detected) or rapidly enters DROWSY when uncovered with the eyes closed. Reopen eyes and confirm return to ALERT within 60 frames.
2. Yawn deliberately for one second and confirm a transition to DROWSY driven by MAR rather than EAR.
3. Turn the head ninety degrees to either side and confirm a transition to DISTRACTED within 45 frames.
4. Return to neutral and confirm a transition back to ALERT.

**Sprint 3 Acceptance Criteria.**

- All classifier unit tests pass.
- The four-step manual integration test above produces the expected state transitions.
- The standard output of `main.py` shows the three SVM probabilities updating in real time.

---

## Part VI: Sprint 4 — Phase 4 Completion: Output and Visualization

**Goal.** Confirm and where necessary correct the implementation of the dual output paths: the standalone OpenCV HUD and the Flask backend (documentation Section 5.5).

**Owner.** Caole.

**Effort.** 3 person-days.

### Task S4.1 — OpenCV Heads-Up Display Audit

1. Open `modules/display.py` and confirm that `draw_overlay` renders, on every frame:
   - The current state label in the appropriate state-specific color: green for ALERT, amber `(0, 165, 255)` for DROWSY, red `(0, 60, 255)` for DISTRACTED.
   - The six metric values from the feature vector, each labeled and formatted to three decimal places.
   - The three SVM probabilities as horizontal bar segments.
   - A footer block listing the originating literature for the formulas (Soukupová & Čech, 2016 for EAR; Wierwille & Ellsworth, 1994 for PERCLOS; Rahman et al., 2015 for MAR; Lepetit et al., 2009 for PnP).
2. Confirm that `draw_no_face` renders a centered "NO FACE DETECTED" message when the landmarker returns no faces.
3. Add a small frames-per-second counter to the upper-right corner of the HUD.

### Task S4.2 — Audio Alert Implementation

**Required because the current pipeline declares audio alerts but does not implement them.**

1. Source or generate a one-second WAV file at 16-bit, 44.1 kHz containing a distinct alert tone. Place it at `assets/alert.wav`. If sourcing externally proves difficult, generate it programmatically:
   ```python
   import numpy as np, wave
   sr = 44100; duration = 1.0; freq = 800
   t = np.linspace(0, duration, int(sr * duration), endpoint=False)
   wave_data = (np.sin(2 * np.pi * freq * t) * 0.4 * 32767).astype(np.int16)
   with wave.open("assets/alert.wav", "w") as f:
       f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
       f.writeframes(wave_data.tobytes())
   ```
2. In `main.py`, import `simpleaudio` and load the WAV once at startup via `wave_obj = simpleaudio.WaveObject.from_wave_file("assets/alert.wav")`.
3. Implement a `play_alert()` function that calls `wave_obj.play()` non-blocking, with a private cooldown of two seconds between playbacks to avoid alert spam.
4. Trigger `play_alert()` on every state transition from ALERT to either DROWSY or DISTRACTED.

### Task S4.3 — Flask Backend Audit

1. Open `app.py` and confirm that the `/metrics` endpoint returns a JSON object with at least the following keys, matching documentation Section 5.5.2:
   - `state` (string)
   - `ear`, `perclos`, `mar`, `pitch`, `yaw`, `roll` (floats)
   - `fps`, `latency_ms` (numbers)
   - `ear_threshold`, `mar_threshold`, `perclos_threshold` (floats from `config.py`)
   - `svm_proba` (object with keys `ALERT`, `DROWSY`, `DISTRACTED`)
   - `ear_consec`, `mar_consec`, `pose_consec` (integers, the hysteresis counters)
2. Confirm that the `/video_feed` endpoint streams Motion-JPEG without textual overlays. The text-rich HUD is reserved for the standalone OpenCV mode; the web stream should display only the face-mesh dots so that the React UI is free to render its own annotations.
3. Confirm that the camera-capture background thread starts when the Flask app starts and joins cleanly on shutdown.
4. Add error handling: if `SVMClassifier.load` fails on startup, the Flask app should still launch but `/metrics` should return `"state": "ERROR"` and `"error_message": "<reason>"` so the frontend can display a helpful message.

### Task S4.4 — Latency Logging

1. In the camera-capture loop of `app.py`, measure the wall-clock duration of each iteration (Phase 1 through Phase 3) using `time.perf_counter`.
2. Maintain a rolling buffer of the last 60 measurements and expose the mean as `latency_ms` in the `/metrics` payload.
3. Compute `fps` as `1000.0 / latency_ms` and expose it in the same payload.

**Sprint 4 Acceptance Criteria.**

- The standalone HUD renders correctly on a live webcam, with the FPS counter visible.
- An audio alert plays when DROWSY or DISTRACTED is first entered.
- The Flask `/metrics` endpoint returns the full JSON schema.
- The Flask `/video_feed` endpoint streams without textual annotations.

---

## Part VII: Sprint 5 — Frontend Polish and Alignment

**Goal.** Bring the React frontend into visual and behavioral alignment with the Mock-Up appendix images and remove any residual calibration references that may exist from earlier iterations.

**Owner.** Caole with support from Soriano.

**Effort.** 3 person-days.

### Task S5.1 — Mock-Up Visual Alignment

For each of the four React components, compare against the corresponding Mock-Up figure and reconcile any visual discrepancies:

| Component | Reference Figure | Action Items |
|---|---|---|
| `Landing.jsx` | Mock-Up Figures 1–2 | Confirm the title typography, the tagline text, the two CTA buttons, and the bottom indicator row. |
| `DriverView.jsx` Alert | Mock-Up Figure 6 | Confirm the green shield iconography, the "Drive safely" directive, and the pulse-ring animation. |
| `DriverView.jsx` Drowsy | Mock-Up Figure 7 | Confirm the amber eyelid iconography and the "Pull over and rest" directive. |
| `DriverView.jsx` Distracted | Mock-Up Figure 8 | Confirm the red hazard triangle and the "Eyes on the road" directive. |
| `SystemDashboard.jsx` Alert | Mock-Up Figure 8 (Dashboard variant) | Confirm the four metric cards (Ocular, Oral, Spatial, Edge Performance) and the live camera feed on the left. |
| `SystemDashboard.jsx` Drowsy | Mock-Up Figure 9 | Confirm the elevated PERCLOS and MAR readings, the "Yawning" subtag on the Oral card, and the unchanged Spatial card. |
| `SystemDashboard.jsx` Distracted | Mock-Up Figure 10 | Confirm the off-axis Spatial card and the unchanged Ocular and Oral cards. |
| `DeepAnalytics.jsx` Alert | Mock-Up Figure 11 | Confirm the FPS and Latency sparklines, the three sub-panels, and the event log. |
| `DeepAnalytics.jsx` Drowsy | Mock-Up Figure 12 | Confirm the highlighted Ocular and Oral sub-panels. |
| `DeepAnalytics.jsx` Distracted | Mock-Up Figure 13 | Confirm the highlighted Spatial sub-panel with the BREACH badge. |

### Task S5.2 — Remove Any Calibration References

1. Search the entire `frontend/src/` directory for the strings "calibrat", "baseline locked", "individualized", and "personalized".
2. Remove any UI element, route, comment, or string that implements or references calibration. The AI Project does not include calibration; that mechanism belongs to the thesis.
3. If a `Calibration.jsx` component file exists, delete it.
4. In `App.jsx`, confirm that the navigation tab bar contains only Landing, Driver, and System (no Calibration tab).

### Task S5.3 — Threshold Hard-Code Audit

1. Search the entire `frontend/src/` directory for hard-coded threshold numbers: `0.25`, `0.60`, `30.0`, `20.0`, and `45`.
2. For every match, confirm that the value is sourced from the `/metrics` JSON payload at runtime, not hard-coded in the JavaScript.
3. The principle is that `config.py` is the single source of truth for all threshold values; the frontend reads them through `/metrics` rather than maintaining a parallel set of constants.

### Task S5.4 — Build and Production Verification

1. From the `frontend/` directory, run `npm run build` and confirm that `frontend/dist/` is produced without errors.
2. With the React dev server stopped, run `py app.py` and confirm that opening `http://localhost:5000/` in a browser serves the compiled React SPA.
3. Test all three navigation transitions (Landing → System Dashboard, Landing → Driver View, System Dashboard → Deep Analytics).
4. Confirm that the live video feed renders inside the dashboard's left panel.

**Sprint 5 Acceptance Criteria.**

- The four components match the Mock-Up images to a casual visual inspection.
- No calibration UI remains anywhere in the frontend.
- All threshold values displayed in the UI originate from the `/metrics` endpoint.
- The production build serves correctly through the Flask backend.

---

## Part VIII: Sprint 6 — Integration Testing and Performance Validation

**Goal.** Confirm end-to-end correctness of the full pipeline and validate the documented 30-FPS throughput target.

**Owner.** Soriano with support from all members.

**Effort.** 3 person-days.

### Task S6.1 — Unit Test Suite

1. Run `py -m pytest tests/ -v --cov=modules --cov-report=html`.
2. Confirm that all 24 existing tests pass.
3. Confirm that overall coverage of the `modules/` package exceeds 80%.
4. If any new functions were added during Sprints 1–5 (for example, the `play_alert` function in Task S4.2), add corresponding unit tests.

### Task S6.2 — End-to-End Integration Test Scenarios

Conduct each of the following scenarios on a live system, with one member performing the action and another verifying the system response. Record observations in `outputs/logs/integration_test_log.md`.

| Scenario | Expected Result |
|---|---|
| Cold start the standalone mode with `py main.py`. | The camera window opens within five seconds. Face mesh dots overlay the face. State is ALERT. |
| Close the eyes for two seconds. | State transitions to DROWSY. Audio alert plays. |
| Open the eyes and remain still. | State returns to ALERT within sixty frames (≈ two seconds). |
| Yawn deliberately and hold the open mouth for one second. | State transitions to DROWSY driven by MAR. |
| Turn the head to the right at approximately 45°. | State transitions to DISTRACTED within forty-five frames. |
| Return the head to neutral. | State returns to ALERT. |
| Cover the camera completely. | "NO FACE DETECTED" overlay appears. |
| Uncover the camera. | Face mesh resumes and state returns to ALERT. |
| Cold start the web mode with `py app.py` and `npm run dev`. | The React app loads in the browser. The video feed shows the face mesh. The metrics panel updates every 300 ms. |
| In web mode, repeat the close-eyes scenario. | The Driver State View transitions to DROWSY with the correct visual treatment. |
| In web mode, open the Deep Analytics drawer. | All sub-panels render correctly. The event log records the recent state transitions with timestamps. |
| Kill the Flask process with Ctrl-C. | The process exits cleanly. The camera is released. |

### Task S6.3 — Performance Benchmark

1. Run `main.py` for sixty seconds on the target hardware.
2. From the FPS counter in the upper-right corner of the HUD, record the average, minimum, and maximum frames per second over that interval.
3. Confirm that the average frames per second is at least 30. If not, profile the loop using `cProfile`:
   ```
   py -m cProfile -o outputs/profile.prof main.py
   ```
   and inspect the result with `snakeviz outputs/profile.prof` to identify the bottleneck.
4. Common bottlenecks to investigate if 30 FPS is not met:
   - Frame resolution above 640 by 480 (reduce in `main.py`).
   - The MediaPipe model being loaded in float32 (confirm float16 model is present at `dataset/face_landmarker.task`).
   - The `predict_proba` call on every frame (consider caching, since the alert decision uses the discrete prediction).
5. Record the final benchmark in `outputs/performance_benchmark.md`.

### Task S6.4 — Cross-Platform Smoke Test

1. If the team has access to both Windows and a Unix-based system, conduct a smoke test on at least one machine of each.
2. Document any platform-specific issues encountered (for example, `simpleaudio` requiring a system-level audio dependency on Linux).
3. Add platform-specific installation notes to `README.md` if needed.

**Sprint 6 Acceptance Criteria.**

- All 24 unit tests pass.
- All 12 integration scenarios produce the expected results.
- The system sustains ≥ 30 FPS on the target hardware.
- `outputs/performance_benchmark.md` documents the measured performance.

---

## Part IX: Sprint 7 — Documentation Polish and Defense Preparation

**Goal.** Finalize all deliverables for submission and prepare for the project defense.

**Owner.** Soriano with full team participation.

**Effort.** 3 person-days.

### Task S7.1 — Final Report Polish

1. Transfer the contents of `ALERTO_AI_Project_Documentation.md` into the PUP CCIS Word template, preserving the section headings, table numbering, and figure numbering.
2. Insert the five required diagrams at the locations marked with `[Insert ... ]`:
   - Figure 1: Adapt the Kim et al. (2023) architecture diagram from the original draft.
   - Figure 2: Create a custom four-phase pipeline diagram. Recommended tool: Draw.io or Figma. Match the visual style of the rest of the document.
   - Figure 3: EAR landmark diagram. Recommended approach: take a screenshot of a live face-mesh frame and annotate the six right-eye landmarks with their indices.
   - Figure 4: MAR landmark diagram. Same approach as Figure 3 for the eight mouth landmarks.
   - Figure 5: Six-point generic 3D face model. Either reproduce from Lepetit et al. (2009) or create a simple labeled diagram.
3. Insert the eleven Mock-Up screenshots into Appendix A at the locations marked.
4. Run a final proofreading pass with all three members for typographical errors, citation consistency, and figure-number sequencing.

### Task S7.2 — Demo Recording

1. Record a five-minute demonstration video that walks through the system end-to-end:
   - 0:00–0:30 — Project introduction and title card.
   - 0:30–1:30 — Standalone mode demonstration. Show the OpenCV HUD with all three state transitions (Alert → Drowsy → Alert → Distracted → Alert).
   - 1:30–3:30 — Web mode demonstration. Show the Landing Page, navigate to the Driver State View, demonstrate the three states, navigate to the System Live Monitoring Dashboard, open the Deep Analytics Panel.
   - 3:30–4:30 — Training pipeline walkthrough. Show the CSV output, the classification report, and the confusion matrix.
   - 4:30–5:00 — Closing remarks and acknowledgement of the ALERTO thesis as the planned successor.
2. Use a screen recorder such as OBS Studio.
3. Add a brief voice-over or onscreen captions.
4. Save the final video to `outputs/demo_video.mp4`.

### Task S7.3 — Defense Slide Deck

Construct a slide deck of approximately fifteen slides:

| Slide | Content |
|---|---|
| 1 | Title slide with project name, members, course, adviser, date. |
| 2 | Outline. |
| 3 | Problem context: driver fatigue statistics. |
| 4 | Why facial-landmark analysis is the right approach (rejection of EEG and vehicle-dynamics methods). |
| 5 | The reference architecture: Kim et al. (2023) + Jabbar et al. (2018). |
| 6 | System architecture diagram (Figure 2 from the final report). |
| 7 | Phase 1 — Acquisition and Landmarking. |
| 8 | Phase 2A — EAR and PERCLOS. Display the formulas. |
| 9 | Phase 2B — MAR. Display the formula. |
| 10 | Phase 2C — Head Pose via PnP. |
| 11 | Phase 3 — SVM classification with hysteresis. |
| 12 | Phase 4 — Dual output (OpenCV HUD + web dashboard). |
| 13 | Training results: cross-validation accuracy, classification report, confusion matrix. |
| 14 | Live demonstration (or embedded video). |
| 15 | Scope, limitations, and the planned ALERTO thesis as next steps. |

### Task S7.4 — Question-and-Answer Preparation

Prepare written answers to the following questions, which the panel is likely to ask:

1. Why a Support Vector Machine rather than a deep neural network?
2. Why the Radial Basis Function kernel specifically?
3. What is the role of the global StandardScaler and what would happen without it?
4. How does the hysteresis layer differ from simple thresholding?
5. How is the Distracted class constructed when NTHU-DDD does not have a native "distracted" label?
6. What is the PERCLOS sliding window doing mathematically and why sixty frames specifically?
7. What is the Perspective-n-Point algorithm and how does it convert landmarks into Euler angles?
8. How does this baseline relate to the planned ALERTO thesis?
9. What are the known limitations of this baseline?
10. What would happen if the camera were placed elsewhere in the cabin?

Each member should be assigned three or four questions to lead on.

### Task S7.5 — Submission Bundle

1. Construct a final deliverable bundle containing:
   - The completed Word report.
   - The demo video.
   - The slide deck.
   - A ZIP archive of the project repository, excluding the `node_modules/`, `venv/`, `dataset/NTHU-DDD/` (too large), `.git/`, and `__pycache__/` directories.
   - A separate text file listing the download URL for `face_landmarker.task` and the download instructions for NTHU-DDD.
2. Place the bundle in `outputs/submission_bundle/`.
3. Validate that a fresh clone of the project from the ZIP can be run end-to-end by following the README.

**Sprint 7 Acceptance Criteria.**

- The Word report is complete with all figures and tables inserted.
- The demo video is recorded and viewable.
- The slide deck is complete.
- The Q&A preparation document covers at least ten anticipated questions.
- The submission bundle is assembled and validated.

---

## Appendix A: Definition-of-Done Criteria

A task is considered Done when, and only when, the following conditions are all satisfied:

1. The code change is committed to the version-controlled repository with a descriptive commit message.
2. Any new function is accompanied by a corresponding unit test that passes.
3. The acceptance criteria listed in the task description are demonstrably met.
4. The change is documented either inline (via docstrings) or in `README.md`.
5. A second member has reviewed the change and confirmed correctness.

## Appendix B: Member Task Allocation Summary

The following table consolidates the principal owner for each sprint and identifies cross-cutting collaborations.

| Sprint | Principal Owner | Supporting Member | Estimated Effort (person-days) |
|---|---|---|---|
| 0 | Bacolor | Caole, Soriano | 3 |
| 1 | Bacolor | — | 1 |
| 2 | Bacolor | — | 2 |
| 3 | Bacolor | Soriano | 3 |
| 4 | Caole | — | 3 |
| 5 | Caole | Soriano | 3 |
| 6 | Soriano | All members | 3 |
| 7 | Soriano | All members | 3 |

> *Table 4.* Member task allocation by sprint.

## Appendix C: Sequencing Constraints

The eight sprints are not fully independent. The diagram below summarizes the dependencies that must be respected.

```
Sprint 0 (Refactoring)
   │
   ├─→ Sprint 1 (Phase 1 validation)
   │       │
   │       └─→ Sprint 2 (Phase 2 validation)
   │               │
   │               └─→ Sprint 3 (Phase 3 completion)
   │                       │
   │                       └─→ Sprint 4 (Phase 4 completion)
   │                               │
   │                               └─→ Sprint 5 (Frontend polish)
   │                                       │
   │                                       └─→ Sprint 6 (Integration)
   │                                               │
   │                                               └─→ Sprint 7 (Defense prep)
```

**Critical sequencing rules.**

- Sprint 0 must complete before any of Sprints 1–7 begin, because the regenerated dataset and retrained model are inputs to everything downstream.
- Within Sprints 1–4, validation flows in the order Phase 1 → 2 → 3 → 4 because each phase depends on the correctness of the preceding phase.
- Sprint 5 (Frontend) can in principle begin in parallel with Sprint 4 (Backend), but the Mock-Up alignment work in Task S5.1 depends on the `/metrics` schema being finalized in Task S4.3.
- Sprint 6 (Integration) cannot begin until Sprints 4 and 5 are both complete.
- Sprint 7 (Defense prep) can begin a partial pass concurrently with Sprint 6, particularly for the slide-deck and Q&A-preparation tasks.

## Appendix D: Quick-Reference Command Cheat Sheet

The following commands are referenced repeatedly throughout the sprints. Pinning them here saves time.

| Purpose | Command |
|---|---|
| Activate virtual environment | `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix) |
| Install Python dependencies | `pip install -r requirements.txt` |
| Install frontend dependencies | `cd frontend && npm install` |
| Run feature extraction | `py training/extract_features.py` |
| Run SVM training | `py training/train_svm.py` |
| Run evaluation | `py evaluation/evaluate_baseline.py` |
| Run standalone OpenCV mode | `py main.py` |
| Run standalone with rule-based classifier | `py main.py --mode rule` |
| Run Flask backend | `py app.py` |
| Run React dev server | `cd frontend && npm run dev` |
| Build React production | `cd frontend && npm run build` |
| Run unit tests | `py -m pytest tests/ -v` |
| Run unit tests with coverage | `py -m pytest tests/ -v --cov=modules --cov-report=html` |
| Profile main loop | `py -m cProfile -o outputs/profile.prof main.py` |

> *Table 5.* Quick-reference command cheat sheet.

---

*End of implementation plan.*
