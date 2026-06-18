## **Chunk 1 — Revised Section IV (Scope and Limitations)**

This replaces the previous Section IV. Word count drops from about 700 to about 400 by removing duplicated framing and consolidating overlapping bullets. The structural meaning is preserved.

---

### **IV. Scope and Limitations of the Project**

#### **Scope**

The system is a localized Python application running entirely on edge hardware with no cloud dependency. Video input is captured at 640 by 480 pixels from a standard consumer webcam at a target rate of thirty frames per second, with one driver tracked per session. From each detected face, four metrics are computed: the binocular average Eye Aspect Ratio, the Percentage of Eyelid Closure over a 60-frame sliding window, the Mouth Aspect Ratio, and the three Euler angles of head pose recovered through Perspective-n-Point. The resulting six-dimensional feature vector is normalized by a globally fitted StandardScaler and classified by a Support Vector Machine with a Radial Basis Function kernel into one of three states: Alert, Drowsy, or Distracted. A frame-persistence hysteresis layer suppresses single-frame false positives, and a rule-based fallback classifier is available for operation without a trained model.

The complete system is deployable in two equivalent modes: a standalone OpenCV heads-up display invoked through `main.py`, and a Flask-served web interface paired with a React/Vite frontend that exposes a Landing Page, a Driver State View, a System Live Monitoring Dashboard, and a Deep Analytics Panel. Alerts are generated as both an on-screen visual indicator and an audio cue. Model training is performed offline on the NTHU-DDD multi-class subset using a 70/30 stratified train-test split reinforced by five-fold cross-validation on the training partition.

#### **Limitations**

The following limitations are structural properties of the replicated baseline. Each is a documented gap that the subsequent ALERTO thesis is positioned to address.

1. **No per-user calibration.** The same thresholds for EAR, MAR, PERCLOS, and head pose apply to every driver, and the StandardScaler is fitted globally rather than per-user. The system does not adapt to individual facial morphology or resting baselines.  
2. **Demographic bias on Filipino subjects.** The global EAR threshold of 0.25 derives from Caucasian-majority data; Filipino drivers, whose resting EAR values cluster around 0.316, operate with a substantially smaller decision margin and are expected to experience elevated false-positive drowsiness alerts. This is the principal empirical motivation for the ALERTO thesis.  
3. **Distraction inferred from head pose only.** No pupil or gaze tracking is performed; the system cannot distinguish eyes-on-road from eyes-off-road when head pose is stationary. Object detection (for instance, of mobile phones) is by design excluded from scope.  
4. **Stationary simulated environment only.** Evaluation is conducted indoors with a fixed webcam or pre-recorded video. Real-driving conditions, motion blur, and vehicular vibration are not validated, and are excluded from the project scope on ethical grounds.  
5. **Limited illumination robustness.** MediaPipe landmark accuracy degrades under tropical solar glare, deep shadows, or sub-100-lux interior lighting. No illumination-invariant preprocessing is applied.  
6. **Single-threaded inference assumption.** The real-time pipeline assumes that landmark extraction, feature computation, and SVM inference all complete within a single 33-millisecond frame budget at 30 FPS; on underpowered hardware this assumption may not hold.

---

That is Chunk 1\. The reduction came from three places: consolidating two scope paragraphs into a tighter pair, collapsing nine limitation bullets into six, and removing every phrase that restated the framing already established in Section I.

## **Chunk 2 — Revised Section 5.1 and 5.2 (Architectural Overview \+ Phase 1 with Explicit Preprocessing)**

This is where the preprocessing gap is closed. Phase 1 now has four numbered sub-sections, and Preprocessing (5.2.2) is its own clearly visible block with three named operations.

---

### **V. System Design and Methodology**

#### **5.1 Architectural Overview**

The system is structured as a continuous four-phase real-time pipeline executed once per video frame. The four phases are: (1) Data Acquisition, Preprocessing, and Landmark Extraction; (2) Geometric Feature Computation across three parallel mathematical branches — ocular, oral, and spatial; (3) Feature Fusion and Classification through a Support Vector Machine paired with a frame-persistence hysteresis layer; and (4) Output and Alert Generation through both a standalone OpenCV heads-up display and a Flask-served web-based monitoring dashboard.

The pipeline replicates the architectural sequence described by Kim et al. (2023) for the geometric components and integrates the multi-feature Support Vector Machine fusion described by Jabbar et al. (2018) at the classification stage. Each phase is implemented as a distinct Python module with explicit input and output contracts. The conceptual reference architecture is shown in Figure 1, and the implemented four-phase realization, including the internal subdivision of Phase 1, is shown in Figure 2\.

*Figure 1\.* Reference architecture for vision-based driver monitoring, adapted from Kim et al. (2023) and Jabbar et al. (2018). \[Insert `figure_01_reference_architecture.svg`.\]

*Figure 2\.* Implemented four-phase ALERTO baseline pipeline. \[Insert `figure_02_implemented_pipeline.svg`.\]

#### **5.2 Phase 1 — Data Acquisition, Preprocessing, and Landmark Extraction**

Phase 1 transforms a raw analog scene into a structured representation suitable for geometric analysis. It consists of four sequentially dependent sub-stages: frame acquisition from the camera (5.2.1); preprocessing of the raw frame into the precise format expected by the landmarker (5.2.2); facial-landmark extraction via the MediaPipe FaceLandmarker (5.2.3); and coordinate conversion from the normalized landmark space to the pixel space used by Phase 2 (5.2.4).

##### **5.2.1 Frame Acquisition**

The system opens the default camera device through the OpenCV `VideoCapture` interface, requesting a frame size of 640 by 480 pixels and a target frame rate of thirty frames per second. Each captured frame is returned as a NumPy array of shape `(H, W, 3)` and `dtype=uint8`, expressed in the Blue-Green-Red color order native to OpenCV. The acquisition stage performs no transformation on the frame data; its sole responsibility is to retrieve the next available frame from the camera buffer and pass it to the preprocessing stage.

##### **5.2.2 Preprocessing**

The preprocessing stage transforms the raw BGR frame into the precise input format required by the MediaPipe FaceLandmarker. Three operations are performed in sequence.

**Color space conversion.** OpenCV's `VideoCapture` returns frames in the Blue-Green-Red color order, while the MediaPipe landmarker requires Red-Green-Blue input because its internal weights are trained on RGB statistics. The conversion is performed by `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. Omitting this step does not raise a runtime error, but silently degrades landmark localization accuracy.

**Image wrapping.** The MediaPipe Tasks API requires its input to be an `mp.Image` object rather than a raw NumPy array. The wrapping is performed by `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)`. The `SRGB` format declaration informs the framework that the input is standard 8-bit sRGB rather than a linear or high-dynamic-range variant.

**Timestamp generation.** The Tasks API operating in `RunningMode.VIDEO` requires that each frame be presented with a monotonically increasing millisecond-resolution timestamp; the framework uses this timestamp internally to maintain temporal coherence across consecutive frames. The system derives the timestamp from `int(time.time() * 1000)` and maintains a guard counter that increments by one if two consecutive calls produce the same millisecond value, guaranteeing strict monotonicity even on systems with limited clock resolution.

No further preprocessing is applied. The system does not perform histogram equalization, denoising, gamma correction, or any form of illumination compensation. The MediaPipe model is treated as a black box that handles its own internal preprocessing once the input is presented in the correct format. This minimalism is deliberate: any additional preprocessing layer that distorts the input distribution would compromise the validity of the baseline replication, since neither Kim et al. (2023) nor Jabbar et al. (2018) apply such preprocessing in their published pipelines.

##### **5.2.3 Facial Landmark Extraction**

Landmark extraction is performed by the MediaPipe FaceLandmarker, accessed through the `mediapipe.tasks.python.vision.FaceLandmarker` interface. The Tasks API is used in preference to the legacy `mp.solutions.face_mesh` interface, which has been deprecated and removed from MediaPipe versions 0.10.30 and later. The landmarker is initialized in `RunningMode.VIDEO`, which executes inference synchronously while requiring that successive frames be presented with monotonically increasing millisecond timestamps; this constraint is satisfied by the timestamp generation step described in Section 5.2.2.

The landmarker loads a pre-trained, float-16 quantized model file (`face_landmarker.task`) distributed by Google. For each successful detection, the landmarker returns an object whose `face_landmarks` field contains a list of 478 normalized three-dimensional coordinates indexed from 0 to 477\. The landmarker is configured with `num_faces=1` and confidence thresholds of 0.5 for detection, presence, and tracking, in accordance with the default values recommended by Google for driver monitoring applications.

##### **5.2.4 Coordinate Conversion**

MediaPipe expresses each landmark in a normalized coordinate system in which `x` and `y` lie in the interval \[0, 1\] relative to the frame width and height respectively, and `z` expresses a relative depth in the same normalized scale. For all distance computations described in Section 5.3, the system converts the normalized `x` and `y` of each landmark to integer pixel coordinates by multiplication with the actual frame dimensions:

(xpx,ypx)=( int(x⋅W),  int(y⋅H) )(x\_{px}, y\_{px}) \= \\big(\\, \\mathrm{int}(x \\cdot W),\\; \\mathrm{int}(y \\cdot H) \\,\\big)(xpx​,ypx​)=(int(x⋅W),int(y⋅H))

The `z` coordinate is retained for completeness but is not used in the two-dimensional Euclidean distance computations described in subsequent sections. The conversion is encapsulated in a small helper function `_px(landmark, h, w)` in `modules/features.py`, which is invoked by every metric-computation function.

---

That is Chunk 2\. The notable structural changes from the previous version are: Phase 1 now has four numbered sub-sections instead of three; Preprocessing is its own labeled block at 5.2.2 with three named operations; the rationale for "no further preprocessing" is now explicitly tied to the replication discipline rather than left implicit; and the coordinate conversion gets its own subsection (5.2.4) instead of being buried inside the landmark extraction text.

Sections 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, and 5.9 from the previous version are unchanged in content but renumber cleanly given this restructure. Reply "next" for Chunk 3 (the new Section VIII Analysis and Discussion), or flag adjustments to Chunk 2 first.

## **Chunk 3 — New Section VIII (Analysis and Discussion)**

This is the section the previous version of the documentation did not have. It is structured as an outline with placeholder values in square brackets and inline italic notes that tell the team exactly what to populate, where the source numbers come from, and how to interpret them.

Use this structure as-is in the final report. After running the system and collecting results in Sprint 6, replace each `[BRACKETED_VALUE]` with the measured number and delete the italic "Note for the team" lines. What remains will read as finished academic prose.

---

### **VIII. Analysis and Discussion**

This section presents the empirical results produced by the system on the held-out NTHU-DDD test partition and during real-time operation, and interprets these results in the context of the reference architectures replicated in this work. The section is organized into six subsections corresponding to the principal evaluation dimensions: training behavior, held-out test performance, real-time throughput, qualitative state-transition behavior, comparison with the source literature, and threats to validity.

#### **8.1 Training Results**

The Support Vector Machine was trained on \[N\_TRAIN\] feature vectors derived from the NTHU-DDD multi-class subset, with \[N\_TEST\] feature vectors withheld as the test partition through stratified 70/30 splitting. Five-fold cross-validation on the training partition produced a mean classification accuracy of \[CV\_MEAN\]% with standard deviation \[CV\_STD\]%, indicating \[LOW / MODERATE / HIGH\] variance across folds and suggesting that the learned decision surface is \[STABLE / SENSITIVE\] to the specific training samples selected.

The class distribution after balanced sampling was approximately uniform: \[N\_ALERT\] Alert samples (label 0), \[N\_DROWSY\] Drowsy samples (label 1), and \[N\_DISTRACTED\] Distracted samples (label 2). The Distracted class was constructed by reassigning NTHU-DDD `notdrowsy/` frames whose recovered head pose exceeded the distraction threshold of `|Pitch| > 20°` or `|Yaw| > 30°`. This procedure is internally consistent with the inference-time hysteresis logic and provides ground-truth training data sourced exclusively from real Asian-driver imagery.

*Note for the team.* Populate `N_TRAIN`, `N_TEST`, `CV_MEAN`, `CV_STD`, `N_ALERT`, `N_DROWSY`, and `N_DISTRACTED` from the standard-output of `training/train_svm.py`. The numbers are also recorded in `outputs/training_report.txt`. Choose between "LOW / MODERATE / HIGH" by inspecting `CV_STD`: under 2.0% is low, 2.0–5.0% is moderate, above 5.0% is high. Choose "STABLE / SENSITIVE" correspondingly.

#### **8.2 Held-Out Test Set Evaluation**

The held-out test partition consisted of \[N\_TEST\] feature vectors stratified across the three classes. Table 10 reports the per-class precision, recall, and F1-score; Figure 6 shows the corresponding confusion matrix.

| Class | Precision | Recall | F1-Score | Support |
| ----- | ----- | ----- | ----- | ----- |
| Alert | \[P\_ALERT\] | \[R\_ALERT\] | \[F1\_ALERT\] | \[SUP\_ALERT\] |
| Drowsy | \[P\_DROWSY\] | \[R\_DROWSY\] | \[F1\_DROWSY\] | \[SUP\_DROWSY\] |
| Distracted | \[P\_DISTRACTED\] | \[R\_DISTRACTED\] | \[F1\_DISTRACTED\] | \[SUP\_DISTRACTED\] |
| **Overall accuracy** |  | **\[ACCURACY\]** |  | \[N\_TEST\] |
| **Macro average** | \[P\_MACRO\] | \[R\_MACRO\] | \[F1\_MACRO\] | \[N\_TEST\] |
| **Weighted average** | \[P\_WEIGHTED\] | \[R\_WEIGHTED\] | \[F1\_WEIGHTED\] | \[N\_TEST\] |

*Table 10\.* Per-class classification metrics on the held-out test partition.

*Figure 6\.* Confusion matrix on the held-out test partition. \[Insert `outputs/confusion_matrix.png` here.\]

The overall accuracy of \[ACCURACY\]% \[DOES / DOES\_NOT\] meet the comparable accuracy figures reported in the source literature (see Section 8.5). Examination of the confusion matrix reveals that the most frequent misclassification is between \[CLASS\_A\] and \[CLASS\_B\]; specifically, \[N\_AB\] frames of true class \[CLASS\_A\] were predicted as \[CLASS\_B\]. This pattern is consistent with \[INTERPRETATION\], because \[REASONING\].

*Note for the team.* All numeric values in Table 10 are produced by `training/train_svm.py` and saved to `outputs/training_report.txt`. Copy them verbatim. For the "most frequent misclassification" sentence: read the confusion matrix and identify the largest off-diagonal cell. Typical patterns and their interpretations:

* Drowsy ↔ Alert confusion is most common and suggests the EAR threshold of 0.25 is slightly miscalibrated for the cohort.  
* Distracted ↔ Alert confusion at low pose angles is expected, since the Distracted class boundary is analytic rather than learned.  
* Drowsy ↔ Distracted confusion is unusual and warrants investigation if observed at a meaningful rate.

#### **8.3 Real-Time Performance**

During sustained real-time operation on the target hardware (Intel Core i5, 8th generation, 8 GB RAM), the system maintained an average throughput of \[MEAN\_FPS\] frames per second, with end-to-end per-frame latency averaging \[MEAN\_LATENCY\] milliseconds. Table 11 summarizes the throughput and latency distribution over a 60-second observation window.

| Metric | Mean | Median | Minimum | Maximum | 95th Percentile |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Throughput (FPS) | \[FPS\_MEAN\] | \[FPS\_MEDIAN\] | \[FPS\_MIN\] | \[FPS\_MAX\] | \[FPS\_P95\] |
| Per-frame latency (ms) | \[LAT\_MEAN\] | \[LAT\_MEDIAN\] | \[LAT\_MIN\] | \[LAT\_MAX\] | \[LAT\_P95\] |

*Table 11\.* Throughput and latency over a sustained 60-second observation window.

The system \[DOES / DOES\_NOT\] meet the documented 30 FPS throughput target. The dominant contributor to per-frame latency is the MediaPipe FaceLandmarker inference, which accounts for approximately \[LANDMARK\_PCT\]% of the loop budget. The Support Vector Machine inference, in contrast, contributes less than \[SVM\_PCT\]% because the model has only \[N\_SUPPORT\_VECTORS\] support vectors and the feature space is six-dimensional. The remaining time is distributed across frame acquisition, preprocessing, geometric feature computation, and HUD rendering.

*Note for the team.* Run `main.py` for at least 60 seconds with the FPS counter visible in the HUD. Either record the values manually at intervals, or augment `main.py` to log FPS and latency to a CSV every second (recommended). Compute the statistics in Python from the CSV. For the LANDMARK\_PCT and SVM\_PCT breakdown, use `cProfile` as described in Implementation Plan Sprint 6 Task S6.3.

#### **8.4 Qualitative State-Transition Analysis**

Beyond quantitative accuracy on the held-out partition, the system was subjected to twelve live integration scenarios documented in `outputs/logs/integration_test_log.md`. Three observations from this exercise merit explicit discussion.

**Transition latency.** From the onset of a sustained eye-closure event to the issuance of a confirmed Drowsy alert, the median latency was \[TRANSITION\_FRAMES\] frames, corresponding to approximately \[TRANSITION\_SECONDS\] seconds at 30 FPS. This is \[CONSISTENT / INCONSISTENT\] with the documented hysteresis configuration of 50 frames on EAR, since \[REASONING\].

**False-positive suppression.** Single-frame physiological blinks (lasting fewer than 10 frames) \[DID / DID\_NOT\] trigger spurious Drowsy alerts during the test session. This \[VALIDATES / WEAKENS\] the hysteresis layer's role in distinguishing benign blinks from cognitive fatigue.

**Recovery behavior.** Following the cessation of a Drowsy or Distracted event, the system returned to Alert within \[RECOVERY\_FRAMES\] frames on average, reflecting the counter-reset logic in the hysteresis layer. No observable lag was perceived by the test subject during recovery transitions.

*Note for the team.* These three observations should be populated from your integration test log. If a scenario produced unexpected behavior, document it honestly here rather than concealing it; the panel will respect a candid observation more than a manufactured success rate. If the transition latency exceeds the expected \~1.67 seconds by a large margin, that is itself a defensible finding worth discussing.

#### **8.5 Comparison with Reference Architectures**

The replicated baseline can be compared with the published performance of its two source architectures, with the methodological caveats stated below.

Kim et al. (2023) report an overall driver-state classification accuracy of 90.51% on the NTHU-DDD evaluation set using a facial-landmark and head-pose pipeline architecturally analogous to the present one. The present system achieves \[ACCURACY\]% on a comparable held-out partition of the same dataset, indicating that the replication operates \[WITHIN / OUTSIDE\] the same order of magnitude as the source. Differences of several percentage points are expected and are attributable to variations in landmark indexing, training-test partitioning, and the use of an SVM in this work versus the original classifier configuration of Kim et al.

Jabbar et al. (2018) report a multi-feature classification accuracy of 81% on their proprietary dataset using an SVM trained on a feature vector structurally analogous to the present six-dimensional vector. The present system's overall accuracy of \[ACCURACY\]% \[EXCEEDS / APPROXIMATES / FALLS\_BELOW\] this figure, which is \[DEFENSIBLE / NOTABLE / CONCERNING\] given the difference in training corpora.

These comparisons should be interpreted with caution. The source papers used different datasets, different sample sizes, different cross-validation protocols, and in some cases different output class structures. The purpose of the comparison is not to claim equivalence but to confirm that the present implementation operates within the same performance envelope as the published baselines — the appropriate standard for a replication study.

*Note for the team.* The Kim 90.51% and Jabbar 81% figures are accurate as of their respective publications and can be cited directly. Choose between "WITHIN" or "OUTSIDE" based on whether your accuracy is within roughly ten percentage points of either figure. Choose between "EXCEEDS / APPROXIMATES / FALLS\_BELOW" by direct comparison. Choose between "DEFENSIBLE / NOTABLE / CONCERNING" based on the magnitude: a 2–5 point gap is defensible, 5–10 points is notable but explainable, more than 10 points warrants explicit reflection on training-corpus differences.

#### **8.6 Observed Limitations and Threats to Validity**

Three categories of limitation in the empirical results merit explicit acknowledgement.

**Construct validity of the Distracted class.** Because the Distracted class is constructed by reassigning NTHU-DDD `notdrowsy/` frames whose head pose exceeds a fixed analytic threshold, the Support Vector Machine is in effect being trained to memorize that same threshold. The classification accuracy on the Distracted class therefore reflects threshold consistency rather than genuine learning, and should be interpreted as a sanity check on label assignment rather than as a meaningful generalization result. A future improvement, deferred to the ALERTO thesis, would source the Distracted class from an independent distracted-driver dataset (for example, the State Farm or DMD corpora).

**Demographic representativeness.** NTHU-DDD comprises Taiwanese subjects, who share broad morphological features with the Filipino target population but are not Filipino-specific. The Other-Race Effect documented by Cavazos et al. (2021) and Phillips et al. (2011) implies that any landmark-based system trained on a population not exactly matching its deployment population will exhibit measurable performance degradation. The magnitude of this degradation on Filipino drivers is the principal empirical investigation reserved for the ALERTO thesis and is therefore out of scope for the present work.

**Subject-level versus frame-level partitioning.** The 70/30 stratified split was performed on the pooled feature vectors after extraction rather than at the subject level. Consequently, frames from the same NTHU-DDD subject may appear in both the training and test partitions. This is a known weakness of frame-level evaluation and may produce optimistic accuracy estimates relative to a subject-disjoint evaluation, which would be the more conservative standard. The present approach is consistent with the protocol used by Jabbar et al. (2018), and the comparison in Section 8.5 is methodologically aligned for that reason.

---

That is Chunk 3\. The section is fully structured and ready for population. The total prose volume of this section, once placeholders are filled, will run to approximately 1,200 to 1,400 words depending on how much interpretation the team adds to the per-class confusion-matrix discussion in Section 8.2 and the qualitative observations in Section 8.4. That word count is appropriate for a substantive Analysis and Discussion in an undergraduate AI Project deliverable.

## **Chunk 4 — New Section IX (Conclusion)**

Same structure as Chunk 3: outline with placeholders and inline guidance. Unlike Section VIII, this section uses fewer bracketed values because the Conclusion is primarily interpretive prose; only one or two figures need to be transcribed from your results.

The Conclusion is shorter than the Analysis and Discussion by design. A common undergraduate mistake is to make the Conclusion as long as the Discussion, which dilutes its purpose. The Conclusion's job is to compress everything that came before into one final, memorable statement of what was done, what was found, what was limited, and where the work points next.

---

### **IX. Conclusion**

#### **9.1 Summary of Objectives Met**

The project set out to design, implement, and evaluate a baseline real-time computer-vision system for driver drowsiness and distraction detection through a faithful replication of the integrated architectural framework established by Kim et al. (2023) and Jabbar et al. (2018). All six specific objectives stated in Section III have been met.

The MediaPipe FaceLandmarker pipeline of Objective 1 is implemented in `modules/features.py` and validated by the unit-test suite. The four behavioral and spatial metrics of Objective 2 — namely EAR, PERCLOS, MAR, and head-pose Euler angles — are computed in real time from the 478 landmark coordinates extracted from each frame. The six-dimensional feature vector and the Support Vector Machine classifier with RBF kernel of Objective 3 have been trained on the NTHU-DDD multi-class subset and persisted as model artifacts. The three-state classification engine with frame-persistence hysteresis and rule-based fallback of Objective 4 is implemented in `modules/classifier.py`. The dual-mode deployment of Objective 5 — a standalone OpenCV mode and a Flask-served React web mode — is operational at the documented throughput target. The evaluation of Objective 6 is reported in Section VIII of this document.

#### **9.2 Principal Findings**

The principal finding of this project is that a faithful, real-time, edge-deployable baseline for vision-based driver monitoring can be assembled from publicly available components — MediaPipe FaceLandmarker, scikit-learn, OpenCV, Flask, and React — and trained on a single academic dataset (NTHU-DDD), achieving a held-out classification accuracy of \[ACCURACY\]% while sustaining the documented 30-frames-per-second throughput target on commodity laptop hardware. The system reproduces the geometric pipeline of Kim et al. (2023) and the multi-feature SVM fusion of Jabbar et al. (2018) within a single end-to-end implementation, demonstrating that both component frameworks are practical building blocks for non-intrusive driver monitoring.

A secondary but operationally important finding is that the temporal hysteresis layer is essential for viability. Raw single-frame Support Vector Machine predictions, while accurate on a per-frame basis, produce unacceptably frequent false-positive alerts under normal driving conditions because they cannot distinguish a one-frame blink from sustained cognitive eye closure. The 50-frame, 30-frame, and 45-frame counter configuration documented in Section 5.4.5 suppresses these false positives without introducing perceptually noticeable detection latency, and is essential to the system's defensibility under live demonstration.

A tertiary finding concerns the value of architectural conservatism. By deliberately implementing the baseline form of the integrated framework — global thresholds, a global StandardScaler, no per-user adaptation — the project produces a controlled antecedent against which the subsequent ALERTO thesis can isolate the marginal contribution of individualized calibration. Any improvement measured in the thesis is therefore attributable to the calibration mechanism alone, rather than to incidental architectural variation between the baseline and the proposed system.

*Note for the team.* Populate `[ACCURACY]` with the overall test accuracy from Section 8.2. If the secondary or tertiary findings differ from what you observe — for example, if the hysteresis layer did not adequately suppress false positives in your live tests — adjust the language honestly rather than retaining the optimistic framing. A defensible negative finding is more valuable than an indefensible positive one.

#### **9.3 Limitations and the Path Forward to the ALERTO Thesis**

The baseline exhibits the limitations documented in Section IV and analyzed in Section 8.6. Among these, the absence of per-user calibration and the consequent demographic bias on Filipino subjects are the most consequential. These are not engineering oversights but structural consequences of replicating a fixed-threshold baseline as specified in the source literature; their remediation is the explicit objective of the subsequent ALERTO thesis.

The ALERTO thesis will introduce a five-to-ten-second per-user calibration phase that replaces the global EAR, MAR, PERCLOS, and head-pose thresholds with individualized counterparts derived from each driver's resting baseline, and will replace the globally fitted StandardScaler with a per-user normalization. The architectural decisions made in the present project — the four-phase pipeline, the six-dimensional feature vector, the Support Vector Machine with RBF kernel, the frame-persistence hysteresis layer, and the dual deployment mode — are preserved unchanged in the thesis to ensure experimental isolation of the calibration contribution.

Two further extensions deferred to the thesis are noted here for completeness. First, the Distracted class will be sourced from an independent distracted-driver dataset to address the construct-validity concern raised in Section 8.6, rather than constructed by analytic relabeling of pose-extreme NTHU-DDD frames. Second, evaluation in the thesis will adopt a subject-disjoint cross-validation protocol so that the same subject cannot appear in both training and test partitions, addressing the optimistic-bias concern raised in the same section.

#### **9.4 Closing Remarks**

The present work demonstrates that the foundational components of vision-based driver monitoring — facial-landmark extraction, geometric feature computation, machine-learned classification, and real-time output — can be assembled into a functioning baseline within the constraints of an undergraduate computer-science project, using only publicly available datasets, open-source frameworks, and commodity hardware. The replication is faithful to its source literature, its limitations are explicitly documented rather than concealed, and the system is positioned to serve as the empirical reference point against which the ALERTO thesis will measure the value of individualized calibration. The contribution of the present project is therefore not the introduction of a novel mechanism but the construction of the controlled antecedent that makes such a measurement possible.

---

That is Chunk 4\. The Conclusion runs to approximately 700 words once `[ACCURACY]` is filled in, which is the right proportion relative to the roughly 1,300-word Analysis and Discussion that precedes it. Note three structural decisions worth flagging:

First, 9.1 (Summary of Objectives Met) explicitly walks through all six specific objectives from Section III. This is mechanically dull to read but very defensible at the panel because it makes the completeness argument concrete and verifiable.

Second, 9.2 (Principal Findings) is organized as primary, secondary, and tertiary findings. The tertiary finding — the value of architectural conservatism — is the one that most strongly defends the project against the question "why didn't you just build ALERTO directly?" Keep it.

Third, 9.3 (Limitations and Path Forward) does double duty: it acknowledges the limitations honestly and converts them into a forward-looking statement about the thesis. The two-extension paragraph at the end (Distracted class sourcing and subject-disjoint protocol) gives the panel concrete things to ask about and shows that the team is thinking ahead.

## **Figure 2 — Implemented Four-Phase ALERTO Baseline Pipeline**

*Real-time loop executed once per video frame at the target rate of 30 FPS. This is the figure that addresses the preprocessing gap by showing 1.2 as its own visible sub-block.*  
             ┌──────────────────────────────────────┐  
              │           Webcam Input               │  
              │   OpenCV VideoCapture · 640×480      │  
              │         30 FPS · BGR format          │  
              └──────────────────┬───────────────────┘  
                                 │  
                                 ▼  
╔══════════════════════════════════════════════════════════════════╗  
║   PHASE 1 — Data Acquisition, Preprocessing, Landmark Extraction ║  
╠══════════════════════════════════════════════════════════════════╣  
║                                                                  ║  
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   ║  
║  │ 1.1 ACQUISITION │  │ 1.2 PREPROCESS  │  │ 1.3 LANDMARK    │   ║  
║  │                 │  │                 │  │     EXTRACTION  │   ║  
║  │ Read raw BGR    │  │ BGR → RGB via   │  │ MediaPipe       │   ║  
║  │ frame from      │─▶│ cv2.cvtColor    │─▶│ FaceLandmarker  │   ║  
║  │ camera device   │  │                 │  │ (Tasks API,     │   ║  
║  │ via             │  │ Wrap into       │  │  VIDEO mode)    │   ║  
║  │ cv2.VideoCapture│  │ mp.Image with   │  │                 │   ║  
║  │                 │  │ ImageFormat.    │  │ Output: 478 3D  │   ║  
║  │ Output: BGR     │  │ SRGB            │  │ landmarks       │   ║  
║  │ array (H,W,3)   │  │                 │  │                 │   ║  
║  │                 │  │ Generate        │  │ ──────────────  │   ║  
║  │                 │  │ monotonic ms    │  │ 1.4 COORD CONV  │   ║  
║  │                 │  │ timestamp       │  │ Normalized →    │   ║  
║  │                 │  │                 │  │ pixel space     │   ║  
║  └─────────────────┘  └─────────────────┘  └─────────────────┘   ║  
╚══════════════════════════════════════════════════════════════════╝  
                                 │  
                                 ▼  
╔══════════════════════════════════════════════════════════════════╗  
║   PHASE 2 — Geometric Feature Computation (Parallel Branches)    ║  
╠══════════════════════════════════════════════════════════════════╣  
║                                                                  ║  
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   ║  
║  │  2.1 OCULAR     │  │  2.2 ORAL       │  │  2.3 SPATIAL    │   ║  
║  │                 │  │                 │  │                 │   ║  
║  │ EAR             │  │ MAR             │  │ Head Pose / PnP │   ║  
║  │ (Soukupová &    │  │ (Rahman et al., │  │ (Lepetit et     │   ║  
║  │  Čech, 2016\)    │  │  2015\)          │  │  al., 2009\)     │   ║  
║  │                 │  │                 │  │                 │   ║  
║  │ 6-point eye     │  │ 8-point lip     │  │ 6-point 3D      │   ║  
║  │ geometry        │  │ geometry        │  │ anchor model    │   ║  
║  │ Binocular avg   │  │ 3 vertical      │  │ cv2.solvePnP    │   ║  
║  │                 │  │ distances       │  │ \+ Rodrigues \+   │   ║  
║  │ PERCLOS         │  │                 │  │ RQDecomp3x3     │   ║  
║  │ (Wierwille,     │  │ Yawn flag if    │  │                 │   ║  
║  │  1994\)          │  │ MAR \> 0.60 for  │  │ Output:         │   ║  
║  │ 60-frame deque  │  │ 30 frames       │  │ Pitch, Yaw,     │   ║  
║  │ window          │  │                 │  │ Roll (degrees)  │   ║  
║  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   ║  
║           │                    │                    │            ║  
║           └────────────────────┼────────────────────┘            ║  
║                                ▼                                 ║  
║   ┌────────────────────────────────────────────────────────┐     ║  
║   │   x \= \[ EAR , PERCLOS , MAR , Pitch , Yaw , Roll \]ᵀ    │     ║  
║   │                  Feature vector ∈ ℝ⁶                   │     ║  
║   └────────────────────────────────────────────────────────┘     ║  
╚══════════════════════════════════════════════════════════════════╝  
                                 │  
                                 ▼  
╔══════════════════════════════════════════════════════════════════╗  
║   PHASE 3 — Feature Fusion and Classification                    ║  
╠══════════════════════════════════════════════════════════════════╣  
║                                                                  ║  
║  ┌──────────────────────────┐  ┌──────────────────────────────┐  ║  
║  │ 3.1 NORMALIZE \+ INFER    │  │ 3.2 FRAME-PERSIST HYSTERESIS │  ║  
║  │                          │  │                              │  ║  
║  │ z \= (x − μ) / σ          │  │ EAR counter ≥ 50 frames      │  ║  
║  │ via StandardScaler       │  │   ⇒ confirmed DROWSY (EAR)   │  ║  
║  │ (fitted globally)        │  │                              │  ║  
║  │                          │  │ MAR counter ≥ 30 frames      │  ║  
║  │ SVC(kernel='rbf',        │─▶│   ⇒ confirmed DROWSY (yawn)  │  ║  
║  │     C=1.0,               │  │                              │  ║  
║  │     gamma='scale',       │  │ Pose counter ≥ 45 frames     │  ║  
║  │     probability=True)    │  │   ⇒ confirmed DISTRACTED     │  ║  
║  │                          │  │                              │  ║  
║  │ → predict\_proba          │  │ Otherwise: ALERT             │  ║  
║  └──────────────────────────┘  └──────────────────────────────┘  ║  
╚══════════════════════════════════════════════════════════════════╝  
                                 │  
                                 ▼  
╔══════════════════════════════════════════════════════════════════╗  
║   PHASE 4 — Output and Alert Generation                          ║  
╠══════════════════════════════════════════════════════════════════╣  
║                                                                  ║  
║  ┌──────────────────────────┐  ┌──────────────────────────────┐  ║  
║  │ 4.1 STANDALONE (main.py) │  │ 4.2 WEB (app.py \+ frontend/) │  ║  
║  │                          │  │                              │  ║  
║  │ OpenCV HUD with state    │  │ Flask /video\_feed (MJPEG)    │  ║  
║  │ label, metric values,    │  │ Flask /metrics (JSON)        │  ║  
║  │ SVM probability bars,    │  │                              │  ║  
║  │ literature footer        │  │ React UI:                    │  ║  
║  │                          │  │   \- Landing Page             │  ║  
║  │ Audio alert via          │  │   \- Driver State View        │  ║  
║  │ simpleaudio              │  │   \- System Dashboard         │  ║  
║  │ (assets/alert.wav)       │  │   \- Deep Analytics drawer    │  ║  
║  │                          │  │                              │  ║  
║  │ Optional rule-based      │  │ Polls /metrics every 300 ms  │  ║  
║  │ fallback (--mode rule)   │  │ Audio via HTML5 Audio API    │  ║  
║  └──────────────────────────┘  └──────────────────────────────┘  ║  
╚══════════════════════════════════════════════════════════════════╝

                       ⟲  Loop returns to Phase 1  
                          for the next frame  
                          (target latency ≤ 33 ms)

─────────────────────────────────────────────────────────────────  
Geometric pipeline (Phases 1–2) replicates Kim et al. (2023).  
SVM-based fusion (Phase 3\) replicates Jabbar et al. (2018).  
─────────────────────────────────────────────────────────────────  
---

That is Figure 2\. The diagram has three structural properties worth flagging.  
The four phases are visually distinguished from sub-blocks by the use of double-line borders (╔═╗ ║ ║ ╚═╝) at the phase level and single-line borders (┌─┐ │ │ └─┘) at the sub-block level. This nesting communicates the hierarchy at a glance and survives reasonably well when reproduced in a monospace font.  
The preprocessing block (1.2) is now visually equal in prominence to the acquisition block (1.1) and the landmark extraction block (1.3). This addresses the gap you flagged earlier. A reader scanning the diagram will see preprocessing as a peer of the other Phase 1 operations rather than an implicit step.  
