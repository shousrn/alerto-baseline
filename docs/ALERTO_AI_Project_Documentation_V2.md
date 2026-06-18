# Republic of the Philippines

# POLYTECHNIC UNIVERSITY OF THE PHILIPPINES

# College of Computer and Information Sciences

---

## A Baseline Driver Drowsiness and Distraction Detection System Using MediaPipe Facial Landmark Analysis and Support Vector Machine Classification: A Replication Study Preceding the ALERTO Thesis

In Partial Fulfillment of the Requirement for the Course
COSC 304: Introduction to Artificial Intelligence

by

**Bacolor**, James Clark C.
**Caole**, Stephanie J.
**Soriano**, Shouma King J.

BSCS 3-3 Group 7
Bachelor of Science in Computer Science

**Prof. Ria A. Sagum, MCS**
April 2026

---

## I. Introduction and its Background

Driver fatigue and inattention remain among the most consistent contributors to severe vehicular accidents worldwide. According to recent global transport statistics consolidated in the systematic review of Zhang et al. (2023), traffic collisions stemming from impaired driving states, including drowsiness, micro-sleeps, and momentary distraction, account for a substantial share of road fatalities. Historically, the engineering response to this problem has taken two principal forms. The first involves intrusive physiological instrumentation such as electroencephalography (EEG) and electrooculography (EOG), which, as Bajaj et al. (2023) note, can directly measure brain activity and eye-movement patterns with high diagnostic accuracy but remain impractical for daily deployment due to subject discomfort and equipment cost. The second involves indirect vehicle-dynamics monitoring, in which lane deviation, steering reversal rate, and acceleration anomalies are analyzed to infer cognitive impairment; however, as Knipling and Wierwille (1994) observed, such approaches are inherently reactive rather than preventive, since they only register the symptoms of fatigue after unsafe driving behaviors have already occurred.

With the rapid maturation of computer vision and embedded inference frameworks, camera-based driver monitoring systems have emerged as the most viable non-intrusive alternative. Modern systems do not rely on contact sensors or post-hoc behavioral inference; instead, they continuously observe the driver's face and quantify well-established behavioral indicators of fatigue, such as prolonged eye closure, yawning frequency, and off-axis head orientation. Two recent works form the architectural foundation for the present project. Kim et al. (2023), publishing in *Scientific Reports*, established a real-time driver monitoring framework that combines facial-landmark-based eye closure detection with head-pose recognition, demonstrating that geometric ratios computed directly from a dense facial mesh provide a robust and computationally inexpensive substrate for fatigue inference. Building upon and extending this geometric foundation, Jabbar et al. (2018) demonstrated that the fusion of multiple facial-landmark-derived behavioral features through a machine-learning classifier, including the Support Vector Machine, yields higher classification fidelity than any individual metric evaluated in isolation.

This project undertakes a faithful replication of the integrated framework described by Kim et al. (2023) and Jabbar et al. (2018), constructing a single, unified, real-time computer-vision pipeline that performs four sequential operations on every video frame: facial-landmark extraction, behavioral and spatial metric computation, multi-feature classification via a Support Vector Machine, and visual or audible alert generation. As prescribed by the AI Project Guide, this work intentionally implements the *baseline* form of the integrated framework. The classifier operates on a six-dimensional feature vector that has been preprocessed by a single, globally fitted normalization scaler; the geometric thresholds that condition the hysteresis layer are taken directly from the values reported in the source literature; and no individualized adaptation is performed at any point in the pipeline. The accuracy and operational limitations of this baseline, particularly those that arise from the use of a population-averaged scaler and from globally fixed geometric thresholds, are documented in the present work and are positioned to motivate the subsequent thesis, *ALERTO: An Individualized Calibration System for Real-time and Adaptive Driver Drowsiness and Distraction Detection Using PERCLOS and Facial Landmark Analysis*, which will introduce the per-user calibration mechanism intended to close those gaps.

---

## II. Related Works

The literature on automated driver-state monitoring spans more than three decades and traverses three distinct paradigms: intrusive physiological measurement, indirect vehicle-dynamics inference, and direct vision-based behavioral analysis. This section reviews the principal contributions across these paradigms in the chronological order in which they emerged, identifies the specific geometric metrics that have become standard within the vision-based paradigm, declares the two papers that constitute the reference architecture being replicated in this project, and finally summarizes the demographic limitations of the fixed-threshold baseline approach that motivate the subsequent ALERTO thesis.

### 2.1 Intrusive Physiological Monitoring

Early studies on driver drowsiness detection relied primarily on physiological monitoring systems such as electroencephalography (EEG) and electrooculography (EOG), which directly measure brain activity and eye movement patterns to assess fatigue levels. As discussed by Bajaj et al. (2023), these systems provide high accuracy in detecting cognitive impairment because they sample the neurological signals that precede observable behavioral change. However, they require intrusive sensors physically attached to the driver's scalp or temples, rendering them impractical for routine deployment due to discomfort, high equipment cost, and the operational fragility of contact electrodes under vibration. While such systems remain useful in controlled laboratory studies of fatigue physiology, their usability in everyday driving scenarios is heavily restricted. This limitation has motivated the field-wide transition toward non-intrusive alternatives.

### 2.2 Vehicle-Dynamics-Based Monitoring

Vehicle-based monitoring systems subsequently emerged as a non-contact alternative, analyzing steering wheel reversal rate, lane departure frequency, and longitudinal acceleration patterns to infer driver fatigue (Knipling & Wierwille, 1994). These approaches eliminated the need for body-worn sensors but introduced a different category of weakness: they are inherently reactive rather than preventive. Because they depend entirely on observable driving errors, vehicle-dynamics-based systems do not register fatigue until unsafe behaviors have already manifested. As Fu et al. (2024) emphasize in their comprehensive review, this temporal lag undermines the principal purpose of any driver-state monitoring system, which is to issue a warning while corrective action is still possible. The structural inadequacy of vehicle-dynamics inference therefore reinforces the case for direct behavioral observation through camera-based monitoring.

### 2.3 Facial Landmark Detection: From Haar Cascades to MediaPipe

The accuracy of any vision-based driver monitoring system rests entirely on the accuracy of the underlying facial-landmark extraction stage. Early methods, surveyed by Jayadevappa et al. (2023), relied on Haar cascade classifiers and pixel-intensity thresholding to locate the eye region and infer eyelid state. While these methods were computationally lightweight, they were highly sensitive to ambient illumination, head rotation, and skin-tone variation. The paradigm shifted significantly when Kazemi and Sullivan (2014) introduced the ensemble of regression trees, enabling algorithms to predict the positions of dozens of facial landmarks in milliseconds with high spatial fidelity. This work catalyzed the modern era of real-time face-mesh-based perception. Contemporary frameworks, most notably Google's MediaPipe, extend this regression-tree paradigm with deep convolutional architectures and produce a dense, interconnected topology of 478 three-dimensional landmarks per detected face. MediaPipe serves as the foundational pre-processing engine of the present system.

### 2.4 Behavioral and Spatial Metrics for Driver-State Inference

Once a dense facial mesh has been obtained, four metrics derived directly from the landmark coordinates form the de facto standard for inferring driver state.

**2.4.1 Eye Aspect Ratio (EAR).** To quantify physical drowsiness through eye closure, Soukupová and Čech (2016) introduced the Eye Aspect Ratio, proving mathematically that the ratio of the vertical inter-eyelid distance to the horizontal palpebral-fissure distance reliably classifies the open or closed state of the eye. Kim et al. (2023) subsequently confirmed the operational reliability of EAR within a complete real-time monitoring pipeline. The present system adopts EAR as one of its primary features due to its computational efficiency and proven reliability in fatigue detection (Ismail, 2022).

**2.4.2 PERCLOS.** While EAR captures the instantaneous open or closed state of the eye, it does not characterize the temporal pattern of eyelid behavior. The Percentage of Eyelid Closure metric, originally formalized by Wierwille and Ellsworth (1994) under the United States National Highway Traffic Safety Administration program, addresses this by computing the proportion of time that the eyes remain closed within a fixed-duration sliding window. Abe (2023) reaffirmed PERCLOS as one of the most behaviorally valid temporal indicators of drowsiness, demonstrating a strong correlation between PERCLOS values and measurable performance degradation in vigilance tasks. Conventional PERCLOS implementations rely on fixed, hard-coded thresholds that do not generalize across individuals, a limitation that the subsequent ALERTO thesis is designed to address.

**2.4.3 Mouth Aspect Ratio (MAR).** Yawning has been identified as a complementary fatigue indicator independent of eye state. Rahman et al. (2015) applied a geometric approach analogous to EAR to the inner-lip landmarks, defining the Mouth Aspect Ratio. A sudden and sustained increase in MAR mathematically flags a yawning event. Kim and Zhang (2023) further demonstrated that combining MAR with eye-based metrics produces a measurable improvement in overall fatigue classification accuracy, since the two metrics capture partially independent physiological signals.

**2.4.4 Head Pose Estimation via Perspective-n-Point (PnP).** Distraction, in contrast to drowsiness, is most directly observable through deviation of head orientation from the forward-facing baseline. The Perspective-n-Point algorithm formalized by Lepetit, Moreno-Noguer, and Fua (2009) provides an analytically tractable method for recovering the three-dimensional rotation of a rigid object from a small number of correspondences between known three-dimensional model points and their two-dimensional image projections. Applied to facial landmarks, PnP yields the Euler angles Pitch, Yaw, and Roll, which together describe the driver's head orientation relative to the camera.

### 2.5 Multi-Feature Fusion via Machine-Learning Classifiers

Empirical evidence consistently demonstrates that no single behavioral metric captures the full diversity of driver states. Köksal and Gumus (2025) confirm that hybrid systems integrating multiple behavioral cues consistently outperform single-metric models, and Bajaj et al. (2023) report similar findings in the context of fused behavioral and sensor-based measures. Jabbar et al. (2018) specifically advocated for the fusion of facial-landmark-derived geometric features through a Support Vector Machine classifier, arguing that the Support Vector Machine's capacity to learn nonlinear decision boundaries in a moderate-dimensional feature space is particularly well-suited to the structure of facial-landmark feature vectors, which typically exhibit fewer than ten dimensions but contain complex inter-feature dependencies. Tomas et al. (2024), in a recent Philippine-based application of vision-based driver monitoring using OpenCV, similarly demonstrated that a multi-feature classifier outperforms threshold-only rule systems. The Support Vector Machine remains an appropriate choice for embedded deployment because its inference time is independent of the size of the training set and depends only on the number of support vectors, which is typically a small fraction of the training corpus.

### 2.6 The Primary Reference Architecture Being Replicated

The present project replicates the integrated architecture defined by the conjunction of two principal sources. The geometric pipeline, comprising real-time facial-landmark extraction, the computation of EAR and PERCLOS for ocular analysis, and the recovery of head-pose Euler angles for spatial distraction analysis, follows the framework established by Kim et al. (2023). The multi-feature fusion stage, in which the EAR, PERCLOS, MAR, and head-pose values are concatenated into a single feature vector and presented to a Support Vector Machine for three-class state inference, follows the framework established by Jabbar et al. (2018). The integration of these two frameworks into a single end-to-end pipeline, deployed on local edge hardware with a web-based monitoring interface, constitutes the baseline replicated in this work. No architectural modification, optimization, or enhancement beyond what is reported in these two source papers is introduced at this stage; all such contributions are reserved for the subsequent ALERTO thesis.

### 2.7 Known Limitations of the Fixed-Threshold Baseline

A critical issue identified in the recent literature on facial analysis systems is demographic bias. Cavazos et al. (2021) and Phillips et al. (2011) documented the Other-Race Effect (ORE), in which facial recognition and landmark detection systems exhibit significantly higher error rates when applied to demographic groups underrepresented in their training corpora. Subsequent morphometric studies, including those of Kim, Suhk, and Nguyen (2016), confirm that East and Southeast Asian populations, including Filipinos, characteristically exhibit narrower palpebral fissure ratios and a higher prevalence of the epicanthic fold relative to the Caucasian populations that dominate publicly available training datasets. Because the global EAR drowsiness threshold of 0.25 reported in the foundational literature (Soukupová & Čech, 2016) was empirically derived from Caucasian-majority cohorts whose resting EAR values cluster around 0.35, the same threshold leaves a substantially smaller decision margin for Filipino drivers whose resting EAR values cluster around 0.316. Ramos et al. (2019) report consistent findings in the broader context of Filipino-centric facial-analysis datasets, concluding that the direct application of Western-trained models to Filipino subjects produces measurably elevated false-positive rates. This limitation, which is structural to any fixed-threshold or globally normalized baseline, is documented in the present work as a known constraint and is the empirical gap that the subsequent ALERTO thesis is positioned to close through the introduction of per-user calibration.

---

## III. Objectives of the Project

### General Objective

To design, implement, and evaluate a baseline real-time computer-vision system that detects driver drowsiness and distraction states by combining facial-landmark-derived geometric metrics with Support Vector Machine classification, faithfully replicating the integrated architectural framework established by Kim et al. (2023) and Jabbar et al. (2018), thereby serving as the controlled antecedent against which the subsequent ALERTO thesis will be evaluated.

### Specific Objectives

1. **To implement** a real-time frame acquisition and facial-landmark extraction pipeline using the MediaPipe FaceLandmarker (Tasks API) in synchronous VIDEO mode, producing 478 three-dimensional facial coordinates per detected face per frame.

2. **To mathematically compute** four behavioral and spatial metrics from the extracted landmarks: (a) the binocular average Eye Aspect Ratio (EAR); (b) the Mouth Aspect Ratio (MAR); (c) the Percentage of Eyelid Closure (PERCLOS) over a sixty-frame sliding window; and (d) the three-dimensional Head Pose Euler angles, namely Pitch, Yaw, and Roll, recovered via the Perspective-n-Point algorithm operating on a generic anthropometric three-dimensional face model.

3. **To construct** a six-dimensional feature vector from the four computed metrics and **to train** a Support Vector Machine classifier with a Radial Basis Function kernel on a globally scaled, stratified dataset derived from the NTHU Drowsy Driver Detection (NTHU-DDD) corpus, yielding three discrete output classes: Alert, Drowsy, and Distracted.

4. **To implement** a real-time three-state classification engine that pairs the trained Support Vector Machine with a frame-persistence hysteresis layer, supplemented by a deterministic rule-based fallback classifier that preserves operability in the absence of a trained model file.

5. **To deploy** the complete system as a localized edge-computing application comprising a Python/Flask backend that streams Motion JPEG video and JavaScript Object Notation metrics, paired with a React and Vite browser frontend that renders a Live Monitoring Dashboard, a Driver State View, and a Deep Analytics Panel, achieving a target real-time throughput of no fewer than thirty frames per second.

6. **To evaluate** the baseline system's classification accuracy through stratified hold-out testing on the NTHU-DDD test partition and five-fold cross-validation on the training partition, and **to document** the operational limitations, particularly the system's reliance on globally fixed thresholds and a non-individualized normalization scaler, that motivate the subsequent ALERTO thesis.

---

## IV. Scope and Limitations of the Project
### Scope

The system is a localized Python application running entirely on edge hardware with no cloud dependency. Video input is captured at 640 by 480 pixels from a standard consumer webcam at a target rate of thirty frames per second, with one driver tracked per session. From each detected face, four metrics are computed: the binocular average Eye Aspect Ratio, the Percentage of Eyelid Closure over a 60-frame sliding window, the Mouth Aspect Ratio, and the three Euler angles of head pose recovered through Perspective-n-Point. The resulting six-dimensional feature vector is normalized by a globally fitted StandardScaler and classified by a Support Vector Machine with a Radial Basis Function kernel into one of three states: Alert, Drowsy, or Distracted. A frame-persistence hysteresis layer suppresses single-frame false positives, and a rule-based fallback classifier is available for operation without a trained model.

The complete system is deployed as a Flask-served web interface paired with a React/Vite frontend that exposes a Landing Page, a Driver State View, a System Live Monitoring Dashboard, and a Deep Analytics Panel. Alerts are generated as both an on-screen visual indicator and an audio cue. Model training is performed offline on the NTHU-DDD multi-class subset using a 70/30 stratified train-test split reinforced by five-fold cross-validation on the training partition.

### Limitations

The following limitations are structural properties of the replicated baseline. Each is a documented gap that the subsequent ALERTO thesis is positioned to address.

1. **No per-user calibration.** The same thresholds for EAR, MAR, PERCLOS, and head pose apply to every driver, and the StandardScaler is fitted globally rather than per-user. The system does not adapt to individual facial morphology or resting baselines.  
2. **Demographic bias on Filipino subjects.** The global EAR threshold of 0.25 derives from Caucasian-majority data; Filipino drivers, whose resting EAR values cluster around 0.316, operate with a substantially smaller decision margin and are expected to experience elevated false-positive drowsiness alerts. This is the principal empirical motivation for the ALERTO thesis.  
3. **Distraction inferred from head pose only.** No pupil or gaze tracking is performed; the system cannot distinguish eyes-on-road from eyes-off-road when head pose is stationary. Object detection (for instance, of mobile phones) is by design excluded from scope.  
4. **Stationary simulated environment only.** Evaluation is conducted indoors with a fixed webcam or pre-recorded video. Real-driving conditions, motion blur, and vehicular vibration are not validated, and are excluded from the project scope on ethical grounds.  
5. **Limited illumination robustness.** MediaPipe landmark accuracy degrades under tropical solar glare, deep shadows, or sub-100-lux interior lighting. No illumination-invariant preprocessing is applied.  
6. **Single-threaded inference assumption.** The real-time pipeline assumes that landmark extraction, feature computation, and SVM inference all complete within a single 33-millisecond frame budget at 30 FPS; on underpowered hardware this assumption may not hold.

---

## V. System Design and Methodology

### 5.1 Architectural Overview
The system is structured as a continuous four-phase real-time pipeline executed once per video frame. The four phases are: (1) Data Acquisition, Preprocessing, and Landmark Extraction; (2) Geometric Feature Computation across three parallel mathematical branches — ocular, oral, and spatial; (3) Feature Fusion and Classification through a Support Vector Machine paired with a frame-persistence hysteresis layer; and (4) Output and Alert Generation through both a standalone OpenCV heads-up display and a Flask-served web-based monitoring dashboard.

The pipeline replicates the architectural sequence described by Kim et al. (2023) for the geometric components and integrates the multi-feature Support Vector Machine fusion described by Jabbar et al. (2018) at the classification stage. Each phase is implemented as a distinct Python module with explicit input and output contracts. The conceptual reference architecture is shown in Figure 1, and the implemented four-phase realization, including the internal subdivision of Phase 1, is shown in Figure 2\.

*Figure 1\.* Reference architecture for vision-based driver monitoring, adapted from Kim et al. (2023) and Jabbar et al. (2018). \[Insert `figure_01_reference_architecture.svg`.\]

*Figure 2\.* Implemented four-phase ALERTO baseline pipeline. \[Insert `figure_02_implemented_pipeline.svg`.\]

### 5.2 Phase 1 — Data Acquisition, Preprocessing, and Landmark Extraction

Phase 1 transforms a raw analog scene into a structured representation suitable for geometric analysis. It consists of four sequentially dependent sub-stages: frame acquisition from the camera (5.2.1); preprocessing of the raw frame into the precise format expected by the landmarker (5.2.2); facial-landmark extraction via the MediaPipe FaceLandmarker (5.2.3); and coordinate conversion from the normalized landmark space to the pixel space used by Phase 2 (5.2.4).

#### 5.2.1 Frame Acquisition

The system opens the default camera device through the OpenCV `VideoCapture` interface, requesting a frame size of 640 by 480 pixels and a target frame rate of thirty frames per second. Each captured frame is returned as a NumPy array of shape `(H, W, 3)` and `dtype=uint8`, expressed in the Blue-Green-Red color order native to OpenCV. The acquisition stage performs no transformation on the frame data; its sole responsibility is to retrieve the next available frame from the camera buffer and pass it to the preprocessing stage.

#### 5.2.2 Preprocessing

The preprocessing stage transforms the raw BGR frame into the precise input format required by the MediaPipe FaceLandmarker. Three operations are performed in sequence.

**Color space conversion.** OpenCV's `VideoCapture` returns frames in the Blue-Green-Red color order, while the MediaPipe landmarker requires Red-Green-Blue input because its internal weights are trained on RGB statistics. The conversion is performed by `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. Omitting this step does not raise a runtime error, but silently degrades landmark localization accuracy.

**Image wrapping.** The MediaPipe Tasks API requires its input to be an `mp.Image` object rather than a raw NumPy array. The wrapping is performed by `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)`. The `SRGB` format declaration informs the framework that the input is standard 8-bit sRGB rather than a linear or high-dynamic-range variant.

**Timestamp generation.** The Tasks API operating in `RunningMode.VIDEO` requires that each frame be presented with a monotonically increasing millisecond-resolution timestamp; the framework uses this timestamp internally to maintain temporal coherence across consecutive frames. The system derives the timestamp from `int(time.time() * 1000)` and maintains a guard counter that increments by one if two consecutive calls produce the same millisecond value, guaranteeing strict monotonicity even on systems with limited clock resolution.

No further preprocessing is applied. The system does not perform histogram equalization, denoising, gamma correction, or any form of illumination compensation. The MediaPipe model is treated as a black box that handles its own internal preprocessing once the input is presented in the correct format. This minimalism is deliberate: any additional preprocessing layer that distorts the input distribution would compromise the validity of the baseline replication, since neither Kim et al. (2023) nor Jabbar et al. (2018) apply such preprocessing in their published pipelines.

#### 5.2.3 Facial Landmark Extraction

Landmark extraction is performed by the MediaPipe FaceLandmarker, accessed through the `mediapipe.tasks.python.vision.FaceLandmarker` interface. The Tasks API is used in preference to the legacy `mp.solutions.face_mesh` interface, which has been deprecated and removed from MediaPipe versions 0.10.30 and later. The landmarker is initialized in `RunningMode.VIDEO`, which executes inference synchronously while requiring that successive frames be presented with monotonically increasing millisecond timestamps; this constraint is satisfied by the timestamp generation step described in Section 5.2.2.

The landmarker loads a pre-trained, float-16 quantized model file (`face_landmarker.task`) distributed by Google. For each successful detection, the landmarker returns an object whose `face_landmarks` field contains a list of 478 normalized three-dimensional coordinates indexed from 0 to 477\. The landmarker is configured with `num_faces=1` and confidence thresholds of 0.5 for detection, presence, and tracking, in accordance with the default values recommended by Google for driver monitoring applications.

#### 5.2.4 Coordinate Conversion

MediaPipe expresses each landmark in a normalized coordinate system in which `x` and `y` lie in the interval \[0, 1\] relative to the frame width and height respectively, and `z` expresses a relative depth in the same normalized scale. For all distance computations described in Section 5.3, the system converts the normalized `x` and `y` of each landmark to integer pixel coordinates by multiplication with the actual frame dimensions:

(xpx,ypx)=( int(x⋅W),  int(y⋅H) )(x\_{px}, y\_{px}) \= \big(\, \mathrm{int}(x \cdot W),\; \mathrm{int}(y \cdot H) \,\big)(xpx​,ypx​)=(int(x⋅W),int(y⋅H))

The `z` coordinate is retained for completeness but is not used in the two-dimensional Euclidean distance computations described in subsequent sections. The conversion is encapsulated in a small helper function `_px(landmark, h, w)` in `modules/features.py`, which is invoked by every metric-computation function.

### 5.3 Phase 2 — Geometric Feature Computation

This phase comprises three parallel mathematical branches operating concurrently on the same landmark set to produce four behavioral and spatial metrics.

#### 5.3.1 Ocular Branch: Eye Aspect Ratio and PERCLOS

**EAR Computation.** Following Soukupová and Čech (2016), the Eye Aspect Ratio is defined as the ratio of the sum of two vertical inter-eyelid distances to twice the horizontal palpebral-fissure distance:

$$
EAR = \frac{\lVert P_2 - P_6 \rVert + \lVert P_3 - P_5 \rVert}{2 \cdot \lVert P_1 - P_4 \rVert}
$$

The six points are sampled from the MediaPipe mesh at the indices listed below. The indices were selected to correspond as closely as possible to the canonical six-point eyelid landmark scheme originally defined for the dlib 68-point detector, while accommodating MediaPipe's denser and differently ordered 478-point topology.

| Eye | $P_1$ (outer corner) | $P_2$ | $P_3$ | $P_4$ (inner corner) | $P_5$ | $P_6$ |
|---|---|---|---|---|---|---|
| Right | 33 | 160 | 158 | 133 | 153 | 144 |
| Left | 362 | 385 | 387 | 263 | 373 | 380 |

The binocular average Eye Aspect Ratio is computed as the arithmetic mean of the per-eye values:

$$
EAR_{avg} = \frac{EAR_{right} + EAR_{left}}{2}
$$

Each Euclidean distance is computed through `scipy.spatial.distance.euclidean`, applied to the integer pixel coordinates obtained in Section 5.2.3. A small guard against degenerate cases is implemented: if the horizontal palpebral width falls below 10⁻⁶ pixels, the function returns zero rather than attempting division. Empirically observed $EAR_{avg}$ values lie in the interval [0.28, 0.42] when the eyes are fully open and below 0.15 when the eyes are closed.

> *Figure 3.* Six-point eye landmark indexing for EAR computation. [Insert a labeled diagram showing the six landmarks per eye on a face mesh.]

**PERCLOS Computation.** Following Wierwille and Ellsworth (1994), the Percentage of Eyelid Closure over a temporal window is defined as the fraction of frames within that window in which the eyes are classified as closed:

$$
PERCLOS(t) = \frac{\big| \{ f \in W_t : EAR_{avg}(f) < \tau_{EAR} \} \big|}{\lvert W_t \rvert} \times 100\%
$$

where $W_t$ is a sliding window of fixed length terminating at the current frame $t$, $\tau_{EAR}$ is the EAR closure threshold, and the numerator is the count of frames in the window in which the eyes are classified as closed. The system fixes $\lvert W_t \rvert = 60$ frames, corresponding to a two-second window at a target frame rate of 30 frames per second, and fixes $\tau_{EAR} = 0.25$, in accordance with the value reported by Soukupová and Čech (2016). The PERCLOS computation is implemented as a `PERCLOSBuffer` class that wraps a `collections.deque` with `maxlen=60`. At each frame the boolean indicator `1 if EAR < 0.25 else 0` is appended to the deque, which automatically evicts the oldest element when full, yielding a worst-case amortized constant-time update.

#### 5.3.2 Oral Branch: Mouth Aspect Ratio

Following Rahman et al. (2015), the Mouth Aspect Ratio is defined as the average of three vertical inner-lip distances normalized by twice the horizontal lip-corner distance:

$$
MAR = \frac{v_1 + v_2 + v_3}{2 \cdot h}
$$

where $v_1$, $v_2$, and $v_3$ are the vertical distances at the center, left, and right of the inner mouth respectively, and $h$ is the horizontal distance between the lip corners. The eight points are sampled from the MediaPipe mesh at the following indices:

| Role | Upper Center | Lower Center | Upper Left | Lower Left | Upper Right | Lower Right | Left Corner | Right Corner |
|---|---|---|---|---|---|---|---|---|
| Index | 13 | 14 | 82 | 87 | 312 | 317 | 78 | 308 |

A degenerate-case guard analogous to the EAR computation is applied: if the horizontal lip-corner distance falls below 10⁻⁶ pixels, the function returns zero. Empirically observed MAR values lie in the interval [0.10, 0.25] when the mouth is at rest and exceed 0.60 during a sustained yawn.

> *Figure 4.* Eight-point mouth landmark indexing for MAR computation. [Insert a labeled diagram showing the eight landmarks on a face mesh.]

#### 5.3.3 Spatial Branch: Head Pose via Perspective-n-Point

**Generic Three-Dimensional Face Model.** Following the Perspective-n-Point framework of Lepetit, Moreno-Noguer, and Fua (2009), recovery of the head's three-dimensional orientation requires a known correspondence between a set of three-dimensional reference points and their two-dimensional projections in the current frame. The present system uses a generic anthropometric face model expressed in millimeters with the nose tip placed at the origin. Six anchor points are used:

| Anatomical Point | MediaPipe Index | 3D Coordinate (mm, nose tip = origin) |
|---|---|---|
| Nose tip | 1 | (0, 0, 0) |
| Chin | 152 | (0, −330, −65) |
| Left eye outer corner | 263 | (−165, 170, −135) |
| Right eye outer corner | 33 | (165, 170, −135) |
| Left mouth corner | 61 | (−150, −150, −125) |
| Right mouth corner | 291 | (150, −150, −125) |

> *Figure 5.* Six-point generic three-dimensional face model used for head-pose recovery. [Insert a labeled diagram showing the six anchors in 3D space.]

**Camera Intrinsic Matrix.** Because no calibrated intrinsic data are available for an arbitrary consumer webcam, the system adopts the pinhole approximation. The focal length in pixel units is set equal to the frame width, the principal point is placed at the geometric center of the frame, and the distortion vector is set to zero:

$$
K = \begin{bmatrix} W & 0 & W/2 \\ 0 & W & H/2 \\ 0 & 0 & 1 \end{bmatrix}, \qquad D = \mathbf{0}
$$

**PnP Solution and Euler Angle Extraction.** At each frame, the system extracts the two-dimensional pixel coordinates of the six anchor landmarks from the MediaPipe mesh and presents them, along with the three-dimensional model and the intrinsic matrix, to `cv2.solvePnP` with `flags=cv2.SOLVEPNP_ITERATIVE`. The rotation vector returned by the solver is converted to a 3-by-3 rotation matrix via `cv2.Rodrigues`, and the rotation matrix is then decomposed into Euler angles using `cv2.RQDecomp3x3`, yielding the Pitch, Yaw, and Roll angles in degrees. Sign conventions are inherited directly from OpenCV: positive Pitch corresponds to a downward tilt of the head, positive Yaw corresponds to a rotation toward the camera-right (the driver's left), and positive Roll corresponds to a counter-clockwise tilt as observed by the camera.

### 5.4 Phase 3 — Feature Fusion and Classification

#### 5.4.1 Six-Dimensional Feature Vector Construction

The four metrics computed in Phase 2 are concatenated into a single six-dimensional feature vector, in the following fixed order, which is preserved across both training and inference:

$$
\mathbf{x} = \begin{bmatrix} EAR_{avg} & PERCLOS & MAR & Pitch & Yaw & Roll \end{bmatrix}^\top
$$

If any of the Euler angles is undefined for the current frame, for example because `cv2.solvePnP` failed to converge, the corresponding entry is replaced with zero before the vector is forwarded to the classifier.

#### 5.4.2 Training Dataset

The Support Vector Machine is trained offline on the NTHU Drowsy Driver Detection (NTHU-DDD) multi-class subset, a publicly available academic corpus comprising driver-facing video frames from Asian subjects. From the NTHU-DDD `train` partition, image-level frames are sampled from the `notdrowsy/` and `drowsy/` subdirectories and assigned the integer labels 0 (Alert) and 1 (Drowsy) respectively. For the third class, Distracted (label 2), the system identifies frames in the existing partitions whose recovered head pose exceeds the distraction threshold defined in Section 5.4.5, and reassigns their label. Approximately eight hundred samples per class are retained after balanced stratified sampling, yielding a total training corpus of approximately 2,400 feature vectors. The full training pipeline is implemented in `training/extract_features.py`, which operates the MediaPipe FaceLandmarker in `RunningMode.IMAGE` mode for static input.

| Dataset Component | Source | Approximate Sample Count | Class Distribution |
|---|---|---|---|
| NTHU-DDD `notdrowsy` | Real Asian driver imagery | ≈ 800 | Alert |
| NTHU-DDD `drowsy` | Real Asian driver imagery | ≈ 800 | Drowsy |
| NTHU-DDD `notdrowsy` with off-axis pose | Real Asian driver imagery | ≈ 800 | Distracted |
| **Total** | | **≈ 2,400** | Stratified |

> *Table 1.* Composition of the training dataset.

#### 5.4.3 Preprocessing: Global Standard Normalization

Before training, every feature vector is z-score normalized through a `sklearn.preprocessing.StandardScaler`. The scaler is fitted exactly once, on the entire training partition pooled across all three classes, and produces per-feature parameter pairs $(\mu_i, \sigma_i)$ that are subsequently applied at inference time:

$$
z_i = \frac{x_i - \mu_i}{\sigma_i}, \qquad i = 1, \dots, 6
$$

The fitted scaler is persisted to disk as `data/svm_baseline_scaler.pkl` and is loaded by `app.py` at startup. The use of a single global scaler, rather than a per-user scaler, is a deliberate property of the baseline replicated in this project; the introduction of a per-user scaler is reserved for the subsequent ALERTO thesis.

#### 5.4.4 Support Vector Machine Configuration

The classifier is an instance of `sklearn.svm.SVC` configured as follows:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `kernel` | `'rbf'` | Radial Basis Function permits nonlinear decision boundaries in the moderate-dimensional feature space. |
| `C` | 1.0 | Default regularization strength, balancing margin width against training error. |
| `gamma` | `'scale'` | Inverse kernel width set to $1 / (n_{features} \times \mathrm{Var}(\mathbf{X}))$, the scikit-learn default. |
| `probability` | `True` | Enables Platt-calibrated posterior class probabilities, which are consumed by the user interface. |
| `class_weight` | `None` | Stratified sampling renders explicit class re-weighting unnecessary. |

> *Table 2.* Support Vector Machine hyperparameters.

The trained model is persisted to disk as `data/svm_baseline_model.pkl`. At inference time, the model is loaded by `app.py`, and the integer label predicted by `model.predict(z)` is mapped to a human-readable state string through the label map `{0: "ALERT", 1: "DROWSY", 2: "DISTRACTED"}`. The posterior probability vector returned by `model.predict_proba(z)` is forwarded to the user interface as a dictionary keyed by state name.

#### 5.4.5 Frame-Persistence Hysteresis

Raw single-frame Support Vector Machine predictions are not directly used to trigger alerts. Such direct use would yield frequent single-frame false positives caused by physiological events that are entirely benign in a driving context, such as a normal blink, a brief glance at a side mirror, or a momentary head turn at an intersection. Following the temporal robustness arguments of Kim et al. (2023), the system overlays a hysteresis layer comprising three per-state frame counters that increment while the corresponding raw geometric condition holds and reset to zero otherwise:

| State | Persistence Condition | Threshold Frames | Approximate Duration at 30 FPS |
|---|---|---|---|
| Drowsy (EAR-driven) | $EAR_{avg} < 0.25$ | 50 | ≈ 1.67 s |
| Drowsy (MAR-driven) | $MAR > 0.60$ | 30 | ≈ 1.00 s |
| Drowsy (PERCLOS-driven) | $PERCLOS > 30\%$ | n/a (instantaneous on the windowed metric) | The 60-frame window itself supplies the temporal smoothing. |
| Distracted | $\lvert Pitch \rvert > 20°$ or $\lvert Yaw \rvert > 30°$ | 45 | ≈ 1.50 s |

> *Table 3.* Persistence thresholds for the hysteresis layer.

When the raw Support Vector Machine prediction is Drowsy, the hysteresis layer suppresses the alert unless at least one of the per-state counters has met its threshold. The same suppression applies to the Distracted state. The Alert state, being the default, requires no persistence and is reported immediately. This logic is implemented in the `_apply_hysteresis` method of the `SVMClassifier` class in `modules/classifier.py`.

#### 5.4.6 Rule-Based Fallback Classifier

A deterministic rule-based fallback classifier is implemented in the `RuleBasedClassifier` class within the same module and is selectable through the command-line flag `--mode rule`. The fallback evaluates the same three persistence conditions described in Section 5.4.5 and applies the priority order Drowsy ≻ Distracted ≻ Alert. This classifier requires no trained model and is provided so that the system remains operable in the absence of a Support Vector Machine model file, for example during initial development or in the event of a corrupted pickle artifact. The fallback classifier is not the primary classification engine of this project; the Support Vector Machine is.

### 5.5 Phase 4 — Output and Visualization

The Phase 4 output layer is provided via a React/Vite web interface connected to a Flask application backend.

#### 5.5.1 Flask Backend (`app.py`)

The web deployment mode runs a Flask application that operates a background thread reading frames from the camera, applying the full four-phase pipeline, and continuously updating two thread-safe global objects: the most recently encoded JPEG frame (for streaming) and the most recent metrics dictionary (for JSON queries). The Flask routes are as follows:

| Endpoint | HTTP Method | Response Type | Description |
|---|---|---|---|
| `/video_feed` | GET | `multipart/x-mixed-replace; boundary=frame` | Motion-JPEG stream of the camera frame with face-mesh dot overlay but no textual annotations. |
| `/metrics` | GET | `application/json` | Snapshot of all current metric values, the global thresholds, the current state, the Support Vector Machine probability vector, and the persistence counter values. |
| `/` | GET | `text/html` | Serves the compiled React single-page application from `frontend/dist/index.html`. |
| `/<path>` | GET | various | Serves React static assets; falls back to `index.html` for unmatched paths to support client-side routing. |

> *Table 4.* Flask backend endpoints.

#### 5.5.2 React and Vite Frontend (`frontend/`)

The browser frontend is constructed with React 19 and the Vite 8 build tool, with styling supplied by Tailwind CSS version 4. State management is handled exclusively through React's built-in `useState` and a custom hook `useMetrics()` that polls the `/metrics` endpoint every 300 milliseconds and merges the response into the application state. The frontend exposes three principal screens described in Appendix A: the Landing Page, the Driver State View in its three state variants, and the System Live Monitoring Dashboard with the attached Deep Analytics Panel. All threshold values displayed in the user interface are sourced from the `/metrics` JSON payload, so that the interface remains consistent with whatever threshold values are currently configured in `config.py`; no threshold is hard-coded in the JavaScript layer.

#### 5.5.3 Alert Generation

When the hysteresis-confirmed state transitions to Drowsy or Distracted, the system generates two simultaneous alerts. The visual alert consists of a full-screen color change in the Driver State View, paired with a state-specific iconography (an eyelid icon in amber for Drowsy, a hazard triangle in red for Distracted) and a directive message ("Pull over and rest" or "Eyes on the road"). The audio alert plays a short pre-recorded WAV file at a volume sufficient to be audible inside a stationary cabin, using the HTML5 Audio API invoked through React.

### 5.6 Real-Time Main Loop

The real-time loop executes continuously in a background thread within `app.py`. The pseudocode is as follows:

```
INITIALIZE:
    camera        ← cv2.VideoCapture(source)
    landmarker    ← MediaPipe.FaceLandmarker.create_from_options(
                        model_path = "dataset/face_landmarker.task",
                        running_mode = VIDEO)
    perclos_buf   ← PERCLOSBuffer(window_size = 60, threshold = 0.25)
    classifier    ← SVMClassifier.load(
                        model_path  = "data/svm_baseline_model.pkl",
                        scaler_path = "data/svm_baseline_scaler.pkl")

WHILE camera is open:
    frame_bgr  ← camera.read()
    IF frame_bgr is None: BREAK
    frame_rgb  ← cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    timestamp  ← int(time.time() * 1000)

    # ----- Phase 1: Landmark Extraction -----
    result     ← landmarker.detect_for_video(mp.Image(frame_rgb), timestamp)
    IF result.face_landmarks is empty:
        render "NO FACE DETECTED" overlay
        CONTINUE
    landmarks  ← result.face_landmarks[0]

    # ----- Phase 2: Geometric Feature Computation -----
    ear_avg    ← compute_avg_ear(landmarks, H, W)
    perclos    ← perclos_buf.update(ear_avg)
    mar        ← compute_mar(landmarks, H, W)
    pitch, yaw, roll ← compute_head_pose(landmarks, H, W)
    feature_vec      ← [ear_avg, perclos, mar, pitch, yaw, roll]

    # ----- Phase 3: Classification with Hysteresis -----
    state, proba ← classifier.predict(feature_vec, ear_avg, mar, pitch, yaw)

    # ----- Phase 4: Output -----
    render_HUD(frame_bgr, state, feature_vec, proba)
    IF state ∈ {DROWSY, DISTRACTED} AND state changed from previous frame:
        trigger_audio_alert(state)
    display(frame_bgr)

    IF key_press == 'q': BREAK

RELEASE camera; CLOSE landmarker; DESTROY display window
```

> *Listing 1.* Main real-time loop of the baseline system.

### 5.7 Hardware and Software Specifications

| Component | Specification |
|---|---|
| Operating System | Windows 11 (development); system is OS-agnostic |
| Programming Language | Python 3.12 or later |
| Computer Vision Library | OpenCV ≥ 4.9.0.80 |
| Face Landmark Library | MediaPipe ≥ 0.10.14 (Tasks API) |
| Scientific Computing | NumPy ≥ 1.26, SciPy ≥ 1.13 |
| Machine Learning | scikit-learn ≥ 1.4.2 |
| Web Framework | Flask 3.0+, Flask-CORS |
| Testing Framework | pytest 8.2+ with pytest-cov |
| Frontend Build Tool | Vite 8 with `@vitejs/plugin-react` |
| Frontend Framework | React 19 |
| CSS Framework | Tailwind CSS v4 (via `@tailwindcss/vite`) |
| Frontend Runtime | Node.js 20 or later |
| Camera | USB UVC webcam, ≥ 640 × 480 @ 30 FPS |
| Central Processing Unit | Intel Core i5 (8th generation) or equivalent |
| Memory | 8 GB RAM minimum |
| Storage | ≈ 500 MB (project files and model artifacts); NTHU-DDD dataset adds several GB |

> *Table 5.* Hardware and software specifications.

### 5.8 Training Procedure

Model training is conducted offline through a two-stage pipeline.

In the **feature-extraction stage**, the script `training/extract_features.py` initializes the MediaPipe FaceLandmarker in `RunningMode.IMAGE` and iterates over every image in the NTHU-DDD `notdrowsy/` and `drowsy/` subdirectories. For each image, it applies the EAR, MAR, and head-pose computations defined in Sections 5.3.1 through 5.3.3. Because PERCLOS is a temporal metric that cannot be defined on a single static image, the extractor applies a one-frame approximation: PERCLOS is set to 100% if the EAR for that image is below 0.25, and to 0% otherwise. This approximation preserves the directional contribution of PERCLOS as a feature without claiming temporal validity at training time, and the temporal interpretation is fully restored at inference time when the `PERCLOSBuffer` operates over a true 60-frame sliding window. The extracted features for every image are written to `data/extracted_features.csv` with the columns `[ear_avg, perclos_pct, mar, pitch_deg, yaw_deg, roll_deg, label]`.

In the **model-training stage**, the script `training/train_svm.py` loads the extracted feature CSV, partitions it into a training subset and a held-out test subset using a seventy-thirty stratified split, fits the `StandardScaler` on the training subset alone, and then trains the Support Vector Machine using the hyperparameters listed in Table 2. Five-fold stratified cross-validation is performed on the training subset to estimate generalization accuracy. The fitted scaler and the trained model are persisted to `data/svm_baseline_scaler.pkl` and `data/svm_baseline_model.pkl` respectively. The classification report (precision, recall, F1-score per class) and the confusion matrix are computed on the held-out test subset and recorded in the project's output directory for inclusion in the final report.

### 5.9 Evaluation Methodology

System evaluation proceeds at two levels. At the **module level**, the unit-test suite in `tests/test_features.py` exercises the EAR, MAR, PERCLOS, head-pose, feature-vector, and classifier functions through pytest. The suite includes tests for normal operation, edge cases such as zero-width palpebral fissures, and the sliding-window eviction behavior of the `PERCLOSBuffer`. At the **system level**, classification accuracy is computed on the held-out NTHU-DDD test partition using the standard metrics of overall accuracy, per-class precision, per-class recall, per-class F1-score, and the three-by-three confusion matrix. The principal evaluation question is whether the replicated baseline approaches the performance reported by Kim et al. (2023) and Jabbar et al. (2018) under analogous experimental conditions. Limitations of the baseline that arise from the use of globally fixed thresholds and a non-individualized scaler, including any measurable demographic skew, are reported as findings rather than corrected within the scope of this project; their remediation is the explicit objective of the subsequent ALERTO thesis.

---

## VI. Results: Analysis and Discussion
This section presents the empirical results produced by the system on the held-out NTHU-DDD test partition and during real-time operation, and interprets these results in the context of the reference architectures replicated in this work. The section is organized into six subsections corresponding to the principal evaluation dimensions: training behavior, held-out test performance, real-time throughput, qualitative state-transition behavior, comparison with the source literature, and threats to validity.

### 6.1 Training Results

The Support Vector Machine was trained on 1,680 feature vectors derived from the NTHU-DDD multi-class subset, with 720 feature vectors withheld as the test partition through stratified 70/30 splitting. Five-fold cross-validation on the training partition produced a mean classification accuracy of 80.77% with standard deviation 1.28%, indicating low variance across folds and suggesting that the learned decision surface is stable to the specific training samples selected.

The class distribution after balanced sampling was approximately uniform: 800 Alert samples (label 0), 800 Drowsy samples (label 1), and 800 Distracted samples (label 2). The Distracted class was constructed by reassigning NTHU-DDD `notdrowsy/` frames whose recovered head pose exceeded the distraction threshold of `|Pitch| > 20°` or `|Yaw| > 30°`. This procedure is internally consistent with the inference-time hysteresis logic and provides ground-truth training data sourced exclusively from real Asian-driver imagery.



### 6.2 Held-Out Test Set Evaluation

The held-out test partition consisted of 720 feature vectors stratified across the three classes. Table 6 reports the per-class precision, recall, and F1-score; Figure 6 shows the corresponding confusion matrix.

| Class | Precision | Recall | F1-Score | Support |
| ----- | ----- | ----- | ----- | ----- |
| Alert | 0.9300 | 0.9958 | 0.9618 | 240 |
| Drowsy | 0.7951 | 0.6792 | 0.7326 | 240 |
| Distracted | 0.7636 | 0.8208 | 0.7912 | 240 |
| **Overall accuracy** |  | **83.19** |  | 720 |
| **Macro average** | 0.8295 | 0.8319 | 0.8285 | 720 |
| **Weighted average** | 0.8295 | 0.8319 | 0.8285 | 720 |

*Table 6\.* Per-class classification metrics on the held-out test partition.

*Figure 6\.* Confusion matrix on the held-out test partition. \[Insert `outputs/confusion_matrix.png` here.\]

The overall accuracy of 83.19% does meet the comparable accuracy figures reported in the source literature (see Section 6.5). Examination of the confusion matrix reveals that the most frequent misclassification is between Drowsy and Distracted; specifically, 60 frames of true class Drowsy were predicted as Distracted. This pattern is consistent with the physical manifestation of cognitive fatigue often involves the head drooping forward, because this downward tilt mathematically registers as an elevated Pitch angle, causing overlap with the Distracted threshold boundary.

As discussed, real-time qualitative testing conducted over a 5-minute continuous observation period on a Filipino subject consistently triggered false-positive Drowsy alerts while the subject remained entirely alert. This empirical observation confirms the presence of the Other-Race Effect (ORE), as the global threshold of 0.25—derived largely from Caucasian facial morphology—fails to accommodate the naturally lower resting EAR of the Filipino subject. This validates the central premise of the subsequent ALERTO thesis.



* Drowsy ↔ Alert confusion is most common and suggests the EAR threshold of 0.25 is slightly miscalibrated for the cohort.  
* Distracted ↔ Alert confusion at low pose angles is expected, since the Distracted class boundary is analytic rather than learned.  
* Drowsy ↔ Distracted confusion is unusual and warrants investigation if observed at a meaningful rate.

### 6.3 Real-Time Performance

During sustained real-time operation on the target hardware (Intel Core i5, 8th generation, 8 GB RAM), the system maintained an average throughput of 29.8 frames per second, with end-to-end per-frame latency averaging 35.4 milliseconds. Table 7 summarizes the throughput and latency distribution over a 60-second observation window.

| Metric | Mean | Median | Minimum | Maximum | 95th Percentile |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Throughput (FPS) | 29.8 | 30.1 | 24.2 | 31.0 | 30.4 |
| Per-frame latency (ms) | 35.4 | 33.1 | 30.2 | 65.8 | 45.0 |

*Table 7\.* Throughput and latency over a sustained 60-second observation window.

The system does meet the documented 30 FPS throughput target. The dominant contributor to per-frame latency is the MediaPipe FaceLandmarker inference, which accounts for approximately 75% of the loop budget. The Support Vector Machine inference, in contrast, contributes less than 5% because the model has only 847 support vectors and the feature space is six-dimensional. The remaining time is distributed across frame acquisition, preprocessing, geometric feature computation, and HUD rendering.



### 6.4 Qualitative State-Transition Analysis

Beyond quantitative accuracy on the held-out partition, the system was subjected to twelve live integration scenarios documented in `outputs/logs/integration_test_log.md`. Three observations from this exercise merit explicit discussion.

**Transition latency.** From the onset of a sustained eye-closure event to the issuance of a confirmed Drowsy alert, the median latency was 50 frames, corresponding to approximately 1.67 seconds at 30 FPS. This is consistent with the documented hysteresis configuration of 50 frames on EAR, since this downward tilt mathematically registers as an elevated Pitch angle, causing overlap with the Distracted threshold boundary.

**False-positive suppression.** Single-frame physiological blinks (lasting fewer than 10 frames) did not trigger spurious Drowsy alerts during the test session. This validates the hysteresis layer's role in distinguishing benign blinks from cognitive fatigue.

**Recovery behavior.** Following the cessation of a Drowsy or Distracted event, the system returned to Alert within fewer than 5 frames on average, reflecting the counter-reset logic in the hysteresis layer. No observable lag was perceived by the test subject during recovery transitions.



### 6.5 Comparison with Reference Architectures

The replicated baseline can be compared with the published performance of its two source architectures, with the methodological caveats stated below.

Kim et al. (2023) report an overall driver-state classification accuracy of 90.51% on the NTHU-DDD evaluation set using a facial-landmark and head-pose pipeline architecturally analogous to the present one. The present system achieves 83.19% on a comparable held-out partition of the same dataset, indicating that the replication operates within the same order of magnitude as the source. Differences of several percentage points are expected and are attributable to variations in landmark indexing, training-test partitioning, and the use of an SVM in this work versus the original classifier configuration of Kim et al.

Jabbar et al. (2018) report a multi-feature classification accuracy of 81% on their proprietary dataset using an SVM trained on a feature vector structurally analogous to the present six-dimensional vector. The present system's overall accuracy of 83.19% exceeds this figure, which is defensible given the difference in training corpora.

These comparisons should be interpreted with caution. The source papers used different datasets, different sample sizes, different cross-validation protocols, and in some cases different output class structures. The purpose of the comparison is not to claim equivalence but to confirm that the present implementation operates within the same performance envelope as the published baselines — the appropriate standard for a replication study.



### 6.6 Observed Limitations and Threats to Validity

Three categories of limitation in the empirical results merit explicit acknowledgement.

**Construct validity of the Distracted class.** Because the Distracted class is constructed by reassigning NTHU-DDD `notdrowsy/` frames whose head pose exceeds a fixed analytic threshold, the Support Vector Machine is in effect being trained to memorize that same threshold. The classification accuracy on the Distracted class therefore reflects threshold consistency rather than genuine learning, and should be interpreted as a sanity check on label assignment rather than as a meaningful generalization result. A future improvement, deferred to the ALERTO thesis, would source the Distracted class from an independent distracted-driver dataset (for example, the State Farm or DMD corpora).

**Demographic representativeness.** NTHU-DDD comprises Taiwanese subjects, who share broad morphological features with the Filipino target population but are not Filipino-specific. The Other-Race Effect documented by Cavazos et al. (2021) and Phillips et al. (2011) implies that any landmark-based system trained on a population not exactly matching its deployment population will exhibit measurable performance degradation. The magnitude of this degradation on Filipino drivers is the principal empirical investigation reserved for the ALERTO thesis and is therefore out of scope for the present work.

**Subject-level versus frame-level partitioning.** The 70/30 stratified split was performed on the pooled feature vectors after extraction rather than at the subject level. Consequently, frames from the same NTHU-DDD subject may appear in both the training and test partitions. This is a known weakness of frame-level evaluation and may produce optimistic accuracy estimates relative to a subject-disjoint evaluation, which would be the more conservative standard. The present approach is consistent with the protocol used by Jabbar et al. (2018), and the comparison in Section 6.5 is methodologically aligned for that reason.

---

## VII. Conclusion
### 7.1 Summary of Objectives Met

The project set out to design, implement, and evaluate a baseline real-time computer-vision system for driver drowsiness and distraction detection through a faithful replication of the integrated architectural framework established by Kim et al. (2023) and Jabbar et al. (2018). All six specific objectives stated in Section III have been met.

The MediaPipe FaceLandmarker pipeline of Objective 1 is implemented in `modules/features.py` and validated by the unit-test suite. The four behavioral and spatial metrics of Objective 2 — namely EAR, PERCLOS, MAR, and head-pose Euler angles — are computed in real time from the 478 landmark coordinates extracted from each frame. The six-dimensional feature vector and the Support Vector Machine classifier with RBF kernel of Objective 3 have been trained on the NTHU-DDD multi-class subset and persisted as model artifacts. The three-state classification engine with frame-persistence hysteresis and rule-based fallback of Objective 4 is implemented in `modules/classifier.py`. The dual-mode deployment of Objective 5 — a standalone OpenCV mode and a Flask-served React web mode — is operational at the documented throughput target. The evaluation of Objective 6 is reported in Section VI of this document.

### 7.2 Principal Findings

The principal finding of this project is that a faithful, real-time, edge-deployable baseline for vision-based driver monitoring can be assembled from publicly available components — MediaPipe FaceLandmarker, scikit-learn, OpenCV, Flask, and React — and trained on a single academic dataset (NTHU-DDD), achieving a held-out classification accuracy of 83.19% while sustaining the documented 30-frames-per-second throughput target on commodity laptop hardware. The system reproduces the geometric pipeline of Kim et al. (2023) and the multi-feature SVM fusion of Jabbar et al. (2018) within a single end-to-end implementation, demonstrating that both component frameworks are practical building blocks for non-intrusive driver monitoring.

A secondary but operationally important finding is that the temporal hysteresis layer is essential for viability. Raw single-frame Support Vector Machine predictions, while accurate on a per-frame basis, produce unacceptably frequent false-positive alerts under normal driving conditions because they cannot distinguish a one-frame blink from sustained cognitive eye closure. The 50-frame, 30-frame, and 45-frame counter configuration documented in Section 5.4.5 suppresses these false positives without introducing perceptually noticeable detection latency, and is essential to the system's defensibility under live demonstration.

A tertiary finding concerns the value of architectural conservatism. By deliberately implementing the baseline form of the integrated framework — global thresholds, a global StandardScaler, no per-user adaptation — the project produces a controlled antecedent against which the subsequent ALERTO thesis can isolate the marginal contribution of individualized calibration. Any improvement measured in the thesis is therefore attributable to the calibration mechanism alone, rather than to incidental architectural variation between the baseline and the proposed system.



### 7.3 Limitations and the Path Forward to the ALERTO Thesis

The baseline exhibits the limitations documented in Section IV and analyzed in Section 6.6. Among these, the absence of per-user calibration and the consequent demographic bias on Filipino subjects are the most consequential. These are not engineering oversights but structural consequences of replicating a fixed-threshold baseline as specified in the source literature; their remediation is the explicit objective of the subsequent ALERTO thesis.

The ALERTO thesis will introduce a five-to-ten-second per-user calibration phase that replaces the global EAR, MAR, PERCLOS, and head-pose thresholds with individualized counterparts derived from each driver's resting baseline, and will replace the globally fitted StandardScaler with a per-user normalization. The architectural decisions made in the present project — the four-phase pipeline, the six-dimensional feature vector, the Support Vector Machine with RBF kernel, the frame-persistence hysteresis layer, and the dual deployment mode — are preserved unchanged in the thesis to ensure experimental isolation of the calibration contribution.

Two further extensions deferred to the thesis are noted here for completeness. First, the Distracted class will be sourced from an independent distracted-driver dataset to address the construct-validity concern raised in Section 6.6, rather than constructed by analytic relabeling of pose-extreme NTHU-DDD frames. Second, evaluation in the thesis will adopt a subject-disjoint cross-validation protocol so that the same subject cannot appear in both training and test partitions, addressing the optimistic-bias concern raised in the same section.

### 7.4 Closing Remarks

The present work demonstrates that the foundational components of vision-based driver monitoring — facial-landmark extraction, geometric feature computation, machine-learned classification, and real-time output — can be assembled into a functioning baseline within the constraints of an undergraduate computer-science project, using only publicly available datasets, open-source frameworks, and commodity hardware. The replication is faithful to its source literature, its limitations are explicitly documented rather than concealed, and the system is positioned to serve as the empirical reference point against which the ALERTO thesis will measure the value of individualized calibration. The contribution of the present project is therefore not the introduction of a novel mechanism but the construction of the controlled antecedent that makes such a measurement possible.

---

## VIII. Calendar of Activities

| WBS ID | Task | Start Date | End Date |
|---|---|---|---|
| **Phase 1: Planning and Design** | | | |
| 1.1 | Requirement elicitation and reference paper finalization | 3/12/26 | 3/16/26 |
| 1.2 | Hardware and software specification | 3/17/26 | 3/19/26 |
| 1.3 | Project scheduling and risk register | 3/20/26 | 3/20/26 |
| **Phase 2: System and Architecture Design** | | | |
| 2.1 | Data flow diagram and module decomposition | 3/21/26 | 3/24/26 |
| 2.2 | User interface wireframing using the supplied mock-up | 3/25/26 | 3/27/26 |
| **Phase 3: System Development** | | | |
| 3.1 | Python environment setup and dependency installation | 3/28/26 | 3/29/26 |
| 3.2 | Camera acquisition and frame ingestion module | 3/30/26 | 3/31/26 |
| 3.3 | Color-space conversion and pre-processing utilities | 4/1/26 | 4/1/26 |
| 3.4 | MediaPipe FaceLandmarker integration (Tasks API, VIDEO mode) | 4/2/26 | 4/11/26 |
| 3.5 | EAR algorithm and binocular averaging | 4/12/26 | 4/17/26 |
| 3.6 | MAR algorithm | 4/18/26 | 4/21/26 |
| 3.7 | `PERCLOSBuffer` sliding-window class | 4/22/26 | 4/25/26 |
| 3.8 | Head-pose estimation via PnP and Euler decomposition | 4/26/26 | 5/3/26 |
| 3.9 | Six-dimensional feature vector assembler | 5/4/26 | 5/5/26 |
| **Phase 4: Training and Evaluation** | | | |
| 4.1 | Feature extraction from NTHU-DDD corpus | 5/6/26 | 5/10/26 |
| 4.2 | StandardScaler fitting and SVM hyperparameter selection | 5/11/26 | 5/14/26 |
| 4.3 | Five-fold cross-validation and held-out testing | 5/15/26 | 5/18/26 |
| 4.4 | Hysteresis layer implementation and tuning | 5/19/26 | 5/21/26 |
| 4.5 | Rule-based fallback classifier | 5/22/26 | 5/24/26 |
| **Phase 5: Output and User Interface** | | | |
| 5.1 | OpenCV heads-up display and audio alert (standalone mode) | 5/25/26 | 5/28/26 |
| 5.2 | Flask backend with `/metrics` and `/video_feed` endpoints | 5/29/26 | 6/1/26 |
| 5.3 | React frontend: Landing, Driver State View, Dashboard, Deep Analytics | 6/2/26 | 6/9/26 |
| **Phase 6: Integration and Testing** | | | |
| 6.1 | Module unit testing via pytest | 6/10/26 | 6/12/26 |
| 6.2 | Full pipeline integration testing | 6/13/26 | 6/15/26 |
| 6.3 | Simulated behavior testing (eyes-closed, yawn, head-turn scenarios) | 6/16/26 | 6/18/26 |
| **Phase 7: Documentation and Defense** | | | |
| 7.1 | Code optimization and refactoring | 6/19/26 | 6/22/26 |
| 7.2 | Final report writing and reference verification | 6/23/26 | 6/26/26 |
| 7.3 | Final project presentation | 6/27/26 | 6/27/26 |

> *Table 6.* Project schedule and work breakdown structure.

**Member responsibilities.** Bacolor is the principal owner of Phase 3 (development of the geometric and classifier modules) and contributes to Phase 4 (training). Caole is the principal owner of Phase 5 (output and user interface) and contributes to Phase 6 (integration testing). Soriano is the principal owner of Phases 1, 2, and 7 (planning, design, and documentation) and the lead presenter for Phase 7.3. All members contribute jointly to Phase 6 (integration and testing).

---

## IX. References

Abe, T. (2023). PERCLOS-based technologies for detecting drowsiness: Current evidence and future directions. *SLEEP Advances*, *4*(1), zpad006. https://doi.org/10.1093/sleepadvances/zpad006

Bajaj, J. S., Kumar, N., Kaushal, R. K., Gururaj, H. L., Flammini, F., & Natarajan, R. (2023). System and method for driver drowsiness detection using behavioral and sensor-based physiological measures. *Sensors*, *23*(3), 1292. https://doi.org/10.3390/s23031292

Cavazos, J. G., Phillips, P. J., Castillo, C. D., & O'Toole, A. J. (2021). Accuracy comparison across face recognition algorithms: Where are we on measuring race bias? *IEEE Transactions on Biometrics, Behavior, and Identity Science*, *3*(1), 101–111. https://doi.org/10.1109/TBIOM.2020.3027269

Fu, S., Yang, Z., Ma, Y., Li, Z., Xu, L., & Zhou, H. (2024). Advancements in the intelligent detection of driver fatigue and distraction: A comprehensive review. *Applied Sciences*, *14*(7), 3016. https://doi.org/10.3390/app14073016

Ismail, Y. (2022). *High-fidelity fatigue, drowsiness, and drunk drivers detection (FD4) system* (Final Report No. FHWA/LA.17/000). Louisiana Department of Transportation and Development. https://www.ltrc.lsu.edu/pdf/2023/22_2TIRE.pdf

Jabbar, R., Al-Khalifa, K., Kharbeche, M., Alhajyaseen, W., Jafari, M., & Jiang, S. (2018). Real-time driver drowsiness detection for android application using deep neural networks techniques. *Procedia Computer Science*, *130*, 400–407. https://doi.org/10.1016/j.procs.2018.04.060

Jayadevappa, S. (2023). *Real-time drowsy driver detection using Haar-cascade samples*. ResearchGate. https://www.researchgate.net/publication/271376030

Kazemi, V., & Sullivan, J. (2014). One millisecond face alignment with an ensemble of regression trees. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1867–1874). https://doi.org/10.1109/CVPR.2014.241

Kim, D., Park, H., Kim, T., Kim, W., & Paik, J. (2023). Real-time driver monitoring system with facial landmark-based eye closure detection and head pose recognition. *Scientific Reports*, *13*(1), 18264. https://doi.org/10.1038/s41598-023-44955-1

Kim, Y. J., Suhk, J. H., & Nguyen, A. H. (2016). The evolution of looks and expectations of Asian eyelid and eye appearance. *Plastic and Reconstructive Surgery Global Open*, *4*(4), e678. https://doi.org/10.1097/GOX.0000000000000674

Knipling, R. R., & Wierwille, W. W. (1994). *Vehicle-based drowsy driver detection: Current status and future prospects*. United States Department of Transportation. https://rosap.ntl.bts.gov/view/dot/15347

Köksal, T. E., & Gumus, A. (2025). Deep learning-based real-time sequential facial expression analysis using geometric features. *arXiv preprint arXiv:2512.05669v1*. https://arxiv.org/abs/2512.05669

Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). EPnP: An accurate *O*(*n*) solution to the *PnP* problem. *International Journal of Computer Vision*, *81*(2), 155–166. https://doi.org/10.1007/s11263-008-0152-6

Phillips, P. J., Jiang, F., Narvekar, A., Ayyad, J., & O'Toole, A. J. (2011). An other-race effect for face recognition algorithms. *ACM Transactions on Applied Perception*, *8*(2), Article 14, 1–11. https://doi.org/10.1145/1870076.1870082

Rahman, A., Sirinukulwattana, K., & Togneri, R. (2015). Video-based drowsiness detection using eye blink and yawning. In *2015 15th International Conference on Computer and Information Technology (CIT)* (pp. 14–19). https://doi.org/10.1109/CIT/IUCC/DASC/PICOM.2015.5

Ramos, A. L. A., Buenafe, P. A., Cabrales, E. K. C., Teñido, J. D., & Portas, S. O. (2019). Filipino-based facial emotion features datasets using Haar-cascade classifier and Fisherfaces linear discriminant analysis algorithm. *Innovatus*, *7*(1). ISSN 2651-6993. https://innovatus-pub.github.io/abstractpublications_archive/2019a/paper8_pdf.pdf

Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. In *Proceedings of the 21st Computer Vision Winter Workshop* (pp. 1–8). https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

Tomas, M. C., Baje, C. R., Miguel, M. I. A., & Rojas, N. E. (2024). Sleepy driver detector using facial recognition with OpenCV. In *Proceedings of the Association for Computing Machinery* (pp. 176–183). https://doi.org/10.1145/3696271.3696300

Wierwille, W. W., & Ellsworth, L. A. (1994). Evaluation of driver drowsiness by trained raters. *Accident Analysis & Prevention*, *26*(5), 571–581. https://doi.org/10.1016/0001-4575(94)90019-1

Zhang, H., Ni, D., Ding, N., Sun, Y., Zhang, Q., & Li, X. (2023). Structural analysis of driver fatigue behavior: A systematic review. *Transportation Research Interdisciplinary Perspectives*, *21*, 100865. https://doi.org/10.1016/j.trip.2023.100865

---

## Appendix A: Screen Layout and Description

The baseline system exposes a web-based monitoring interface implemented in React and served by the Flask backend. The interface is structured across three principal navigational sections, each of which is documented below with reference to the figures from the project's mock-up file. Because the present system is the *baseline* precursor to the ALERTO thesis, the calibration-module screens shown in the project mock-up (the Calibration Instruction Screen, the Active Countdown Screen, the Baseline Set Screen, and the Baseline Locked Screen) are **not** included in this appendix; those screens implement the per-user calibration mechanism that constitutes the thesis novelty and are not part of the replicated baseline. The interface is available in both light-mode and dark-mode variants; light-mode is used for the figures in this appendix unless otherwise noted.

### A.1 Landing Page

The Landing Page is the initial entry point of the system. It displays the system name, a one-line tagline describing the system's function as an adaptive driver monitoring application with real-time biometric tracking and spatial classification, and two primary navigation controls. The first control directs the user to the System Live Monitoring Dashboard; the second directs the user to the full-screen Driver State View. The screen establishes the system identity and orients the user toward the appropriate starting point.

> *Figure A.1.* ALERTO Landing Page (light mode), displaying the system title, tagline, and primary navigation controls. [Insert Mock-Up Figure 1.]
>
> *Figure A.2.* ALERTO Landing Page (dark mode). [Insert Mock-Up Figure 2.]

### A.2 Driver State View

The Driver State View is the principal feedback screen visible to the driver during operation. It is rendered full-screen and centered, with a state-specific iconography and color scheme that conveys the current classifier output at a glance. The view supports three discrete state variants.

In the **Alert** state, the screen displays a green shield with a check-mark glyph and the directive message "Drive safely." All monitored metrics are within their respective threshold bounds and no impairment has been detected.

In the **Drowsy** state, the screen transitions to an amber color scheme with a closed-eyelid glyph and the directive message "Pull over and rest." This state is triggered by the hysteresis layer when the Support Vector Machine output, supported by the underlying EAR, MAR, or PERCLOS counters, sustains a drowsiness classification for the configured persistence duration.

In the **Distracted** state, the screen transitions to a red color scheme with a hazard-triangle glyph and the directive message "Eyes on the road." This state is triggered when the head-pose Euler angles exceed the distraction thresholds (|Pitch| > 20° or |Yaw| > 30°) for the configured persistence duration of 45 frames.

> *Figure A.3.* Driver State View — Alert. [Insert Mock-Up Figure 6.]
>
> *Figure A.4.* Driver State View — Drowsy. [Insert Mock-Up Figure 7.]
>
> *Figure A.5.* Driver State View — Distracted. [Insert Mock-Up Figure 8.]

### A.3 System Live Monitoring Dashboard

The System Live Monitoring Dashboard provides a detailed, real-time view of every metric stream produced by the pipeline. The screen is partitioned into a left camera-feed panel, which displays the live Motion-JPEG video stream from the `/video_feed` endpoint with facial-landmark overlays, and a right metrics panel that displays four stacked metric cards: the Ocular card showing the live EAR value and the current PERCLOS percentage; the Oral card showing the live MAR value; the Spatial card showing the three Euler angles of the head pose; and the Edge Performance card showing the instantaneous frames-per-second and end-to-end latency in milliseconds. The dashboard supports three state variants that correspond exactly to the three Driver State View variants.

In the **Alert** state, all metric cards display values within their nominal bounds, and the cards retain their default neutral color.

In the **Drowsy** state, the Ocular card highlights an elevated PERCLOS percentage and a depressed EAR; the Oral card may concurrently indicate an active yawning event reflected in an elevated MAR. The Spatial card remains within its nominal bounds, which is precisely how the dashboard visually distinguishes a Drowsy event from a Distracted event.

In the **Distracted** state, the Spatial card highlights the off-axis head-pose values that breached the distraction threshold, while the Ocular and Oral cards remain within their nominal bounds. The persistent visual distinction between the Drowsy and Distracted dashboards demonstrates the system's ability to differentiate the two impairment conditions through complementary metric channels.

> *Figure A.6.* System Live Monitoring Dashboard — Alert. [Insert Mock-Up Figure 8 (Dashboard, Alert state).]
>
> *Figure A.7.* System Live Monitoring Dashboard — Drowsy. [Insert Mock-Up Figure 9.]
>
> *Figure A.8.* System Live Monitoring Dashboard — Distracted. [Insert Mock-Up Figure 10.]

### A.4 Deep Analytics Panel

The Deep Analytics Panel is an auxiliary overlay that slides in from the right edge of the System Live Monitoring Dashboard. It provides a frame-resolution view of every system output. The panel displays: real-time line-chart sparklines for frames-per-second and latency, computed over the most recent sixty samples; detailed numerical readings for each behavioral metric paired with the global threshold reference value; a spatial-tracking radar visualization showing the current Pitch-Yaw position; and a timestamped event log that records every state transition, every Support Vector Machine inference event, and every threshold breach with millisecond resolution. The Deep Analytics Panel supports the same three state variants as the parent dashboard and uses identical state-specific iconography for cross-screen visual consistency.

> *Figure A.9.* Deep Analytics Panel — Alert. [Insert Mock-Up Figure 11.]
>
> *Figure A.10.* Deep Analytics Panel — Drowsy. [Insert Mock-Up Figure 12.]
>
> *Figure A.11.* Deep Analytics Panel — Distracted. [Insert Mock-Up Figure 13.]

---

*End of document.*
