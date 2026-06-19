# modules/classifier.py — Phase 3: Classification Engine
# =============================================================================
# STUDY REFERENCES:
#   Jabbar, R., et al. (2018). Real-time driver drowsiness detection for
#   embedded system using model compression of deep neural networks.
#   Procedia Computer Science, 130, 736–743.
#
#   Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). A realistic dataset
#   and baseline temporal model for early drowsiness detection. CVPR Workshops.
# =============================================================================

import os
import pickle
import numpy as np
from collections import deque

from config import (
    SVM_MODEL_PATH, SVM_SCALER_PATH,
    EAR_CONSECUTIVE_FRAMES, MAR_CONSECUTIVE_FRAMES, DISTRACTION_FRAMES,
    EAR_CLOSED_THRESHOLD, MAR_YAWN_THRESHOLD,
    PITCH_THRESH_LOW, PITCH_THRESH_HIGH, YAW_THRESHOLD,
    PERCLOS_THRESHOLD, PERCLOS_WINDOW_FRAMES, LABEL_MAP,
)


# ── Phase 3A: SVM Classifier ──────────────────────────────────────────────────

class SVMClassifier:
    """
    Phase 3 — SVM-based 3-class driver state classifier.

    Machine Learning: Support Vector Machine (SVM)
      Kernel  : Radial Basis Function (RBF)
        RBF maps features into a high-dimensional space so the SVM can
        find non-linear decision boundaries between overlapping EAR/MAR/
        head-pose distributions across different subjects.
      C       : 1.0 (regularisation; globally fixed, not tuned per user)
      gamma   : 'scale' = 1 / (n_features × X.var())
      Classes : 0=ALERT, 1=DROWSY, 2=DISTRACTED (One-vs-Rest)
      Scaler  : sklearn StandardScaler — global mean/std from training set,
                NOT per-user. This is the key architectural limitation vs.
                the ALERTO thesis (which uses a per-user calibrated scaler).

    Frame-persistence hysteresis:
      The raw SVM prediction must be confirmed by raw metric counters for N
      consecutive frames before a state change is emitted, reducing flicker.
    """

    def __init__(self):
        self.model   = None
        self.scaler  = None
        self._loaded = False

        # Frame-persistence counters for hysteresis
        self._ear_cnt  = 0
        self._mar_cnt  = 0
        self._pose_cnt = 0

    # ── Load / Save ───────────────────────────────────────────────────────

    def load(self, model_path=SVM_MODEL_PATH, scaler_path=SVM_SCALER_PATH):
        """Load a pre-trained SVM model and global StandardScaler from disk."""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"SVM model not found at '{model_path}'.\n"
                "Run:  py training/train_svm.py  to generate it first."
            )
        with open(model_path,  "rb") as f:
            self.model  = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self._loaded = True
        print(f"[SVM] Model loaded: {model_path}")

    def save(self, model_path=SVM_MODEL_PATH, scaler_path=SVM_SCALER_PATH):
        """Persist trained SVM and scaler to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path,  "wb") as f:
            pickle.dump(self.model,  f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[SVM] Saved → {model_path}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, feature_vec, ear, mar, pitch, yaw):
        """
        Classify driver state from a 6-dimensional feature vector.

        Parameters
        ----------
        feature_vec : np.ndarray shape (6,) — [ear, perclos, mar, p, y, r]
        ear, mar    : raw metric values (used for hysteresis counters)
        pitch, yaw  : raw Euler angles  (used for hysteresis counters)

        Returns
        -------
        state  : str  — "ALERT", "DROWSY", or "DISTRACTED"
        proba  : dict — class probabilities keyed by label string, or None
        """
        if not self.is_loaded:
            raise RuntimeError("SVM not loaded. Call .load() first.")

        # Scale using the GLOBAL scaler (population mean/std, NOT per-user)
        X_scaled = self.scaler.transform(feature_vec.reshape(1, -1))

        # SVM raw prediction
        pred_int  = int(self.model.predict(X_scaled)[0])
        raw_state = LABEL_MAP.get(pred_int, "ALERT")

        # Class probabilities (requires SVC(probability=True))
        try:
            proba_arr = self.model.predict_proba(X_scaled)[0]
            proba = {LABEL_MAP[i]: float(p) for i, p in enumerate(proba_arr)}
        except AttributeError:
            proba = None

        # Apply frame-persistence hysteresis
        state = self._apply_hysteresis(raw_state, ear, mar, pitch, yaw)
        return state, proba

    def _apply_hysteresis(self, raw_state, ear, mar, pitch, yaw):
        """
        Require N consecutive SVM agreements + raw-metric confirmation before
        emitting a state change. Prevents single-frame false alarms.
        """
        # Update raw-metric counters
        self._ear_cnt  = self._ear_cnt  + 1 if ear < EAR_CLOSED_THRESHOLD else 0
        self._mar_cnt  = self._mar_cnt  + 1 if mar > MAR_YAWN_THRESHOLD    else 0

        if pitch is not None and yaw is not None:
            off_axis = (
                pitch < PITCH_THRESH_LOW or
                pitch > PITCH_THRESH_HIGH or
                abs(yaw) > YAW_THRESHOLD
            )
            self._pose_cnt = self._pose_cnt + 1 if off_axis else 0

        # Suppress DROWSY if raw metrics are not yet sustained
        if raw_state == "DROWSY" and (
            self._ear_cnt < EAR_CONSECUTIVE_FRAMES and
            self._mar_cnt < MAR_CONSECUTIVE_FRAMES
        ):
            return "ALERT"

        # Suppress DISTRACTED if head pose is not yet sustained
        if raw_state == "DISTRACTED" and self._pose_cnt < DISTRACTION_FRAMES:
            return "ALERT"

        return raw_state

    def reset(self):
        """Reset all frame-persistence counters."""
        self._ear_cnt  = 0
        self._mar_cnt  = 0
        self._pose_cnt = 0


# ── Phase 3B: Rule-Based Classifier (Fallback) ───────────────────────────────

class RuleBasedClassifier:
    """
    Pure threshold-based classifier — no ML, no training required.

    Based on Soukupová & Čech (2016) for EAR/PERCLOS, extended with
    MAR from Rahman et al. (2015) and head pose from Kim et al. (2023).

    Used as:
      - Fallback when SVM model has not been trained yet
      - Reference comparison for the lower bound of classification performance
      - Default mode: python main.py --mode rule

    Decision priority: DROWSY > DISTRACTED > ALERT
    """

    def __init__(self):
        self._ear_cnt  = 0
        self._mar_cnt  = 0
        self._pose_cnt = 0

    def classify(self, ear, perclos, mar, pitch, yaw):
        """
        Apply IF-THEN threshold rules to classify the current driver state.

        Parameters
        ----------
        ear, perclos, mar : float — ocular and oral feature values
        pitch, yaw        : float or None — head-pose Euler angles

        Returns
        -------
        str — "ALERT", "DROWSY", or "DISTRACTED"
        """
        # Update frame-persistence counters
        self._ear_cnt  = self._ear_cnt  + 1 if ear < EAR_CLOSED_THRESHOLD else 0
        self._mar_cnt  = self._mar_cnt  + 1 if mar > MAR_YAWN_THRESHOLD    else 0

        if pitch is not None and yaw is not None:
            off_axis = (
                pitch < PITCH_THRESH_LOW or
                pitch > PITCH_THRESH_HIGH or
                abs(yaw) > YAW_THRESHOLD
            )
            self._pose_cnt = self._pose_cnt + 1 if off_axis else 0

        # Priority 1: DROWSY (sustained eye closure, yawning, or PERCLOS)
        if (self._ear_cnt  >= EAR_CONSECUTIVE_FRAMES or
                self._mar_cnt  >= MAR_CONSECUTIVE_FRAMES or
                perclos        >  PERCLOS_THRESHOLD):
            return "DROWSY"

        # Priority 2: DISTRACTED (sustained head pose deviation)
        if self._pose_cnt >= DISTRACTION_FRAMES:
            return "DISTRACTED"

        return "ALERT"

    def reset(self):
        """Reset all frame-persistence counters."""
        self._ear_cnt  = 0
        self._mar_cnt  = 0
        self._pose_cnt = 0
