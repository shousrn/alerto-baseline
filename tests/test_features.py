"""
=============================================================================
Unit Tests — ALERTO Baseline: modules/features.py & modules/classifier.py
=============================================================================
Tests cover:
    - compute_ear()          Eye Aspect Ratio (Soukupová & Čech, 2016)
    - compute_mar()          Mouth Aspect Ratio (Rahman et al., 2015)
    - PERCLOSBuffer          Sliding-window eye closure tracker
    - extract_feature_vector 6D feature vector assembly
    - RuleBasedClassifier    IF-THEN 3-state classification
    - SVMClassifier          Load/predict interface (mocked)

Run with:  py -m pytest tests/ -v
=============================================================================
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path so config.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.features import (
    compute_ear,
    compute_avg_ear,
    compute_mar,
    PERCLOSBuffer,
    extract_feature_vector,
)
from modules.classifier import SVMClassifier, RuleBasedClassifier


# ─── Landmark mock helpers ────────────────────────────────────────────────────

def _lm(x, y, z=0.0):
    """Create a mock MediaPipe landmark object."""
    m = MagicMock()
    m.x, m.y, m.z = x, y, z
    return m


def _make_lm_list(size=478):
    """Create a list of zeroed mock landmarks."""
    return [_lm(0.0, 0.0) for _ in range(size)]


def _set_eye(lms, idx, ear_value, base_x, base_y, width=0.10):
    """
    Place 6 eye landmarks (p1…p6) at normalised positions that produce
    the requested EAR value when measured on a 100×100 px frame.
    EAR = vert / (2 × horiz) → vert = ear_value * 2 * horiz
    horiz = width * 100 px → vert_px = ear_value * 2 * width * 100
    → half_vert (normalised) = ear_value * width
    """
    hv = ear_value * width   # half-vertical offset (normalised)
    lms[idx["p1"]] = _lm(base_x,         base_y)
    lms[idx["p2"]] = _lm(base_x + 0.25 * width, base_y - hv)
    lms[idx["p3"]] = _lm(base_x + 0.75 * width, base_y - hv)
    lms[idx["p4"]] = _lm(base_x + width,  base_y)
    lms[idx["p5"]] = _lm(base_x + 0.75 * width, base_y + hv)
    lms[idx["p6"]] = _lm(base_x + 0.25 * width, base_y + hv)


# ─── EAR Tests ────────────────────────────────────────────────────────────────

class TestComputeEAR:

    def test_open_eye_returns_high_ear(self):
        from config import RIGHT_EYE_IDX
        lms = _make_lm_list()
        _set_eye(lms, RIGHT_EYE_IDX, ear_value=0.30, base_x=0.40, base_y=0.50)
        ear = compute_ear(lms, RIGHT_EYE_IDX, h=100, w=100)
        assert ear > 0.25, f"Open eye EAR should be > 0.25, got {ear:.4f}"

    def test_closed_eye_returns_low_ear(self):
        from config import RIGHT_EYE_IDX
        lms = _make_lm_list()
        _set_eye(lms, RIGHT_EYE_IDX, ear_value=0.05, base_x=0.40, base_y=0.50)
        ear = compute_ear(lms, RIGHT_EYE_IDX, h=100, w=100)
        assert ear < 0.12, f"Closed eye EAR should be < 0.12, got {ear:.4f}"

    def test_degenerate_zero_horizontal_returns_zero(self):
        from config import RIGHT_EYE_IDX
        lms = _make_lm_list()
        # p1 == p4 → zero horizontal distance
        for k in RIGHT_EYE_IDX.values():
            lms[k] = _lm(0.5, 0.5)
        ear = compute_ear(lms, RIGHT_EYE_IDX, h=100, w=100)
        assert ear == 0.0

    def test_avg_ear_averages_both_eyes(self):
        from config import RIGHT_EYE_IDX, LEFT_EYE_IDX
        lms = _make_lm_list()
        _set_eye(lms, RIGHT_EYE_IDX, 0.30, base_x=0.30, base_y=0.50)
        _set_eye(lms, LEFT_EYE_IDX,  0.30, base_x=0.60, base_y=0.50)
        avg = compute_avg_ear(lms, h=100, w=100)
        assert avg > 0.20


# ─── MAR Tests ────────────────────────────────────────────────────────────────

class TestComputeMAR:

    def _make_mouth_lms(self, mar_value, cx=0.50, cy=0.70, width=0.15):
        """Place MOUTH_IDX landmarks producing approximately `mar_value`."""
        from config import MOUTH_IDX
        lms = _make_lm_list()
        # Vertical half-offset:  v = mar_value * 2/3 * width
        hv = mar_value * (2.0 / 3.0) * width / 2.0
        lms[MOUTH_IDX["left_corner"]]   = _lm(cx - width / 2, cy)
        lms[MOUTH_IDX["right_corner"]]  = _lm(cx + width / 2, cy)
        lms[MOUTH_IDX["upper_center"]]  = _lm(cx, cy - hv)
        lms[MOUTH_IDX["lower_center"]]  = _lm(cx, cy + hv)
        lms[MOUTH_IDX["upper_left"]]    = _lm(cx - width / 4, cy - hv)
        lms[MOUTH_IDX["lower_left"]]    = _lm(cx - width / 4, cy + hv)
        lms[MOUTH_IDX["upper_right"]]   = _lm(cx + width / 4, cy - hv)
        lms[MOUTH_IDX["lower_right"]]   = _lm(cx + width / 4, cy + hv)
        return lms

    def test_closed_mouth_returns_low_mar(self):
        lms = self._make_mouth_lms(0.15)
        mar = compute_mar(lms, h=100, w=100)
        assert mar < 0.35, f"Closed mouth MAR should be < 0.35, got {mar:.4f}"

    def test_open_mouth_returns_high_mar(self):
        lms = self._make_mouth_lms(1.20)
        mar = compute_mar(lms, h=100, w=100)
        assert mar > 0.60, f"Yawning MAR should be > 0.60, got {mar:.4f}"

    def test_degenerate_zero_width_returns_zero(self):
        from config import MOUTH_IDX
        lms = _make_lm_list()
        for k in MOUTH_IDX.values():
            lms[k] = _lm(0.5, 0.5)
        mar = compute_mar(lms, h=100, w=100)
        assert mar == 0.0


# ─── PERCLOS Tests ────────────────────────────────────────────────────────────

class TestPERCLOSBuffer:

    def test_initial_state_not_ready(self):
        buf = PERCLOSBuffer(window_size=10, threshold=0.25)
        assert not buf.ready

    def test_all_open_eyes_gives_zero_perclos(self):
        buf = PERCLOSBuffer(window_size=5, threshold=0.25)
        for _ in range(5):
            result = buf.update(0.35)
        assert result == 0.0
        assert buf.ready

    def test_all_closed_eyes_gives_100_perclos(self):
        buf = PERCLOSBuffer(window_size=5, threshold=0.25)
        for _ in range(5):
            result = buf.update(0.10)
        assert result == 100.0

    def test_half_closed_gives_50_perclos(self):
        buf = PERCLOSBuffer(window_size=10, threshold=0.25)
        for i in range(10):
            buf.update(0.10 if i < 5 else 0.35)
        result = buf.update(0.35)   # push one more open frame
        # window = [0,0,0,0,1,1,1,1,1,1] = 4 closed / 10 = 40%
        # (the 11th update evicts the oldest closed frame)
        assert 35.0 <= result <= 65.0

    def test_sliding_window_evicts_old_frames(self):
        buf = PERCLOSBuffer(window_size=5, threshold=0.25)
        for _ in range(5):
            buf.update(0.10)       # fill with closed
        assert buf.update(0.10) == 100.0

        for _ in range(5):
            result = buf.update(0.35)  # replace with open
        assert result == 0.0

    def test_reset_clears_buffer(self):
        buf = PERCLOSBuffer(window_size=5, threshold=0.25)
        for _ in range(5):
            buf.update(0.10)
        buf.reset()
        assert not buf.ready


# ─── Feature Vector Tests ─────────────────────────────────────────────────────

class TestExtractFeatureVector:

    def test_shape_is_six(self):
        fv = extract_feature_vector(0.30, 10.0, 0.20, -5.0, 2.0, 1.0)
        assert fv.shape == (6,)

    def test_none_angles_default_to_zero(self):
        fv = extract_feature_vector(0.30, 10.0, 0.20, None, None, None)
        assert fv[3] == 0.0
        assert fv[4] == 0.0
        assert fv[5] == 0.0

    def test_values_in_correct_order(self):
        fv = extract_feature_vector(0.28, 15.0, 0.18, -3.0, 5.0, 1.0)
        assert fv[0] == pytest.approx(0.28)
        assert fv[1] == pytest.approx(15.0)
        assert fv[2] == pytest.approx(0.18)
        assert fv[3] == pytest.approx(-3.0)
        assert fv[4] == pytest.approx(5.0)
        assert fv[5] == pytest.approx(1.0)


# ─── RuleBasedClassifier Tests ───────────────────────────────────────────────

class TestRuleBasedClassifier:

    @pytest.fixture
    def clf(self):
        """Classifier with small thresholds for fast testing."""
        from unittest.mock import patch
        import config as cfg
        # Patch thresholds in-place for speed
        with (patch.object(cfg, 'EAR_CONSECUTIVE_FRAMES', 3),
              patch.object(cfg, 'MAR_CONSECUTIVE_FRAMES', 2),
              patch.object(cfg, 'DISTRACTION_FRAMES', 3),
              patch.object(cfg, 'PERCLOS_THRESHOLD', 90.0)):
            yield RuleBasedClassifier()

    def test_initial_state_is_alert(self, clf):
        state = clf.classify(0.30, 5.0, 0.20, 0.0, 0.0)
        assert state == "ALERT"

    def test_sustained_closed_eye_triggers_drowsy(self):
        """Drive _ear_cnt above the real threshold; expect DROWSY."""
        from config import EAR_CONSECUTIVE_FRAMES
        clf = RuleBasedClassifier()
        # Manually advance counter to one below threshold, then trigger
        clf._ear_cnt = EAR_CONSECUTIVE_FRAMES - 1
        # classify() will increment to EAR_CONSECUTIVE_FRAMES then check >=
        state = clf.classify(0.10, 5.0, 0.20, 0.0, 0.0)
        assert state == "DROWSY"

    def test_high_perclos_triggers_drowsy(self):
        """PERCLOS > threshold should immediately trigger DROWSY."""
        clf = RuleBasedClassifier()
        state = clf.classify(0.30, 95.0, 0.20, 0.0, 0.0)  # PERCLOS=95%
        assert state == "DROWSY"

    def test_reset_clears_counters(self, clf):
        for _ in range(3):
            clf.classify(0.10, 5.0, 0.20, 0.0, 0.0)
        clf.reset()
        state = clf.classify(0.30, 5.0, 0.20, 0.0, 0.0)
        assert state == "ALERT"

    def test_none_pose_does_not_crash(self, clf):
        state = clf.classify(0.30, 5.0, 0.20, None, None)
        assert state == "ALERT"


# ─── SVMClassifier Interface Tests ───────────────────────────────────────────

class TestSVMClassifier:

    def test_raises_if_model_not_loaded(self):
        clf = SVMClassifier()
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict(np.zeros(6), 0.30, 0.20, 0.0, 0.0)

    def test_raises_if_model_file_missing(self):
        clf = SVMClassifier()
        with pytest.raises(FileNotFoundError):
            clf.load(model_path="nonexistent.pkl", scaler_path="nonexistent.pkl")

    def test_predict_returns_str_and_proba(self):
        """Mock out model + scaler and verify predict() return types."""
        clf = SVMClassifier()
        clf._loaded = True

        mock_model = MagicMock()
        mock_model.predict.return_value = [0]           # ALERT
        mock_model.predict_proba.return_value = [[0.8, 0.15, 0.05]]

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((1, 6))

        clf.model  = mock_model
        clf.scaler = mock_scaler

        state, proba = clf.predict(np.zeros(6), 0.30, 0.20, 0.0, 0.0)

        assert isinstance(state, str)
        assert state in ("ALERT", "DROWSY", "DISTRACTED")
        assert isinstance(proba, dict)
        assert set(proba.keys()) == {"ALERT", "DROWSY", "DISTRACTED"}
