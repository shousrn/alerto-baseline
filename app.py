"""
app.py — ALERTO Baseline — Flask Web Backend (Module 4.4)
=========================================================
Runs the four-phase detection pipeline in a background thread and exposes
three HTTP endpoints consumed by the React frontend.

Endpoints:
  GET /metrics      — JSON snapshot of the latest detection state.
  GET /video_feed   — MJPEG stream (multipart/x-mixed-replace) with face-mesh
                      dots only; no textual HUD (the web UI renders its own).
  GET /             — Serves the compiled React frontend from frontend/dist/.
  GET /<path>       — SPA fallback: returns index.html for unmatched paths.

/metrics JSON schema (Section 5.5.2 of the documentation):
  {
    "state":             str,          // "ALERT" | "DROWSY" | "DISTRACTED" | "ERROR"
    "ear":               float,
    "perclos":           float,
    "mar":               float,
    "pitch":             float,
    "yaw":               float,
    "roll":              float,
    "fps":               int,
    "latency_ms":        float,
    "ear_threshold":     float,
    "mar_threshold":     float,
    "perclos_threshold": float,
    "pitch_threshold":   float,
    "yaw_threshold":     float,
    "svm_proba":         {ALERT, DROWSY, DISTRACTED},
    "ear_consec":        int,
    "mar_consec":        int,
    "pose_consec":       int,
    "error_message":     str | null
  }

Usage:
    py app.py

Course: COSC 304 — Introduction to Artificial Intelligence
Group : BSCS 3-3, Group 7 — PUP CCIS, A.Y. 2025–2026
"""

import cv2
import time
import threading
import os
import logging
import datetime
from collections import deque

from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from modules.features import compute_avg_ear, compute_mar, compute_head_pose, PERCLOSBuffer, extract_feature_vector
from modules.classifier import SVMClassifier
from config import (
    PERCLOS_WINDOW_FRAMES, EAR_CLOSED_THRESHOLD, MAR_YAWN_THRESHOLD,
    PERCLOS_THRESHOLD, PITCH_THRESH_LOW, YAW_THRESHOLD, SVM_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="frontend/dist")
CORS(app)

# ---------------------------------------------------------------------------
# Thread-safe globals
# ---------------------------------------------------------------------------

latest_frame_bytes: bytes | None = None
latest_metrics: dict = {
    "state":             "ALERT",
    "ear":               0.0,
    "perclos":           0.0,
    "mar":               0.0,
    "pitch":             0.0,
    "yaw":               0.0,
    "roll":              0.0,
    "fps":               0,
    "latency_ms":        0.0,
    "ear_threshold":     EAR_CLOSED_THRESHOLD,
    "mar_threshold":     MAR_YAWN_THRESHOLD,
    "perclos_threshold": PERCLOS_THRESHOLD,
    "pitch_threshold":   abs(PITCH_THRESH_LOW),
    "yaw_threshold":     YAW_THRESHOLD,
    "svm_proba":         {"ALERT": 0.0, "DROWSY": 0.0, "DISTRACTED": 0.0},
    "ear_consec":        0,
    "mar_consec":        0,
    "pose_consec":       0,
    "error_message":     None,
}

_frame_lock   = threading.Lock()
_metrics_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Session logger
# ---------------------------------------------------------------------------

def _setup_session_logger() -> logging.Logger:
    """
    Create a per-session log file at outputs/logs/alerto_YYYYMMDD_HHMMSS.log.

    Each run gets its own file so sessions don't overwrite each other.
    Log format mirrors the old drowsiness logger style for consistency:
      YYYY-MM-DD HH:MM:SS | LEVEL | message
    """
    os.makedirs("outputs/logs", exist_ok=True)
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"outputs/logs/alerto_{ts}.log"

    logger = logging.getLogger("alerto")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # avoid duplicate handlers on reload

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Also mirror to console so you see it while running
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[ALERTO] %(levelname)s %(message)s"))
    logger.addHandler(ch)

    logger.info(f"Session started — log: {logfile}")
    return logger


_session_log = _setup_session_logger()


# ---------------------------------------------------------------------------
# Detection background thread
# ---------------------------------------------------------------------------

def _detection_loop() -> None:
    """
    Run the four-phase ALERTO pipeline continuously in a background thread.

    Writes to `latest_frame_bytes` (JPEG) and `latest_metrics` (dict) using
    thread locks so Flask route handlers read consistent values.
    """
    global latest_frame_bytes, latest_metrics

    # -- MediaPipe init ------------------------------------------------------
    model_path = "dataset/face_landmarker.task"
    if not os.path.exists(model_path):
        error_msg = (
            f"MediaPipe model file '{model_path}' not found. "
            "Download from https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
            "and place at the indicated path."
        )
        print(f"[APP ERROR] {error_msg}")
        with _metrics_lock:
            latest_metrics["state"] = "ERROR"
            latest_metrics["error_message"] = error_msg
        return

    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
    except Exception as exc:
        error_msg = f"Failed to create FaceLandmarker: {exc}"
        print(f"[APP ERROR] {error_msg}")
        with _metrics_lock:
            latest_metrics["state"] = "ERROR"
            latest_metrics["error_message"] = error_msg
        return

    # -- SVM init ------------------------------------------------------------
    clf = SVMClassifier()
    svm_error: str | None = None
    try:
        clf.load()
        print("[APP] SVM Classifier loaded successfully.")
    except Exception as exc:
        svm_error = str(exc)
        print(f"[APP] SVM load failed: {exc} — server will still run, returning ERROR state.")

    if svm_error:
        with _metrics_lock:
            latest_metrics["state"] = "ERROR"
            latest_metrics["error_message"] = svm_error

    # -- Camera init ---------------------------------------------------------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    video_fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx         = 0
    last_timestamp_ms = -1

    # Rolling 60-iteration latency buffer for FPS computation
    latency_buf: deque = deque(maxlen=60)

    perclos_buf = PERCLOSBuffer(window_size=PERCLOS_WINDOW_FRAMES,
                                threshold=EAR_CLOSED_THRESHOLD)

    prev_state  = "ALERT"   # track transitions for logging
    frame_count = 0

    _session_log.info(
        f"Detection loop started — classifier={'SVM' if not svm_error else 'NONE (ERROR)'}"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        t_start = time.perf_counter()

        # ── Phase 1: FaceLandmarker ────────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(
            int((frame_idx / video_fps) * 1000),
            last_timestamp_ms + 1,
        )
        last_timestamp_ms = timestamp_ms
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        state       = "ALERT" if not svm_error else "ERROR"
        proba       = {"ALERT": 0.0, "DROWSY": 0.0, "DISTRACTED": 0.0}
        ear = perclos = mar = pitch = yaw = roll = 0.0
        ear_consec  = mar_consec = pose_consec = 0

        if results.face_landmarks:
            lm = results.face_landmarks[0]
            h, w = frame.shape[:2]

            # Draw face-mesh dots (no textual HUD in web mode)
            for landmark in lm:
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                cv2.circle(frame, (px, py), 1, (0, 200, 100), -1)

            ear     = compute_avg_ear(lm, h, w)
            perclos = perclos_buf.update(ear)
            mar     = compute_mar(lm, h, w)
            pitch, yaw, roll = compute_head_pose(lm, h, w)

            if not svm_error and hasattr(clf, "model") and clf.model is not None:
                fvec       = extract_feature_vector(ear, perclos, mar, pitch, yaw, roll)
                state, p   = clf.predict(fvec, ear, mar, pitch, yaw)
                if p:
                    proba = p
                ear_consec  = clf._ear_cnt
                mar_consec  = clf._mar_cnt
                pose_consec = clf._pose_cnt

        frame_count += 1

        # ── Latency and FPS ───────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        latency_buf.append(latency_ms)
        avg_latency = sum(latency_buf) / len(latency_buf)
        fps_val     = int(1000.0 / avg_latency) if avg_latency > 0 else 0

        # ── Log state transitions + screenshot on alert ───────────────────
        if state != prev_state:
            _session_log.info(
                f"State transition: {prev_state} → {state} | "
                f"EAR={ear:.3f} | PERCLOS={perclos:.1f}% | "
                f"MAR={mar:.3f} | Pitch={float(pitch or 0.0):+.1f}° | "
                f"Yaw={float(yaw or 0.0):+.1f}° | frame={frame_count}"
            )
            if state in ("DROWSY", "DISTRACTED"):
                _session_log.warning(
                    f"ALERT: {state} detected at frame {frame_count}"
                )
                # Save a screenshot of the triggering frame
                os.makedirs("outputs/screenshots", exist_ok=True)
                shot_ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                shot_path = f"outputs/screenshots/alert_{state.lower()}_{shot_ts}.jpg"
                cv2.imwrite(shot_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                _session_log.warning(f"Screenshot saved: {shot_path}")
            prev_state = state

        # ── Update shared metrics ─────────────────────────────────────────
        with _metrics_lock:
            latest_metrics["state"]             = state
            latest_metrics["ear"]               = round(ear, 4)
            latest_metrics["perclos"]           = round(perclos, 2)
            latest_metrics["mar"]               = round(mar, 4)
            latest_metrics["pitch"]             = round(float(pitch or 0.0), 2)
            latest_metrics["yaw"]               = round(float(yaw or 0.0), 2)
            latest_metrics["roll"]              = round(float(roll or 0.0), 2)
            latest_metrics["fps"]               = fps_val
            latest_metrics["latency_ms"]        = round(avg_latency, 1)
            latest_metrics["svm_proba"]         = proba
            latest_metrics["ear_consec"]        = ear_consec
            latest_metrics["mar_consec"]        = mar_consec
            latest_metrics["pose_consec"]       = pose_consec
            if not svm_error:
                latest_metrics["error_message"] = None

        # ── JPEG encode for video feed ────────────────────────────────────
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with _frame_lock:
                latest_frame_bytes = buf.tobytes()

    _session_log.info(
        f"Session ended — {frame_count} frames processed. "
        f"Camera released."
    )
    cap.release()
    face_landmarker.close()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/metrics")
def metrics() -> Response:
    """Return the latest detection metrics as JSON."""
    with _metrics_lock:
        return jsonify(latest_metrics)


def _generate_frames():
    """Generator that yields MJPEG frames for the /video_feed endpoint."""
    while True:
        with _frame_lock:
            frame = latest_frame_bytes
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.033)   # ~30 FPS cap to avoid saturating the network


@app.route("/video_feed")
def video_feed() -> Response:
    """Stream the camera feed as multipart/x-mixed-replace MJPEG."""
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path: str) -> Response:
    """Serve the compiled React frontend; fall back to index.html for SPA routing."""
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    index = os.path.join(app.static_folder, "index.html")
    if os.path.exists(index):
        return send_from_directory(app.static_folder, "index.html")
    return "Frontend not built yet. Run 'npm run build' in the frontend/ directory.", 503


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[APP] Starting background detection thread...")
    t = threading.Thread(target=_detection_loop, daemon=True)
    t.start()

    print("[APP] Flask server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
