"""
evaluation/evaluate_baseline.py — Module 3.3: Hold-Out Test Set Evaluation
============================================================================
Evaluates the trained SVM baseline on the held-out test partition and
produces standard classification metrics. No demographic subgroup analysis
is performed; this is an AI Project deliverable, not the thesis.

Methodology (Section 5.9 of the documentation):
  1. Load data/extracted_features.csv.
  2. Reproduce the same 70/30 stratified split used in train_svm.py
     (random_state=42) to obtain the identical held-out test partition.
  3. Load data/svm_baseline_model.pkl and data/svm_baseline_scaler.pkl.
  4. Apply the scaler and predict.
  5. Compute overall accuracy, per-class precision/recall/F1, and the
     3×3 confusion matrix.
  6. Write outputs/evaluation_report.txt and
     outputs/evaluation_confusion_matrix.png.

Usage:
    py evaluation/evaluate_baseline.py

Course: COSC 304 — Introduction to Artificial Intelligence
Group : BSCS 3-3, Group 7 — PUP CCIS, A.Y. 2025–2026
"""

import os
import sys
import pickle
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURE_NAMES, SVM_MODEL_PATH, SVM_SCALER_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_PATH     = "data/extracted_features.csv"
OUT_DIR      = "outputs"
TARGET_NAMES = ["ALERT", "DROWSY", "DISTRACTED"]


def main() -> None:
    """Run the held-out test set evaluation and save results."""
    print("\n" + "=" * 65)
    print("  ALERTO Baseline — Held-Out Test Set Evaluation  (Module 3.3)")
    print("=" * 65 + "\n")

    # -- 1. Load CSV ---------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found.")
        print("        Run: py training/extract_features.py  first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_NAMES].values
    y = df["label"].values
    print(f"  Loaded {len(df):,} samples from {CSV_PATH}")

    # -- 2. Reproduce same 70/30 split (must match train_svm.py) -------------
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    print(f"  Test partition : {len(X_test):,} samples\n")

    # -- 3. Load model and scaler --------------------------------------------
    for path in (SVM_MODEL_PATH, SVM_SCALER_PATH):
        if not os.path.exists(path):
            print(f"[ERROR] Artifact not found: {path}")
            print("        Run: py training/train_svm.py  first.")
            sys.exit(1)

    with open(SVM_MODEL_PATH,  "rb") as fh:
        model  = pickle.load(fh)
    with open(SVM_SCALER_PATH, "rb") as fh:
        scaler = pickle.load(fh)

    # -- 4. Scale and predict ------------------------------------------------
    X_test_scaled = scaler.transform(X_test)
    y_pred        = model.predict(X_test_scaled)

    # -- 5. Compute metrics --------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    
    # Overall metrics
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro        = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report   = classification_report(
        y_test, y_pred,
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Overall accuracy          : {accuracy * 100:.2f}%")
    print(f"  Overall Precision (Macro) : {precision_macro:.4f}")
    print(f"  Overall Recall (Macro)    : {recall_macro:.4f}")
    print(f"  Overall F1-Score (Macro)  : {f1_macro:.4f}")
    print(f"  Overall Precision (Weight): {precision_weighted:.4f}")
    print(f"  Overall Recall (Weight)   : {recall_weighted:.4f}")
    print(f"  Overall F1-Score (Weight) : {f1_weighted:.4f}\n")

    print("  Per-class classification report:")
    print(report)

    # -- 6. Save outputs -----------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- 6a. Text report -----------------------------------------------------
    report_path = os.path.join(OUT_DIR, "evaluation_report.txt")
    sep = "=" * 65
    lines = [
        sep,
        "  ALERTO BASELINE — Held-Out Test Set Evaluation",
        sep,
        f"  Generated       : {timestamp}",
        f"  Test partition  : {len(X_test):,} samples",
        f"  Overall accuracy: {accuracy * 100:.2f}%",
        f"  Macro Precision : {precision_macro:.4f}",
        f"  Macro Recall    : {recall_macro:.4f}",
        f"  Macro F1-Score  : {f1_macro:.4f}",
        f"  Weighted Prec.  : {precision_weighted:.4f}",
        f"  Weighted Recall : {recall_weighted:.4f}",
        f"  Weighted F1     : {f1_weighted:.4f}",
        "",
        "  Per-class metrics:",
        "-" * 65,
        report,
        "  Confusion matrix (rows = true label, columns = predicted label):",
        "  Labels: [ALERT, DROWSY, DISTRACTED]",
        str(cm),
        "",
        sep,
    ]
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Evaluation report saved -> {report_path}")

    # -- 6b. Confusion matrix PNG -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TARGET_NAMES,
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        "ALERTO Baseline — Confusion Matrix (Hold-Out Test Set)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved  -> {cm_path}")

    print("\n" + sep + "\n")


if __name__ == "__main__":
    main()
