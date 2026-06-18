"""
training/train_svm.py — Module 3.2: SVM Training
==================================================
Trains a Support Vector Machine (RBF kernel) on the feature vectors produced
by extract_features.py and persists the trained model and scaler.

Pipeline:
  1. Load data/extracted_features.csv.
  2. Stratified 70/30 train-test split (random_state=42).
  3. Fit a global StandardScaler on the training partition only.
  4. Train SVC(kernel='rbf', C=1.0, gamma='scale', probability=True).
  5. 5-fold cross-validation on the scaled training partition.
  6. Evaluate on the held-out test partition.
  7. Persist model → data/svm_baseline_model.pkl
             scaler → data/svm_baseline_scaler.pkl
  8. Save outputs/confusion_matrix.png and outputs/training_report.txt.

SVM configuration (Section 5.4.4 of the documentation):
  kernel='rbf'       — Radial Basis Function, nonlinear decision boundary.
  C=1.0              — Default regularization strength.
  gamma='scale'      — 1 / (n_features × Var(X)), scikit-learn default.
  probability=True   — Enables Platt-calibrated posterior probabilities
                       consumed by the user interface.

Usage:
    py training/train_svm.py

Course: COSC 304 — Introduction to Artificial Intelligence
Group : BSCS 3-3, Group 7 — PUP CCIS, A.Y. 2025–2026
"""

import os
import sys
import pickle
import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURE_NAMES, SVM_MODEL_PATH, SVM_SCALER_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_PATH    = "data/extracted_features.csv"
OUT_DIR     = "outputs"
TARGET_NAMES = ["ALERT", "DROWSY", "DISTRACTED"]


def main() -> None:
    """Run the full SVM training pipeline."""
    print("\n" + "=" * 65)
    print("  ALERTO Baseline — SVM Training  (Module 3.2)")
    print("=" * 65 + "\n")

    # -- 1. Load data --------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found.")
        print("        Run: py training/extract_features.py  first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} samples from {CSV_PATH}")
    print(f"  Class distribution:\n{df['label'].value_counts().sort_index().to_string()}\n")

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # -- 2. Stratified 70/30 train-test split --------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    print(f"  Training set : {len(X_train):,} samples")
    print(f"  Test set     : {len(X_test):,} samples\n")

    # -- 3. Fit global StandardScaler on training partition only -------------
    #    This is the global scaler prescribed by the documentation (Section
    #    5.4.3). It is fitted once on the pooled training partition; no
    #    per-user normalization is performed — that is reserved for the
    #    subsequent ALERTO thesis.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs("data", exist_ok=True)
    with open(SVM_SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  Scaler saved -> {SVM_SCALER_PATH}")

    # -- 4. Train SVC (RBF kernel) -------------------------------------------
    print("\n  Training SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)...")
    model = SVC(kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=42)

    # -- 5. 5-fold cross-validation on training partition --------------------
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full training set
    model.fit(X_train_scaled, y_train)
    with open(SVM_MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  Model saved  -> {SVM_MODEL_PATH}")

    # -- 6. Evaluate on held-out test partition ------------------------------
    y_pred = model.predict(X_test_scaled)

    report = classification_report(
        y_test, y_pred,
        target_names=TARGET_NAMES,
        digits=4,
    )
    print("\n" + "=" * 65)
    print("  Hold-Out Test Set Classification Report")
    print("=" * 65)
    print(report)

    # -- 7. Save outputs -----------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- 7a. Confusion matrix PNG ------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=TARGET_NAMES,
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        "ALERTO Baseline — SVM Confusion Matrix (Hold-Out Test Set)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    cm_path = "outputs/training_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}")

    # -- 7b. Text training report -------------------------------------------
    report_path = os.path.join(OUT_DIR, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("ALERTO BASELINE — SVM TRAINING REPORT\n")
        fh.write("=" * 65 + "\n")
        fh.write(f"Generated  : {timestamp}\n")
        fh.write(f"CSV source : {CSV_PATH}  ({len(df):,} samples)\n")
        fh.write(f"Train/Test : {len(X_train):,} / {len(X_test):,} samples\n\n")
        fh.write(f"5-fold CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        fh.write("Hold-Out Test Set Classification Report\n")
        fh.write("-" * 65 + "\n")
        fh.write(report)
        fh.write("\nConfusion Matrix (rows=true, cols=predicted):\n")
        fh.write(str(cm) + "\n")
    print(f"  Training report saved → {report_path}")

    print("\n  Training complete. Run: py main.py --mode svm")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
