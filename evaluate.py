"""
model/evaluate.py
=================
Evaluate a trained model and print a full report + confusion matrix.

Usage
-----
    python model/evaluate.py --data_dir /path/to/GTZAN/genres_original
    python model/evaluate.py --data_dir /path/to/GTZAN --plot   # save confusion matrix PNG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.train import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent


def evaluate(data_dir: Path, plot: bool = False):
    # Load artefacts
    clf = joblib.load(MODEL_DIR / "genre_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")

    # Load data
    X, y_raw = load_dataset(data_dir)
    y = le.transform(y_raw)
    X_scaled = scaler.transform(X)

    # Predict
    y_pred = clf.predict(X_scaled)
    acc = (y_pred == y).mean()

    print(f"\nOverall Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:\n")
    print(classification_report(y, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):\n")
    header = "           " + "  ".join(f"{c[:4]:>6}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:6d}" for v in row)
        print(f"  {le.classes_[i]:<10} {row_str}")

    if plot:
        _save_confusion_matrix_png(cm, le.classes_)


def _save_confusion_matrix_png(cm: np.ndarray, class_names: list[str]):
    """Save a confusion matrix as a PNG if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set(
            xticks=range(len(class_names)), yticks=range(len(class_names)),
            xticklabels=class_names, yticklabels=class_names,
            ylabel="Actual", xlabel="Predicted",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.colorbar(im)
        out = MODEL_DIR / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(str(out), dpi=120)
        logger.info("Confusion matrix saved to '%s'", out)
    except ImportError:
        logger.warning("matplotlib not installed — skipping PNG output.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained genre classifier.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="Save confusion matrix PNG")
    args = parser.parse_args()
    evaluate(args.data_dir, args.plot)


if __name__ == "__main__":
    main()
