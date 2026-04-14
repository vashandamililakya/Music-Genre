"""
model/train.py
==============
Train and save the genre classification model.

Usage
-----
    # Train on GTZAN dataset
    python model/train.py --data_dir /path/to/GTZAN/genres_original

    # Train with a different classifier
    python model/train.py --data_dir /path/to/GTZAN --model svm

    # Generate demo model with synthetic data (no dataset needed)
    python model/train.py --demo

Dataset
-------
The GTZAN dataset is the standard benchmark for music genre classification.
Download: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Expected directory layout:
    genres_original/
        blues/
            blues.00000.wav
            blues.00001.wav
            ...
        classical/
            classical.00000.wav
            ...
        (10 genres × 100 tracks)

Output
------
Saves three artefacts used by the API:
    model/genre_model.pkl
    model/scaler.pkl
    model/label_encoder.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Make sure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.feature_extractor import extract_features, FEATURE_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Classifier registry
# ─────────────────────────────────────────────────────────────────────────────

def _build_classifier(name: str):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )
    elif name == "svm":
        return SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )
    elif name == "gb":
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose: rf, svm, gb")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: Path, sample_rate: int = 22050, clip_duration: float = 30.0):
    """
    Walk *data_dir*, extract features from every audio file found.

    Returns
    -------
    X : np.ndarray, shape (n_samples, FEATURE_DIM)
    y : np.ndarray, shape (n_samples,)  — string genre labels
    """
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    X_list, y_list = [], []
    errors = 0

    genre_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not genre_dirs:
        raise ValueError(f"No genre sub-directories found in '{data_dir}'")

    logger.info("Found %d genre directories: %s", len(genre_dirs), [d.name for d in genre_dirs])

    for genre_dir in genre_dirs:
        genre = genre_dir.name
        audio_files = [f for f in genre_dir.iterdir() if f.suffix.lower() in audio_extensions]
        logger.info("  %-15s  %d files", genre, len(audio_files))

        for audio_path in sorted(audio_files):
            try:
                features = extract_features(
                    audio_path,
                    sample_rate=sample_rate,
                    clip_duration=clip_duration,
                )
                X_list.append(features)
                y_list.append(genre)
            except Exception as exc:
                logger.warning("    Skipping '%s': %s", audio_path.name, exc)
                errors += 1

    logger.info("Extracted features from %d files (%d errors).", len(X_list), errors)

    if not X_list:
        raise RuntimeError("No features extracted. Check dataset path and audio files.")

    return np.stack(X_list), np.array(y_list)


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — synthetic data (no real dataset needed)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


def generate_demo_data(n_per_class: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature vectors with class-specific means so the
    model can still make reasonable-looking predictions in demo mode.
    Real-world accuracy is not achievable without real audio data.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for i, genre in enumerate(DEMO_GENRES):
        # Each genre gets a unique mean offset to separate clusters
        mean = rng.uniform(-2, 2, FEATURE_DIM) + (i * 0.5)
        cov  = np.eye(FEATURE_DIM) * rng.uniform(0.5, 1.5)
        samples = rng.multivariate_normal(mean, cov, n_per_class).astype(np.float32)
        X_list.append(samples)
        y_list.extend([genre] * n_per_class)

    return np.vstack(X_list), np.array(y_list)


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir: Path | None = None,
    model_name: str = "rf",
    test_size: float = 0.2,
    demo: bool = False,
    sample_rate: int = 22050,
    clip_duration: float = 30.0,
):
    # 1. Load data
    if demo:
        logger.info("DEMO MODE — generating synthetic data (no real accuracy).")
        X, y_raw = generate_demo_data()
    else:
        if data_dir is None or not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: '{data_dir}'. "
                "Pass --demo to generate synthetic data instead."
            )
        X, y_raw = load_dataset(data_dir, sample_rate=sample_rate, clip_duration=clip_duration)

    logger.info("Dataset shape: X=%s  classes=%s", X.shape, np.unique(y_raw).tolist())

    # 2. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    # 5. Train
    clf = _build_classifier(model_name)
    logger.info("Training %s …", type(clf).__name__)
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    logger.info("Test accuracy: %.2f%%", acc * 100)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    if not demo:
        logger.info("Running 5-fold CV …")
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy", n_jobs=-1)
        logger.info("CV accuracy: %.2f%% ± %.2f%%", cv_scores.mean() * 100, cv_scores.std() * 100)

    # 7. Save artefacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf,    MODEL_DIR / "genre_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(le,     MODEL_DIR / "label_encoder.pkl")

    logger.info("Artefacts saved to '%s'", MODEL_DIR)
    logger.info("  genre_model.pkl  |  scaler.pkl  |  label_encoder.pkl")

    return clf, scaler, le, acc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the music genre classifier.")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Path to GTZAN genres_original directory.")
    parser.add_argument("--model", choices=["rf", "svm", "gb"], default="rf",
                        help="Classifier: rf=RandomForest (default), svm=SVM, gb=GradientBoosting")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data — no real dataset needed.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Audio clip duration in seconds (default: 30)")
    args = parser.parse_args()

    train(
        data_dir     = args.data_dir,
        model_name   = args.model,
        test_size    = args.test_size,
        demo         = args.demo,
        sample_rate  = args.sr,
        clip_duration= args.duration,
    )


if __name__ == "__main__":
    main()
