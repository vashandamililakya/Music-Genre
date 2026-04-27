"""
app/predictor.py
================
Loads the trained model artefacts and provides a clean `predict()` interface.

The model pipeline is:
    raw audio → feature vector → StandardScaler → classifier → probabilities

Artefacts (written by model/train.py):
    model/genre_model.pkl    — sklearn estimator (RandomForest / SVM / etc.)
    model/scaler.pkl         — fitted StandardScaler
    model/label_encoder.pkl  — fitted LabelEncoder
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.feature_extractor import extract_features, FEATURE_DIM

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes (plain dicts for JSON serialisability)
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidate(genre: str, confidence: float) -> dict:
    return {"genre": genre.capitalize(), "confidence": round(float(confidence), 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────────────

class GenrePredictor:
    """
    Lazy-loads model artefacts on first call.

    Usage
    -----
    predictor = GenrePredictor(model_path, scaler_path, label_encoder_path)
    result = predictor.predict("/tmp/upload/track.wav")
    """

    def __init__(
        self,
        model_path: str | Path,
        scaler_path: str | Path,
        label_encoder_path: str | Path,
        sample_rate: int = 22050,
        clip_duration: float = 30.0,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.model_path         = Path(model_path)
        self.scaler_path        = Path(scaler_path)
        self.label_encoder_path = Path(label_encoder_path)

        # Audio config forwarded to feature extractor
        self.sample_rate   = sample_rate
        self.clip_duration = clip_duration
        self.n_mfcc        = n_mfcc
        self.n_fft         = n_fft
        self.hop_length    = hop_length

        # Loaded lazily
        self._model         = None
        self._scaler        = None
        self._label_encoder = None
        self._loaded        = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Explicitly load all model artefacts into memory."""
        logger.info("Loading model artefacts …")
        self._model         = joblib.load(self.model_path)
        self._scaler        = joblib.load(self.scaler_path)
        self._label_encoder = joblib.load(self.label_encoder_path)
        self._loaded        = True
        logger.info(
            "Model loaded: %s | classes: %s",
            type(self._model).__name__,
            list(self._label_encoder.classes_),
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at '{self.model_path}'. "
                    "Please run `python model/train.py` first."
                )
            self.load()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, file_path: str | Path) -> dict:
        """
        Predict the genre of an audio file.

        Returns
        -------
        {
            "genre": "Jazz",
            "confidence": 0.91,
            "candidates": [
                {"genre": "Blues", "confidence": 0.05},
                ...
            ],
            "features": {
                "tempo": 124.3,
                "energy": 0.63,
                "zero_crossing_rate": 0.08
            }
        }
        """
        self._ensure_loaded()

        file_path = Path(file_path)
        logger.info("Predicting genre for '%s'", file_path.name)

        # 1. Extract features
        feature_vec = extract_features(
            file_path,
            sample_rate=self.sample_rate,
            clip_duration=self.clip_duration,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )  # shape: (FEATURE_DIM,)

        # 2. Scale
        X = self._scaler.transform(feature_vec.reshape(1, -1))  # (1, FEATURE_DIM)

        # 3. Predict probabilities
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]   # (n_classes,)
        else:
            # SVM with decision_function — convert to pseudo-probabilities
            decision = self._model.decision_function(X)[0]
            proba = _softmax(decision)

        # 4. Decode labels
        classes = self._label_encoder.classes_
        top_idx = int(np.argmax(proba))
        top_genre      = classes[top_idx]
        top_confidence = float(proba[top_idx])

        # All candidates sorted by confidence (excluding top)
        sorted_idx = np.argsort(proba)[::-1]
        candidates = [
            _make_candidate(classes[i], proba[i])
            for i in sorted_idx
            if i != top_idx
        ]

        # 5. Readable feature summary (subset for UI display)
        feature_summary = _summarise_features(feature_vec)

        return {
            "genre":      top_genre.capitalize(),
            "confidence": round(top_confidence, 4),
            "candidates": candidates[:5],           # top-5 runners-up
            "features":   feature_summary,
        }

    # ── Health check ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if model artefacts exist on disk."""
        return (
            self.model_path.exists()
            and self.scaler_path.exists()
            and self.label_encoder_path.exists()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _summarise_features(vec: np.ndarray) -> dict:
    """
    Return a small human-readable dict extracted from the feature vector.

    Feature vector layout (see feature_extractor.py):
        [0:80]   MFCC mean (40) + std (40)
        [80:104] chroma mean (12) + std (12)
        [104:106] spectral centroid mean + std
        [106:108] spectral rolloff mean + std
        [108:110] spectral bandwidth mean + std
        [110:124] spectral contrast mean (7) + std (7)
        [124:126] ZCR mean + std
        [126:128] RMS mean + std
        [128:140] tonnetz mean (6) + std (6)
        [140]    tempo
        [141]    beat strength mean
    """
    if len(vec) < FEATURE_DIM:
        return {}

    return {
        "tempo":              round(float(vec[140]), 1),
        "energy":             round(float(vec[126]), 4),   # RMS mean
        "spectral_centroid":  round(float(vec[104]), 1),
        "zero_crossing_rate": round(float(vec[124]), 4),
        "beat_strength":      round(float(vec[141]), 4),
    }
