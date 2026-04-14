"""
config.py — Central configuration for the Music Genre Classifier API.
Override any value via environment variable (e.g. export MAX_CONTENT_LENGTH=20971520).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # ── Server ──────────────────────────────────────────────────────────────
    DEBUG   = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST    = os.getenv("HOST", "0.0.0.0")
    PORT    = int(os.getenv("PORT", 5000))
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

    # ── File upload ──────────────────────────────────────────────────────────
    UPLOAD_FOLDER       = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH  = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))  # 50 MB
    ALLOWED_EXTENSIONS  = {"mp3", "wav", "flac", "ogg", "aiff", "m4a"}

    # ── Audio processing ─────────────────────────────────────────────────────
    SAMPLE_RATE         = 22050          # Hz — librosa default
    CLIP_DURATION       = 30            # seconds — how much audio to analyse
    N_MFCC              = 40            # number of MFCC coefficients
    N_FFT               = 2048          # FFT window size
    HOP_LENGTH          = 512           # STFT hop length
    N_MELS              = 128           # mel filterbanks (for spectrogram)

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_PATH          = BASE_DIR / "model" / "genre_model.pkl"
    SCALER_PATH         = BASE_DIR / "model" / "scaler.pkl"
    LABEL_ENCODER_PATH  = BASE_DIR / "model" / "label_encoder.pkl"

    # ── Supported genres (must match training labels) ─────────────────────────
    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Origins allowed to call the API (set to ["*"] for open access)
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500").split(",")
