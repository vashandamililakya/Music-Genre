"""
app/routes.py
=============
All Flask API routes.

Endpoints
---------
POST /api/predict   — Upload audio → genre prediction
GET  /api/health    — Health check
GET  /api/genres    — List supported genres
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from flask import Blueprint, current_app, request

from app.utils import (
    allowed_file,
    cleanup_file,
    cleanup_old_uploads,
    error_response,
    secure_temp_path,
    success_response,
    validate_audio_file,
)

logger = logging.getLogger(__name__)
bp = Blueprint("api", __name__, url_prefix="/api")


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict
# ─────────────────────────────────────────────────────────────────────────────

@bp.route("/predict", methods=["POST"])
def predict():
    """
    Upload an audio file and receive a genre prediction.

    Request
    -------
    Content-Type: multipart/form-data
    Field       : audio  (file)

    Response 200
    ------------
    {
        "status": "success",
        "data": {
            "genre": "Jazz",
            "confidence": 0.91,
            "candidates": [
                {"genre": "Blues", "confidence": 0.05},
                {"genre": "Classical", "confidence": 0.03}
            ],
            "features": {
                "tempo": 124.3,
                "energy": 0.63,
                "spectral_centroid": 1850.2,
                "zero_crossing_rate": 0.08,
                "beat_strength": 0.42
            },
            "processing_time_ms": 840
        }
    }
    """
    t_start = time.monotonic()

    # ── Periodic temp-file cleanup (every ~50 requests) ──────────────────────
    if int(time.time()) % 50 == 0:
        upload_folder: Path = current_app.config["UPLOAD_FOLDER"]
        deleted = cleanup_old_uploads(upload_folder)
        if deleted:
            logger.info("Cleaned up %d stale upload(s).", deleted)

    # ── Validate request ──────────────────────────────────────────────────────
    if "audio" not in request.files:
        return error_response("No audio file in request. Use field name 'audio'.", 400)

    file = request.files["audio"]

    if not file.filename:
        return error_response("No file selected.", 400)

    if not allowed_file(file.filename):
        return error_response(
            f"Unsupported file type '{Path(file.filename).suffix}'. "
            "Supported formats: MP3, WAV, FLAC, OGG.",
            415,
        )

    # ── Magic-byte validation ─────────────────────────────────────────────────
    header_bytes = file.read(32)
    file.seek(0)
    valid, reason = validate_audio_file(header_bytes)
    if not valid:
        return error_response(reason, 415)

    # ── Save to temp file ─────────────────────────────────────────────────────
    upload_folder: Path = current_app.config["UPLOAD_FOLDER"]
    upload_folder.mkdir(parents=True, exist_ok=True)
    temp_path = secure_temp_path(upload_folder, file.filename)

    try:
        file.save(str(temp_path))
        logger.info("Saved upload to '%s' (%d bytes)", temp_path.name, temp_path.stat().st_size)
    except Exception as exc:
        cleanup_file(temp_path)
        logger.error("Failed to save upload: %s", exc)
        return error_response("Failed to save uploaded file.", 500, str(exc))

    # ── Run inference ─────────────────────────────────────────────────────────
    predictor = current_app.predictor  # set in create_app()
    try:
        result = predictor.predict(temp_path)
    except FileNotFoundError as exc:
        cleanup_file(temp_path)
        return error_response(
            "Model not trained yet. Run `python model/train.py` first.",
            503,
            str(exc),
        )
    except RuntimeError as exc:
        cleanup_file(temp_path)
        logger.warning("Feature extraction failed for '%s': %s", file.filename, exc)
        return error_response(str(exc), 422)
    except Exception as exc:
        cleanup_file(temp_path)
        logger.exception("Unexpected prediction error for '%s': %s", file.filename, exc)
        return error_response("Prediction failed due to an internal error.", 500, str(exc))
    finally:
        cleanup_file(temp_path)

    # ── Attach processing time ────────────────────────────────────────────────
    result["processing_time_ms"] = round((time.monotonic() - t_start) * 1000)

    return success_response(result, 200)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/health
# ─────────────────────────────────────────────────────────────────────────────

@bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Response 200: {"status": "success", "data": {"healthy": true, "model_ready": true}}
    Response 503: model artefacts not found
    """
    predictor = current_app.predictor
    model_ready = predictor.is_ready()

    data = {
        "healthy":     True,
        "model_ready": model_ready,
        "model_path":  str(predictor.model_path),
    }

    if not model_ready:
        data["message"] = "Model not trained yet. Run `python model/train.py`."
        return success_response(data, 503)

    return success_response(data, 200)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/genres
# ─────────────────────────────────────────────────────────────────────────────

@bp.route("/genres", methods=["GET"])
def genres():
    """Return the list of genres the model was trained on."""
    genre_list = [g.capitalize() for g in current_app.config.get("GENRES", [])]
    return success_response({"genres": genre_list, "count": len(genre_list)})
