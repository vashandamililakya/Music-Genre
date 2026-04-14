"""
app/__init__.py
===============
Flask application factory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from config import Config

logger = logging.getLogger(__name__)


def create_app(config: type = Config) -> Flask:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    config : config class (default: Config from config.py)

    Returns
    -------
    Flask app instance
    """
    app = Flask(__name__)
    app.config.from_object(config)

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if config.DEBUG else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Upload folder ─────────────────────────────────────────────────────────
    upload_folder: Path = app.config["UPLOAD_FOLDER"]
    upload_folder.mkdir(parents=True, exist_ok=True)

    # ── CORS ──────────────────────────────────────────────────────────────────
    _register_cors(app)

    # ── Predictor (attached to app for global access) ─────────────────────────
    from app.predictor import GenrePredictor

    predictor = GenrePredictor(
        model_path         = config.MODEL_PATH,
        scaler_path        = config.SCALER_PATH,
        label_encoder_path = config.LABEL_ENCODER_PATH,
        sample_rate        = config.SAMPLE_RATE,
        clip_duration      = config.CLIP_DURATION,
        n_mfcc             = config.N_MFCC,
        n_fft              = config.N_FFT,
        hop_length         = config.HOP_LENGTH,
    )
    app.predictor = predictor  # type: ignore[attr-defined]

    # Eagerly load model if artefacts exist (avoids cold-start latency)
    if predictor.is_ready():
        try:
            predictor.load()
        except Exception as exc:
            logger.warning("Could not pre-load model: %s", exc)

    # ── Blueprints ────────────────────────────────────────────────────────────
    from app.routes import bp as api_bp
    app.register_blueprint(api_bp)

    # ── Root route (sanity check) ─────────────────────────────────────────────
    @app.get("/")
    def index():
        return {
            "name": "AI Music Genre Classifier API",
            "version": "1.0.0",
            "docs": "/api/health",
        }

    logger.info("App created. Model ready: %s", predictor.is_ready())
    return app


def _register_cors(app: Flask) -> None:
    """
    Add CORS headers without requiring flask-cors.
    Reads allowed origins from app.config['CORS_ORIGINS'].
    """
    allowed: list[str] = app.config.get("CORS_ORIGINS", ["*"])

    @app.after_request
    def add_cors_headers(response):
        origin = request_origin()
        if "*" in allowed or origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.before_request
    def handle_preflight():
        from flask import request, make_response
        if request.method == "OPTIONS":
            resp = make_response()
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.status_code = 204
            return resp


def request_origin() -> str:
    from flask import request
    return request.headers.get("Origin", "")


def _serve_static(app: Flask) -> None:
    """Serve the frontend from /static/index.html at the root URL."""
    from flask import send_from_directory

    static_dir = Path(__file__).parent.parent / "static"

    @app.get("/ui")
    def ui():
        return send_from_directory(str(static_dir), "index.html")
