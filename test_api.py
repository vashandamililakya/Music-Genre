"""
tests/test_api.py
=================
Automated tests for the Music Genre Classifier API.

Run
---
    pytest tests/ -v
    pytest tests/ -v --tb=short   # compact tracebacks
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Test config (uses a separate temp upload folder)
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig(Config):
    TESTING     = True
    DEBUG       = False
    UPLOAD_FOLDER = Path("/tmp/test_uploads")


@pytest.fixture(scope="session")
def app():
    _app = create_app(TestConfig)
    yield _app


@pytest.fixture()
def client(app):
    return app.test_client()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — generate a minimal valid WAV in memory
# ─────────────────────────────────────────────────────────────────────────────

def _make_wav_bytes(duration: float = 1.0, sr: int = 22050) -> bytes:
    """Return raw bytes of a valid mono WAV file."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Mix of sine waves to resemble a musical signal
    audio = (
        0.4 * np.sin(2 * np.pi * 440 * t)   # A4
        + 0.3 * np.sin(2 * np.pi * 880 * t)  # A5
        + 0.2 * np.random.default_rng(0).normal(0, 0.1, n_samples)
    )
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


WAV_BYTES = _make_wav_bytes(duration=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_json(self, client):
        resp = client.get("/api/health")
        assert resp.content_type == "application/json"

    def test_health_status_key(self, client):
        data = client.get("/api/health").get_json()
        assert "status" in data

    def test_health_data_has_healthy(self, client):
        data = client.get("/api/health").get_json()
        assert data["data"]["healthy"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Genres endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestGenres:
    def test_genres_200(self, client):
        assert client.get("/api/genres").status_code == 200

    def test_genres_list_nonempty(self, client):
        data = client.get("/api/genres").get_json()
        assert len(data["data"]["genres"]) > 0

    def test_genres_count_matches(self, client):
        data = client.get("/api/genres").get_json()
        assert data["data"]["count"] == len(data["data"]["genres"])


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint — validation errors
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictValidation:
    def test_no_file_returns_400(self, client):
        resp = client.post("/api/predict")
        assert resp.status_code == 400

    def test_wrong_field_name_returns_400(self, client):
        data = {"file": (io.BytesIO(WAV_BYTES), "track.wav")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_empty_filename_returns_400(self, client):
        data = {"audio": (io.BytesIO(WAV_BYTES), "")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_unsupported_extension_returns_415(self, client):
        data = {"audio": (io.BytesIO(b"fake data"), "track.txt")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 415

    def test_fake_audio_bytes_returns_415(self, client):
        # Correct extension but garbage bytes (fails magic-byte check)
        data = {"audio": (io.BytesIO(b"this is not audio content at all!!"), "track.wav")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 415


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint — happy path (requires trained model)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictHappyPath:
    """
    These tests only run when a trained model is available.
    To generate a demo model first run:
        python model/train.py --demo
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self, app):
        if not app.predictor.is_ready():
            pytest.skip("No trained model — run `python model/train.py --demo` first.")

    def _post_wav(self, client, wav_bytes: bytes = WAV_BYTES):
        data = {"audio": (io.BytesIO(wav_bytes), "test_track.wav")}
        return client.post("/api/predict", data=data, content_type="multipart/form-data")

    def test_predict_returns_200(self, client):
        assert self._post_wav(client).status_code == 200

    def test_predict_response_has_genre(self, client):
        resp_data = self._post_wav(client).get_json()["data"]
        assert "genre" in resp_data
        assert isinstance(resp_data["genre"], str)

    def test_predict_confidence_in_range(self, client):
        resp_data = self._post_wav(client).get_json()["data"]
        conf = resp_data["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_predict_has_candidates(self, client):
        resp_data = self._post_wav(client).get_json()["data"]
        assert isinstance(resp_data["candidates"], list)
        assert len(resp_data["candidates"]) > 0

    def test_predict_has_features(self, client):
        resp_data = self._post_wav(client).get_json()["data"]
        assert "features" in resp_data
        assert "tempo" in resp_data["features"]

    def test_predict_has_processing_time(self, client):
        resp_data = self._post_wav(client).get_json()["data"]
        assert "processing_time_ms" in resp_data
        assert resp_data["processing_time_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Root endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_has_name(self, client):
        data = client.get("/").get_json()
        assert "name" in data
