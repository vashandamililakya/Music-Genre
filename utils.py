"""
app/utils.py
============
File validation, upload helpers, and response builders.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from flask import jsonify

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# File validation
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "ogg", "aiff", "m4a"}

# Magic-byte signatures for audio formats
# Format: (byte_offset, byte_sequence)
_MAGIC_BYTES: list[tuple[int, bytes]] = [
    (0,  b"RIFF"),           # WAV
    (0,  b"fLaC"),           # FLAC
    (0,  b"OggS"),           # OGG
    (0,  b"ID3"),            # MP3 ID3 tag
    (0,  b"\xff\xfb"),       # MP3 frame sync (CBR)
    (0,  b"\xff\xf3"),       # MP3 frame sync
    (0,  b"\xff\xf2"),       # MP3 frame sync
    (4,  b"ftyp"),           # M4A / AAC
]


def allowed_file(filename: str) -> bool:
    """Return True if *filename* has an allowed audio extension."""
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in ALLOWED_EXTENSIONS


def validate_audio_file(stream: bytes) -> tuple[bool, str]:
    """
    Validate that *stream* (first few bytes of the file) looks like audio.

    Returns
    -------
    (is_valid, reason_if_invalid)
    """
    if len(stream) < 16:
        return False, "File too small to be a valid audio file."

    for offset, magic in _MAGIC_BYTES:
        end = offset + len(magic)
        if len(stream) >= end and stream[offset:end] == magic:
            return True, ""

    return False, "File does not appear to be a supported audio format."


def secure_temp_path(upload_folder: Path, original_filename: str) -> Path:
    """
    Generate a unique, safe temporary file path.
    Uses UUID so filenames cannot collide or be guessed.
    """
    ext = Path(original_filename).suffix.lower()
    if not ext:
        ext = ".wav"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    return upload_folder / unique_name


def cleanup_file(path: Path | str) -> None:
    """Silently remove a file if it exists."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Could not delete temp file %s: %s", path, exc)


def cleanup_old_uploads(upload_folder: Path, max_age_seconds: int = 3600) -> int:
    """
    Delete files in *upload_folder* older than *max_age_seconds*.
    Returns the count of deleted files.
    """
    now = time.time()
    deleted = 0
    try:
        for f in upload_folder.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
                f.unlink(missing_ok=True)
                deleted += 1
    except Exception as exc:
        logger.warning("Upload cleanup error: %s", exc)
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Response builders
# ─────────────────────────────────────────────────────────────────────────────

def success_response(data: dict, status: int = 200):
    """Wrap *data* in a standard success envelope."""
    return jsonify({"status": "success", "data": data}), status


def error_response(message: str, status: int = 400, details: Optional[str] = None):
    """Wrap *message* in a standard error envelope."""
    body: dict = {"status": "error", "message": message}
    if details:
        body["details"] = details
    return jsonify(body), status


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of a file (for dedup / caching)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
