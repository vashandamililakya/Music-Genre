"""
app/feature_extractor.py
========================
Extracts a rich, fixed-length feature vector from an audio file.

Features extracted
------------------
- MFCCs                : 40 coefficients × (mean + std)  = 80
- Chroma STFT          : 12 bins × (mean + std)          = 24
- Spectral Centroid     : mean + std                      =  2
- Spectral Rolloff      : mean + std                      =  2
- Spectral Bandwidth    : mean + std                      =  2
- Spectral Contrast     : 7 bands × (mean + std)          = 14
- Zero Crossing Rate    : mean + std                      =  2
- RMS Energy            : mean + std                      =  2
- Tonnetz               : 6 dims × (mean + std)           = 12
- Tempo                 : scalar                          =  1
- Beat strength         : mean                            =  1
─────────────────────────────────────────────────────────────────
Total                                                     = 142

Notes
-----
- If librosa is not available the module falls back to a scipy-only
  approximation (MFCCs via DCT + basic spectral stats).  This is
  accurate enough for demos but you should install librosa for
  production use.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy imports (graceful degradation) ────────────────────────────
try:
    import librosa
    import librosa.feature
    import librosa.effects
    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    _HAS_LIBROSA = False
    logger.warning(
        "librosa not installed — falling back to scipy-only feature extraction. "
        "Run `pip install librosa soundfile` for full accuracy."
    )

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except ImportError:
    _HAS_PYDUB = False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM = 142  # must stay in sync with extract_features() output


def extract_features(
    file_path: str | Path,
    sample_rate: int = 22050,
    clip_duration: float = 30.0,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Load *file_path* and return a 1-D float32 feature vector of length
    ``FEATURE_DIM``.

    Parameters
    ----------
    file_path     : path to audio file (.wav / .mp3 / .flac / .ogg)
    sample_rate   : target sample rate (audio is resampled if needed)
    clip_duration : how many seconds to analyse (centre-crop or pad)
    n_mfcc        : number of MFCC coefficients
    n_fft         : FFT window size
    hop_length    : STFT hop length

    Returns
    -------
    np.ndarray, shape (FEATURE_DIM,), dtype float32
    """
    file_path = Path(file_path)

    if _HAS_LIBROSA:
        return _extract_librosa(
            file_path, sample_rate, clip_duration, n_mfcc, n_fft, hop_length
        )
    else:
        return _extract_scipy(file_path, sample_rate, clip_duration)


# ─────────────────────────────────────────────────────────────────────────────
# Librosa implementation (full accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_librosa(
    file_path: Path,
    sr: int,
    duration: float,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Full feature extraction pipeline using librosa."""
    y = _load_audio(file_path, sr, duration)

    features: list[float] = []

    # 1. MFCCs  (40 × 2 = 80)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    features.extend(_stat(mfcc))

    # 2. Chroma STFT  (12 × 2 = 24)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.extend(_stat(chroma))

    # 3. Spectral Centroid  (1 × 2 = 2)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.extend(_stat(cent))

    # 4. Spectral Rolloff  (1 × 2 = 2)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.extend(_stat(rolloff))

    # 5. Spectral Bandwidth  (1 × 2 = 2)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.extend(_stat(bw))

    # 6. Spectral Contrast  (7 × 2 = 14)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features.extend(_stat(contrast))

    # 7. Zero Crossing Rate  (1 × 2 = 2)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    features.extend(_stat(zcr))

    # 8. RMS Energy  (1 × 2 = 2)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    features.extend(_stat(rms))

    # 9. Tonnetz  (6 × 2 = 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    features.extend(_stat(tonnetz))

    # 10. Tempo  (scalar = 1)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    # 11. Beat strength mean  (scalar = 1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(float(onset_env.mean()))

    vec = np.array(features, dtype=np.float32)

    if vec.shape[0] != FEATURE_DIM:
        logger.warning(
            "Feature vector length %d != expected %d — padding/truncating.",
            vec.shape[0], FEATURE_DIM,
        )
        vec = _fix_length(vec, FEATURE_DIM)

    return vec


# ─────────────────────────────────────────────────────────────────────────────
# SciPy fallback implementation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scipy(file_path: Path, sr: int, duration: float) -> np.ndarray:
    """
    Minimal feature extraction without librosa.
    Computes DCT-based pseudo-MFCCs + basic spectral stats.
    Result is padded/truncated to FEATURE_DIM.
    """
    from scipy.fft import fft, dct
    from scipy.io import wavfile
    import struct

    # We can only reliably read WAV without librosa
    try:
        file_sr, data = wavfile.read(str(file_path))
    except Exception as exc:
        raise RuntimeError(
            f"scipy fallback can only read WAV files. "
            f"Install librosa+soundfile for MP3/FLAC/OGG support. Original error: {exc}"
        ) from exc

    # Mono + normalise
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1.0:
        data /= np.iinfo(np.int16).max

    # Resample crudely if needed (integer ratio only)
    if file_sr != sr:
        ratio = sr / file_sr
        n_new = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, n_new).astype(int)
        data = data[indices]

    # Clip to duration
    n_samples = int(sr * duration)
    if len(data) > n_samples:
        start = (len(data) - n_samples) // 2
        data = data[start: start + n_samples]
    else:
        data = np.pad(data, (0, max(0, n_samples - len(data))))

    hop = 512
    n_fft = 2048
    frames = librosa_frame(data, n_fft, hop)  # (n_frames, n_fft)

    # Power spectrum per frame
    spectrum = np.abs(fft(frames, axis=1))[:, : n_fft // 2 + 1] ** 2

    # DCT → pseudo MFCCs (40 coefficients)
    log_spec = np.log(spectrum + 1e-9)
    mfcc_like = dct(log_spec, type=2, axis=1, norm="ortho")[:, :40]  # (n_frames, 40)

    features: list[float] = []
    features.extend(_stat(mfcc_like.T))  # 80

    # Spectral centroid
    freqs = np.linspace(0, sr / 2, spectrum.shape[1])
    centroid = (spectrum * freqs).sum(axis=1) / (spectrum.sum(axis=1) + 1e-9)
    features.extend([centroid.mean(), centroid.std()])  # 2

    # Spectral rolloff (85 %)
    cumsum = np.cumsum(spectrum, axis=1)
    threshold = 0.85 * spectrum.sum(axis=1, keepdims=True)
    rolloff_idx = (cumsum < threshold).sum(axis=1)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]
    features.extend([rolloff.mean(), rolloff.std()])  # 2

    # ZCR
    zcr = ((np.diff(np.sign(frames), axis=1) != 0).sum(axis=1) / n_fft)
    features.extend([zcr.mean(), zcr.std()])  # 2

    # RMS
    rms = np.sqrt((frames ** 2).mean(axis=1))
    features.extend([rms.mean(), rms.std()])  # 2

    vec = np.array(features, dtype=np.float32)
    return _fix_length(vec, FEATURE_DIM)


def librosa_frame(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Simple framing without librosa."""
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stat(arr: np.ndarray) -> list[float]:
    """Return [mean_row0, ..., mean_rowN, std_row0, ..., std_rowN] for a 2-D array."""
    return list(arr.mean(axis=1)) + list(arr.std(axis=1))


def _fix_length(vec: np.ndarray, target: int) -> np.ndarray:
    if len(vec) >= target:
        return vec[:target]
    return np.pad(vec, (0, target - len(vec)))


def _load_audio(file_path: Path, sr: int, duration: float) -> np.ndarray:
    """
    Load audio to a mono float32 numpy array, resampled to *sr*,
    centre-cropped / zero-padded to *duration* seconds.
    Tries librosa → soundfile → pydub in order.
    """
    ext = file_path.suffix.lower().lstrip(".")

    y: Optional[np.ndarray] = None

    # librosa handles WAV/FLAC/OGG natively; MP3 needs ffmpeg or audioread
    if _HAS_LIBROSA:
        try:
            y, _ = librosa.load(str(file_path), sr=sr, mono=True, duration=duration)
        except Exception as exc:
            logger.debug("librosa.load failed (%s), trying soundfile", exc)

    if y is None and _HAS_SOUNDFILE and ext in {"wav", "flac", "ogg"}:
        try:
            data, file_sr = sf.read(str(file_path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if file_sr != sr:
                import librosa as _lb
                data = _lb.resample(data, orig_sr=file_sr, target_sr=sr)
            y = data
        except Exception as exc:
            logger.debug("soundfile failed (%s)", exc)

    if y is None and _HAS_PYDUB and ext == "mp3":
        try:
            seg = AudioSegment.from_mp3(str(file_path))
            seg = seg.set_frame_rate(sr).set_channels(1)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            y = samples / (2 ** (seg.sample_width * 8 - 1))
        except Exception as exc:
            logger.debug("pydub failed (%s)", exc)

    if y is None:
        raise RuntimeError(
            f"Could not decode '{file_path.name}'. "
            "Install librosa + soundfile (and ffmpeg for MP3 support)."
        )

    # Centre-crop or zero-pad to exactly *duration* seconds
    target_len = int(sr * duration)
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start: start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    return y.astype(np.float32)
