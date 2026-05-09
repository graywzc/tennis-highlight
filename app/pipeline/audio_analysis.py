"""Audio impact (ball-strike) detection.

Tennis racket-on-ball impacts produce a sharp, high-frequency transient that's
trivial to spot once the band of interest is isolated. Pipeline:

  ffmpeg → mono float32 PCM
  → Butterworth bandpass (default 1–8 kHz)
  → short-window RMS envelope
  → adaptive median+MAD threshold + scipy.signal.find_peaks

The 1–8 kHz band emphasises racket clicks while suppressing ball bounces
(<800 Hz thud) and most voice fundamentals.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str, float | None], None]


def analyze_audio_impacts(
    video_path: Path,
    knobs: dict,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    """Decode audio and return candidate ball-impact times."""
    sr = int(knobs["audio_sample_rate"])
    if progress_cb:
        progress_cb(0.0, "extracting audio for impact detection", None)
    samples = extract_audio_pcm(video_path, sr)
    duration_s = len(samples) / sr if sr > 0 else 0.0

    if progress_cb:
        progress_cb(5.0, f"bandpass {int(knobs['bandpass_low_hz'])}–{int(knobs['bandpass_high_hz'])} Hz", None)
    filtered = bandpass(
        samples,
        sr,
        float(knobs["bandpass_low_hz"]),
        float(knobs["bandpass_high_hz"]),
    )

    if progress_cb:
        progress_cb(8.0, "computing audio envelope", None)
    env = envelope(filtered, sr)

    if progress_cb:
        progress_cb(10.0, "finding impact peaks", None)
    impacts, noise_floor = find_impacts(env, sr, knobs, raw_samples=samples)

    return {
        "sample_rate": sr,
        "duration_s": float(duration_s),
        "noise_floor": float(noise_floor),
        "impacts": impacts,
    }


def analyze_audio_impacts_range(
    video_path: Path,
    range_start_s: float,
    range_end_s: float,
    knobs: dict,
    *,
    boundary_pad_s: float = 0.5,
) -> dict:
    """Re-run impact detection on a sub-range of the video.

    Decodes a slightly larger window than [start, end] so the bandpass
    sosfiltfilt edge transients don't show up as fake peaks at the boundaries,
    then trims the impact list back to the requested range. Times are
    returned in *absolute* video time.
    """
    if range_end_s <= range_start_s:
        raise ValueError("range_end_s must be > range_start_s")
    sr = int(knobs["audio_sample_rate"])
    decode_start = max(0.0, float(range_start_s) - boundary_pad_s)
    decode_end = float(range_end_s) + boundary_pad_s
    samples = extract_audio_pcm_range(video_path, decode_start, decode_end, sr)
    if samples.size == 0:
        return {
            "sample_rate": sr,
            "range_start_s": float(range_start_s),
            "range_end_s": float(range_end_s),
            "noise_floor": 0.0,
            "impacts": [],
        }
    filtered = bandpass(
        samples,
        sr,
        float(knobs["bandpass_low_hz"]),
        float(knobs["bandpass_high_hz"]),
    )
    env = envelope(filtered, sr)
    raw_impacts, noise_floor = find_impacts(env, sr, knobs, raw_samples=samples)
    # Convert peak-relative times to absolute video time and clip to range.
    absolute = []
    for imp in raw_impacts:
        abs_time = decode_start + imp["time_s"]
        if range_start_s <= abs_time < range_end_s:
            absolute.append({**imp, "time_s": float(abs_time)})
    return {
        "sample_rate": sr,
        "range_start_s": float(range_start_s),
        "range_end_s": float(range_end_s),
        "noise_floor": float(noise_floor),
        "impacts": absolute,
    }


def extract_audio_pcm(video_path: Path, sample_rate: int = 22050) -> np.ndarray:
    """Decode video's audio track to mono float32 PCM via ffmpeg."""
    return _ffmpeg_pcm(["-i", str(video_path)], sample_rate)


def extract_audio_pcm_range(
    video_path: Path,
    t_start: float,
    t_end: float,
    sample_rate: int = 22050,
) -> np.ndarray:
    """Decode just [t_start, t_end] seconds of audio."""
    duration = max(0.0, float(t_end) - float(t_start))
    if duration <= 0:
        return np.zeros(0, dtype=np.float32)
    return _ffmpeg_pcm(
        [
            "-ss", f"{float(t_start):.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
        ],
        sample_rate,
    )


def _ffmpeg_pcm(input_args: list[str], sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        *input_args,
        "-vn",
        "-ac", "1",
        "-ar", str(int(sample_rate)),
        "-f", "f32le",
        "-",
    ]
    chunks: list[bytes] = []
    with tempfile.TemporaryFile(mode="w+b") as stderr_tmp:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_tmp,
            bufsize=1024 * 1024,
        )
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait(timeout=60)
        if proc.returncode != 0:
            stderr_tmp.seek(0)
            err = stderr_tmp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg audio decode failed: {err.strip()[-500:]}")

    raw = b"".join(chunks)
    if not raw:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(raw, dtype=np.float32)
    return np.ascontiguousarray(arr)


def bandpass(
    samples: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass via SOS+sosfiltfilt."""
    if samples.size == 0:
        return samples.astype(np.float32)
    nyq = sr / 2.0
    low = max(1e-6, min(low_hz, nyq - 1.0)) / nyq
    high = max(low + 1e-6, min(high_hz, nyq - 1.0)) / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    # sosfiltfilt needs float64 internally for stability; cast back to float32.
    out = sosfiltfilt(sos, samples.astype(np.float64))
    return out.astype(np.float32)


def envelope(filtered: np.ndarray, sr: int, win_ms: float = 5.0) -> np.ndarray:
    """Short-window RMS envelope (rectify + boxcar smoothing of squared signal)."""
    if filtered.size == 0:
        return filtered.astype(np.float32)
    win = max(1, int(round(sr * win_ms / 1000.0)))
    sq = np.square(filtered.astype(np.float64))
    smoothed = uniform_filter1d(sq, size=win, mode="nearest")
    return np.sqrt(np.maximum(smoothed, 0.0)).astype(np.float32)


def find_impacts(
    env: np.ndarray,
    sr: int,
    knobs: dict,
    *,
    raw_samples: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    """Adaptive peak detection on the envelope.

    When ``raw_samples`` is provided, computes spectral centroid for each
    detected peak from a 30 ms window of the original (un-bandpassed) audio
    so the value reflects the true spectral shape of the transient (useful
    for distinguishing tennis impacts ~3–5 kHz from basketball thumps
    ~0.5–1.5 kHz). If ``min_spectral_centroid_hz`` > 0, peaks below that
    are dropped.
    """
    if env.size == 0:
        return [], 0.0
    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median)))
    k = float(knobs["peak_height_mad_k"])
    floor = max(median + k * mad, 1e-6)
    prom_mult = float(knobs["peak_prominence_mult"])
    distance = max(1, int(round(sr * float(knobs["min_impact_separation_s"]))))

    peaks, props = find_peaks(
        env,
        height=floor,
        prominence=floor * prom_mult,
        distance=distance,
    )
    if peaks.size == 0:
        return [], floor

    heights = props.get("peak_heights")
    if heights is None:
        heights = env[peaks]
    min_centroid = float(knobs.get("min_spectral_centroid_hz", 0.0))
    impacts: list[dict] = []
    for idx, peak_idx in enumerate(peaks):
        amp = float(heights[idx])
        centroid = 0.0
        if raw_samples is not None and raw_samples.size > 0:
            centroid = spectral_centroid_at(raw_samples, sr, int(peak_idx))
        if min_centroid > 0 and centroid < min_centroid:
            continue
        impacts.append({
            "time_s": float(peak_idx) / sr,
            "amplitude": amp,
            "snr": amp / floor if floor > 0 else 0.0,
            "spectral_centroid_hz": float(centroid),
        })
    return impacts, floor


def spectral_centroid_at(
    samples: np.ndarray,
    sr: int,
    center_idx: int,
    *,
    win_ms: float = 30.0,
    low_hz: float = 200.0,
    high_hz: float = 9000.0,
) -> float:
    """Spectral centroid (Hz) over a Hann-windowed snippet around ``center_idx``.

    Computed in the 200 Hz–9 kHz band so we capture both the basketball-thump
    energy (~0.5–1.5 kHz) and the racket-impact "ping" (~3–5 kHz) without
    being skewed by HVAC rumble or ultrasonic noise.
    """
    n = samples.size
    if n == 0:
        return 0.0
    half = max(8, int(round(sr * win_ms / 2000.0)))
    a = max(0, center_idx - half)
    b = min(n, center_idx + half)
    win = samples[a:b]
    if win.size < 16:
        return 0.0
    w = np.hanning(win.size)
    spectrum = np.abs(np.fft.rfft(win * w))
    freqs = np.fft.rfftfreq(win.size, d=1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    s = spectrum[mask]
    f = freqs[mask]
    total = float(s.sum())
    if total <= 0.0:
        return 0.0
    return float(np.sum(f * s) / total)
