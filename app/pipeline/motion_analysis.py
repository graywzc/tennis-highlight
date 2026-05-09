"""Motion analysis using median-frame background subtraction.

Why median-frame instead of MOG2:
  MOG2 adapts its background model frame-by-frame, so on long footage a player
  who rallies from a similar spot for minutes gradually gets incorporated into
  the "background" — their motion stops registering. The median-frame method
  picks a fixed background derived from the per-pixel median over the whole
  clip. As long as the players move around the court (which they do during
  practice), no single pixel is occupied by them most of the time, so the
  median naturally captures the empty court. Foreground detection is then a
  simple, non-adaptive |frame - median| > threshold check that doesn't suffer
  from learn-in or warmup.
"""

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# Width of the downscaled processing stream. 480 px is plenty for foreground
# blob detection on a wide court shot and keeps median computation fast.
TARGET_WIDTH = 480
DETECTOR_NAME = "median_frame"
COURT_DETECTOR_NAME = "median_court_roi"
DETECTOR_VERSION = 1


def default_knobs() -> dict:
    return {
        "diff_threshold": settings.diff_threshold,
        "motion_threshold": settings.motion_threshold,
        "merge_gap_s": settings.merge_gap_s,
        "min_segment_s": settings.min_segment_s,
        "segment_padding_s": settings.segment_padding_s,
        "sample_fps": settings.sample_fps,
        "median_bg_samples": settings.median_bg_samples,
        "enable_merge_gap": True,
        "enable_min_segment": True,
        "enable_padding": True,
        "court_weight": 1.0,
        "outside_weight": 0.15,
        "near_camera_weight": 0.0,
        "audio_sample_rate": 22050,
        "bandpass_low_hz": 1000.0,
        "bandpass_high_hz": 8000.0,
        "peak_height_mad_k": 6.0,
        "peak_prominence_mult": 2.0,
        "min_impact_separation_s": 0.15,
        "min_spectral_centroid_hz": 2500.0,
        "pose_window_s": 0.75,
        "wrist_conf_min": 0.3,
        "min_wrist_velocity": 0.4,
        "max_gap_s": 5.0,
        "min_hits_per_rally": 2,
        "rally_padding_s": 1.0,
    }


def analyze_video_to_artifact(
    analysis_id: str,
    video_path: Path,
    video_duration_s: float,
    progress_cb: ProgressCallback | None = None,
    knobs_override: dict | None = None,
    algorithm: str = DETECTOR_NAME,
    court_calibration: dict | None = None,
) -> tuple[Path, list[dict], dict]:
    """Run the expensive detector once and persist reusable per-frame histograms."""
    info = _probe_video(video_path)
    src_w, src_h = info["width"], info["height"]
    target_w = TARGET_WIDTH
    target_h = max(2, ((src_h * target_w) // src_w) & ~1)

    knobs = normalize_knobs(knobs_override, default_knobs())
    sample_fps = float(knobs["sample_fps"])
    total_samples = max(1, int(round(video_duration_s * sample_fps)))

    logger.info("motion analysis starting: %s", video_path.name)
    logger.info(
        "  source: %dx%d, %.2f fps, codec=%s, duration=%.1fs",
        src_w, src_h, info["fps"], info["codec"], video_duration_s,
    )
    logger.info(
        "  pipeline: scale=%dx%d, sample_fps=%.1f, expected ~%d samples",
        target_w, target_h, sample_fps, total_samples,
    )

    if progress_cb is not None:
        progress_cb(0.0, "[1/2] building static background...")

    t0 = time.monotonic()
    median_bg = _compute_median_background(
        video_path, video_duration_s, target_w, target_h,
        int(knobs["median_bg_samples"]), progress_cb,
    )
    logger.info(
        "phase 1 (median background): %d samples in %.1fs",
        knobs["median_bg_samples"], time.monotonic() - t0,
    )

    if progress_cb is not None:
        progress_cb(50.0, "[2/2] scanning frames against background...")

    roi_weights = None
    if algorithm == COURT_DETECTOR_NAME:
        if not court_calibration:
            raise RuntimeError("court calibration is required for median_court_roi")
        roi_weights = _court_roi_weights(
            target_w,
            target_h,
            court_calibration,
            float(knobs["court_weight"]),
            float(knobs["outside_weight"]),
            float(knobs["near_camera_weight"]),
        )

    motion, histograms, weighted_histograms = _scan_frames_against_background(
        video_path, target_w, target_h, sample_fps, median_bg, total_samples,
        progress_cb, roi_weights=roi_weights,
    )

    _log_foreground_distribution(motion)

    times = np.array([t for t, _ in motion], dtype=np.float32)
    ratios = np.array([r for _, r in motion], dtype=np.float32)
    smoothed = _moving_average_by_time(
        times.astype(np.float64), ratios.astype(np.float64), window_s=3.0
    ).astype(np.float32)
    metadata = {
        "detector": algorithm,
        "detector_version": DETECTOR_VERSION,
        "duration_s": float(video_duration_s),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "sample_fps": sample_fps,
        "median_bg_samples": int(knobs["median_bg_samples"]),
        "sample_count": int(len(times)),
        "default_knobs": knobs,
        "foreground_percentiles": _foreground_percentiles(ratios),
        "court_calibration": court_calibration,
    }

    settings.analysis_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = settings.analysis_dir / f"{analysis_id}.npz"
    np.savez_compressed(
        artifact_path,
        metadata_json=np.array(json.dumps(metadata)),
        times=times,
        histograms=np.asarray(histograms, dtype=np.uint32),
        weighted_histograms=np.asarray(weighted_histograms, dtype=np.float32) if weighted_histograms else np.zeros((0, 256), dtype=np.float32),
        default_ratios=ratios,
        default_smoothed=smoothed,
    )

    result = recompute_from_artifact(artifact_path, knobs, include_samples=True)
    logger.info("motion analysis: %d in-play segments detected", result["summary"]["on_count"])
    return artifact_path, result["segments"], result


def detect_in_play_segments(
    video_path: Path,
    video_duration_s: float,
    progress_cb: ProgressCallback | None = None,
) -> list[tuple[float, float]]:
    artifact_path, segments, _summary = analyze_video_to_artifact(
        "adhoc", video_path, video_duration_s, progress_cb
    )
    try:
        artifact_path.unlink(missing_ok=True)
    except Exception:
        logger.debug("failed to remove temporary adhoc artifact", exc_info=True)
    return [
        (float(s["start_s"]), float(s["end_s"]))
        for s in segments
        if s.get("is_on")
    ]


# -- Phase 1: median background -------------------------------------------

def _compute_median_background(
    video_path: Path,
    duration_s: float,
    target_w: int,
    target_h: int,
    n_samples: int,
    progress_cb: ProgressCallback | None = None,
) -> np.ndarray:
    """Sample n frames evenly across the video and take per-pixel median.

    Returns an HxWx3 uint8 image of what the empty court looks like.
    """
    if duration_s <= 0:
        raise RuntimeError("invalid duration for background sampling")
    bg_fps = max(0.01, n_samples / duration_s)

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-hwaccel", "videotoolbox",
        "-i", str(video_path),
        "-an",
        "-vf", f"fps={bg_fps:.6f},scale={target_w}:{target_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    logger.debug("$ %s", " ".join(cmd))

    frame_bytes = target_w * target_h * 3
    samples: list[np.ndarray] = []
    start_time = time.monotonic()
    last_report = start_time
    interval = settings.progress_report_interval_s

    with tempfile.TemporaryFile(mode="w+b") as stderr_tmp:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_tmp,
            bufsize=frame_bytes * 4,
        )
        assert proc.stdout is not None
        try:
            while True:
                buf = _read_exact(proc.stdout, frame_bytes)
                if buf is None:
                    break
                samples.append(
                    np.frombuffer(buf, dtype=np.uint8).reshape(target_h, target_w, 3)
                )

                now = time.monotonic()
                if now - last_report >= interval:
                    elapsed = now - start_time
                    rate = len(samples) / elapsed if elapsed > 0 else 0.0
                    frac = len(samples) / n_samples
                    pct = min(49.0, frac * 50.0)  # phase 1 gets 0-50%
                    remaining = max(0, n_samples - len(samples))
                    eta = remaining / rate if rate > 0 else 0.0
                    msg = (
                        f"[1/2] background frame {len(samples)}/{n_samples} "
                        f"({frac * 100:.0f}%) — ETA {_fmt_eta(eta)}"
                    )
                    logger.info("phase 1 progress: %s", msg)
                    if progress_cb is not None:
                        try:
                            progress_cb(pct, msg)
                        except Exception:
                            logger.exception("progress_cb raised")
                    last_report = now
        finally:
            proc.stdout.close()
            proc.wait(timeout=30)
        if proc.returncode != 0:
            stderr_tmp.seek(0)
            err = stderr_tmp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg phase-1 failed: {err.strip()[-500:]}")

    if not samples:
        raise RuntimeError("no frames extracted for background sampling")

    stack = np.stack(samples, axis=0)  # (n, H, W, 3)
    median = np.median(stack, axis=0).astype(np.uint8)
    logger.debug(
        "  median background: stacked %d frames (%.1f MB)",
        len(samples), stack.nbytes / 1e6,
    )
    return median


# -- Phase 2: frame scan --------------------------------------------------

def _scan_frames_against_background(
    video_path: Path,
    target_w: int,
    target_h: int,
    sample_fps: float,
    median_bg: np.ndarray,
    total_samples: int,
    progress_cb: ProgressCallback | None,
    roi_weights: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], list[np.ndarray], list[np.ndarray]]:
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-hwaccel", "videotoolbox",
        "-i", str(video_path),
        "-an",
        "-vf", f"fps={sample_fps},scale={target_w}:{target_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    logger.debug("$ %s", " ".join(cmd))

    frame_bytes = target_w * target_h * 3
    diff_thr = settings.diff_threshold
    motion: list[tuple[float, float]] = []
    histograms: list[np.ndarray] = []
    weighted_histograms: list[np.ndarray] = []
    sample_idx = 0
    start_time = time.monotonic()
    last_report = start_time
    interval = settings.progress_report_interval_s

    with tempfile.TemporaryFile(mode="w+b") as stderr_tmp:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_tmp,
            bufsize=frame_bytes * 4,
        )
        assert proc.stdout is not None
        try:
            while True:
                buf = _read_exact(proc.stdout, frame_bytes)
                if buf is None:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(target_h, target_w, 3)
                # Per-pixel max channel difference vs the static background.
                diff = cv2.absdiff(frame, median_bg)
                gray_diff = diff.max(axis=2)  # H x W, 0-255
                hist = np.bincount(gray_diff.ravel(), minlength=256).astype(np.uint32)
                histograms.append(hist)
                if roi_weights is not None:
                    weighted_hist = np.bincount(
                        gray_diff.ravel(),
                        weights=roi_weights.ravel(),
                        minlength=256,
                    ).astype(np.float32)
                    weighted_histograms.append(weighted_hist)
                    ratio = _ratio_from_histogram(weighted_hist, diff_thr)
                else:
                    ratio = _ratio_from_histogram(hist, diff_thr)
                ts = sample_idx / sample_fps
                motion.append((ts, ratio))
                logger.debug(
                    "  sample @ %.1fs (#%d): foreground=%.3f", ts, sample_idx, ratio
                )
                sample_idx += 1

                now = time.monotonic()
                if now - last_report >= interval:
                    elapsed = now - start_time
                    rate = sample_idx / elapsed if elapsed > 0 else 0.0
                    frac = sample_idx / total_samples
                    pct = min(98.0, 50.0 + frac * 48.0)  # phase 2 gets 50-98%
                    remaining = max(0, total_samples - sample_idx)
                    eta = remaining / rate if rate > 0 else 0.0
                    msg = (
                        f"[2/2] sample {sample_idx:,}/{total_samples:,} "
                        f"({frac * 100:.0f}%) — {rate:.0f} samples/s — "
                        f"ETA {_fmt_eta(eta)}"
                    )
                    logger.info("phase 2 progress: %s", msg)
                    if progress_cb is not None:
                        try:
                            progress_cb(pct, msg)
                        except Exception:
                            logger.exception("progress_cb raised")
                    last_report = now
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait(timeout=30)

        if proc.returncode != 0:
            stderr_tmp.seek(0)
            err = stderr_tmp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg phase-2 failed: {err.strip()[-500:]}")

    elapsed = time.monotonic() - start_time
    logger.info(
        "phase 2 (scan) done: %d samples in %.1fs (%.0f samples/s)",
        sample_idx, elapsed, sample_idx / elapsed if elapsed > 0 else 0,
    )
    return motion, histograms, weighted_histograms


# -- Helpers --------------------------------------------------------------

def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)
        buf.extend(chunk)
    return bytes(buf)


def _probe_video(video_path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-print_format", "json", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError("no video stream found")
    s = streams[0]
    fps_str = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/0"
    if "/" in fps_str:
        n, d = fps_str.split("/")
        fps = float(n) / float(d) if float(d) > 0 else 0.0
    else:
        fps = float(fps_str)
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": fps,
        "codec": s.get("codec_name", "unknown"),
    }


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = seconds / 60
    if mins < 60:
        return f"{mins:.1f}m"
    hours = mins / 60
    return f"{hours:.1f}h"


def _log_foreground_distribution(motion: list[tuple[float, float]]) -> None:
    if not motion:
        return
    arr = np.array([r for _, r in motion], dtype=np.float64)
    pct_levels = [0, 25, 50, 75, 90, 95, 99, 100]
    p_vals = np.percentile(arr, pct_levels)
    above = float((arr >= settings.motion_threshold).mean()) * 100
    logger.info(
        "foreground distribution (n=%d): "
        "min=%.4f p25=%.4f p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f max=%.4f",
        arr.size, *p_vals,
    )
    logger.info(
        "  current MOTION_THRESHOLD=%.4f → %.1f%% of samples qualify as in-play",
        settings.motion_threshold, above,
    )
    top_idx = np.argsort([r for _, r in motion])[-5:][::-1]
    hot = [motion[i] for i in top_idx]
    logger.info(
        "  hottest samples: %s",
        ", ".join(f"{t:.1f}s={r:.3f}" for t, r in hot),
    )
    for share, label in [(0.10, "~10%"), (0.20, "~20%"), (0.30, "~30%")]:
        thr = float(np.percentile(arr, (1 - share) * 100))
        logger.info("  for %s in-play set MOTION_THRESHOLD=%.4f", label, thr)


def _foreground_percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    levels = [0, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(values.astype(np.float64), levels)
    return {f"p{level}": float(val) for level, val in zip(levels, vals)}


def _court_roi_weights(
    width: int,
    height: int,
    calibration: dict,
    court_weight: float,
    outside_weight: float,
    near_camera_weight: float,
) -> np.ndarray:
    points = calibration.get("points") or []
    if len(points) != 4:
        raise RuntimeError("court calibration requires exactly 4 points")
    weights = np.full((height, width), float(outside_weight), dtype=np.float32)
    poly = np.array(
        [[[float(p["x"]) * width, float(p["y"]) * height] for p in points]],
        dtype=np.int32,
    )
    cv2.fillPoly(weights, poly, float(court_weight))
    # Bottom band catches near-camera pickup/walking when the camera is behind
    # or near the baseline. This is intentionally simple for v1 and tunable.
    band_start = int(height * 0.78)
    if band_start < height:
        weights[band_start:, :] = np.minimum(weights[band_start:, :], float(near_camera_weight))
    return weights


def _ratio_from_histogram(hist: np.ndarray, diff_threshold: int) -> float:
    threshold = int(diff_threshold)
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    if threshold < 0:
        foreground = total
    elif threshold >= 255:
        foreground = 0
    else:
        foreground = int(hist[threshold + 1:].sum())
    return float(foreground / total)


def ratios_from_histograms(histograms: np.ndarray, diff_threshold: int) -> np.ndarray:
    if histograms.size == 0:
        return np.array([], dtype=np.float64)
    totals = histograms.sum(axis=1).astype(np.float64)
    if diff_threshold < 0:
        foreground = totals
    elif diff_threshold >= 255:
        foreground = np.zeros_like(totals)
    else:
        foreground = histograms[:, int(diff_threshold) + 1:].sum(axis=1).astype(np.float64)
    return np.divide(foreground, totals, out=np.zeros_like(foreground), where=totals > 0)


def load_artifact(artifact_path: Path) -> dict:
    with np.load(artifact_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return {
            "metadata": metadata,
            "times": data["times"].astype(np.float64),
            "histograms": data["histograms"].astype(np.uint32),
            "weighted_histograms": data["weighted_histograms"].astype(np.float64) if "weighted_histograms" in data.files else np.zeros((0, 256), dtype=np.float64),
            "default_ratios": data["default_ratios"].astype(np.float64),
            "default_smoothed": data["default_smoothed"].astype(np.float64),
        }


def recompute_from_artifact(
    artifact_path: Path,
    knobs: dict,
    *,
    include_samples: bool = True,
) -> dict:
    artifact = load_artifact(artifact_path)
    metadata = artifact["metadata"]
    times = artifact["times"]
    histograms = artifact["weighted_histograms"] if artifact["weighted_histograms"].size else artifact["histograms"]
    duration = float(metadata["duration_s"])

    clean_knobs = normalize_knobs(knobs, metadata.get("default_knobs") or default_knobs())
    ratios = ratios_from_histograms(histograms, int(clean_knobs["diff_threshold"]))
    smoothed = _moving_average_by_time(times, ratios, window_s=3.0)
    segments = segments_from_scores(
        times,
        ratios,
        smoothed,
        duration,
        clean_knobs,
        include_samples=include_samples,
    )

    on_segments = [s for s in segments if s.get("is_on")]
    on_duration = sum(float(s["end_s"]) - float(s["start_s"]) for s in on_segments)
    summary = {
        "sample_count": int(len(times)),
        "on_count": int(len(on_segments)),
        "total_count": int(len(segments)),
        "on_duration_s": float(on_duration),
        "on_percent": float((on_duration / duration) * 100.0) if duration > 0 else 0.0,
        "foreground_percentiles": _foreground_percentiles(ratios),
        "smoothed_percentiles": _foreground_percentiles(smoothed),
        "changed_segment_count": None,
    }
    return {
        "metadata": metadata,
        "knobs": clean_knobs,
        "segments": segments,
        "summary": summary,
    }


def normalize_knobs(knobs: dict | None, defaults: dict | None = None) -> dict:
    base = dict(defaults or default_knobs())
    if knobs:
        base.update({k: v for k, v in knobs.items() if v is not None})
    base["diff_threshold"] = max(0, min(255, int(base["diff_threshold"])))
    base["motion_threshold"] = max(0.0, float(base["motion_threshold"]))
    base["merge_gap_s"] = max(0.0, float(base["merge_gap_s"]))
    base["min_segment_s"] = max(0.0, float(base["min_segment_s"]))
    base["segment_padding_s"] = max(0.0, float(base["segment_padding_s"]))
    base["sample_fps"] = max(0.1, float(base["sample_fps"]))
    base["median_bg_samples"] = max(1, int(base["median_bg_samples"]))
    base["court_weight"] = max(0.0, float(base.get("court_weight", 1.0)))
    base["outside_weight"] = max(0.0, float(base.get("outside_weight", 0.15)))
    base["near_camera_weight"] = max(0.0, float(base.get("near_camera_weight", 0.0)))
    base["enable_merge_gap"] = bool(base.get("enable_merge_gap", True))
    base["enable_min_segment"] = bool(base.get("enable_min_segment", True))
    base["enable_padding"] = bool(base.get("enable_padding", True))
    base["audio_sample_rate"] = max(8000, int(base.get("audio_sample_rate", 22050)))
    base["bandpass_low_hz"] = max(1.0, float(base.get("bandpass_low_hz", 1000.0)))
    base["bandpass_high_hz"] = max(
        base["bandpass_low_hz"] + 1.0,
        float(base.get("bandpass_high_hz", 8000.0)),
    )
    base["peak_height_mad_k"] = max(0.0, float(base.get("peak_height_mad_k", 6.0)))
    base["peak_prominence_mult"] = max(0.0, float(base.get("peak_prominence_mult", 2.0)))
    base["min_impact_separation_s"] = max(0.0, float(base.get("min_impact_separation_s", 0.15)))
    base["min_spectral_centroid_hz"] = max(0.0, float(base.get("min_spectral_centroid_hz", 2500.0)))
    base["pose_window_s"] = max(0.0, float(base.get("pose_window_s", 0.75)))
    base["wrist_conf_min"] = max(0.0, min(1.0, float(base.get("wrist_conf_min", 0.3))))
    base["min_wrist_velocity"] = max(0.0, float(base.get("min_wrist_velocity", 0.4)))
    base["max_gap_s"] = max(0.0, float(base.get("max_gap_s", 5.0)))
    base["min_hits_per_rally"] = max(1, int(base.get("min_hits_per_rally", 2)))
    base["rally_padding_s"] = max(0.0, float(base.get("rally_padding_s", 1.0)))
    return base


def segments_from_scores(
    times: np.ndarray,
    ratios: np.ndarray,
    smoothed: np.ndarray,
    video_duration_s: float,
    knobs: dict,
    *,
    include_samples: bool = True,
) -> list[dict]:
    if len(times) == 0:
        return []

    threshold = float(knobs["motion_threshold"])
    in_play = smoothed >= threshold
    raw: list[dict] = []
    start_idx: int | None = None
    for i, flag in enumerate(in_play):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            raw.append(_on_segment_from_indices(
                times, ratios, smoothed, start_idx, i - 1,
                "raw", include_samples, threshold,
            ))
            start_idx = None
    if start_idx is not None:
        raw.append(_on_segment_from_indices(
            times, ratios, smoothed, start_idx, len(times) - 1,
            "raw", include_samples, threshold,
        ))

    current = raw
    if knobs.get("enable_merge_gap", True):
        current = _merge_segment_dicts(current, float(knobs["merge_gap_s"]), "merged")

    if knobs.get("enable_min_segment", True):
        min_s = float(knobs["min_segment_s"])
        current = [
            {**seg, "decision_stage": "filtered"}
            for seg in current
            if (float(seg["end_s"]) - float(seg["start_s"])) >= min_s
        ]

    if knobs.get("enable_padding", True):
        pad = float(knobs["segment_padding_s"])
        current = [
            {
                **seg,
                "start_s": max(0.0, float(seg["start_s"]) - pad),
                "end_s": min(video_duration_s, float(seg["end_s"]) + pad),
                "decision_stage": "padded",
            }
            for seg in current
        ]
        current = _merge_segment_dicts(current, 0.0, "padded")

    on_segments = sorted(current, key=lambda s: float(s["start_s"]))
    return _derive_full_timeline_with_stats(
        on_segments, video_duration_s, times, ratios, smoothed, include_samples, threshold
    )


def _on_segment_from_indices(
    times: np.ndarray,
    ratios: np.ndarray,
    smoothed: np.ndarray,
    start_idx: int,
    end_idx: int,
    stage: str,
    include_samples: bool,
    threshold: float,
) -> dict:
    start_s = float(times[start_idx])
    end_s = float(times[end_idx])
    samples = _sample_rows(times, ratios, smoothed, start_s, end_s, threshold) if include_samples else []
    stats = _score_stats(smoothed[start_idx:end_idx + 1])
    return {
        "start_s": start_s,
        "end_s": end_s,
        "is_on": True,
        "source": "detector",
        "raw_start_s": start_s,
        "raw_end_s": end_s,
        "decision_stage": stage,
        "sample_count": int(end_idx - start_idx + 1),
        "samples": samples,
        **stats,
    }


def _score_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"avg_score": None, "max_score": None, "min_score": None}
    return {
        "avg_score": float(np.mean(values)),
        "max_score": float(np.max(values)),
        "min_score": float(np.min(values)),
    }


def _sample_rows(
    times: np.ndarray,
    ratios: np.ndarray,
    smoothed: np.ndarray,
    start_s: float,
    end_s: float,
    threshold: float,
) -> list[dict]:
    mask = (times >= start_s) & (times <= end_s)
    rows = []
    for t, raw, score in zip(times[mask], ratios[mask], smoothed[mask]):
        rows.append({
            "time_s": float(t),
            "foreground_ratio": float(raw),
            "smoothed_score": float(score),
            "threshold_result": bool(score >= threshold),
        })
    return rows


def _merge_segment_dicts(segments: list[dict], gap: float, stage: str) -> list[dict]:
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: float(s["start_s"]))
    out = [dict(sorted_segs[0])]
    for seg in sorted_segs[1:]:
        last = out[-1]
        if float(seg["start_s"]) - float(last["end_s"]) <= gap:
            combined_samples = (last.get("samples") or []) + (seg.get("samples") or [])
            scores = [float(s["smoothed_score"]) for s in combined_samples]
            last["end_s"] = max(float(last["end_s"]), float(seg["end_s"]))
            raw_starts = [v for v in [last.get("raw_start_s"), seg.get("raw_start_s")] if v is not None]
            raw_ends = [v for v in [last.get("raw_end_s"), seg.get("raw_end_s")] if v is not None]
            last["raw_start_s"] = min(raw_starts) if raw_starts else None
            last["raw_end_s"] = max(raw_ends) if raw_ends else None
            last["sample_count"] = int((last.get("sample_count") or 0) + (seg.get("sample_count") or 0))
            last["samples"] = combined_samples
            last["decision_stage"] = stage
            if scores:
                last["avg_score"] = float(sum(scores) / len(scores))
                last["max_score"] = float(max(scores))
                last["min_score"] = float(min(scores))
        else:
            out.append(dict(seg))
    for seg in out:
        seg["decision_stage"] = stage
    return out


def _derive_full_timeline_with_stats(
    in_play_segments: list[dict],
    video_duration_s: float,
    times: np.ndarray,
    ratios: np.ndarray,
    smoothed: np.ndarray,
    include_samples: bool,
    threshold: float,
) -> list[dict]:
    timeline: list[dict] = []
    cursor = 0.0
    for seg in in_play_segments:
        start = float(seg["start_s"])
        end = float(seg["end_s"])
        if start > cursor:
            timeline.append(_off_segment(cursor, start, times, ratios, smoothed, include_samples, threshold))
        timeline.append(seg)
        cursor = end
    if cursor < video_duration_s:
        timeline.append(_off_segment(cursor, video_duration_s, times, ratios, smoothed, include_samples, threshold))
    return timeline


def _off_segment(
    start_s: float,
    end_s: float,
    times: np.ndarray,
    ratios: np.ndarray,
    smoothed: np.ndarray,
    include_samples: bool,
    threshold: float,
) -> dict:
    mask = (times >= start_s) & (times <= end_s)
    samples = _sample_rows(times, ratios, smoothed, start_s, end_s, threshold) if include_samples else []
    stats = _score_stats(smoothed[mask])
    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "is_on": False,
        "source": "detector",
        "raw_start_s": None,
        "raw_end_s": None,
        "decision_stage": "off_gap",
        "sample_count": int(mask.sum()),
        "samples": samples,
        **stats,
    }


# -- Post-processing (unchanged) ------------------------------------------

def _segments_from_motion(
    motion: list[tuple[float, float]],
    video_duration_s: float,
) -> list[tuple[float, float]]:
    if not motion:
        return []

    threshold = settings.motion_threshold
    times = np.array([t for t, _ in motion], dtype=np.float64)
    ratios = np.array([r for _, r in motion], dtype=np.float64)
    smoothed = _moving_average_by_time(times, ratios, window_s=3.0)

    in_play = smoothed >= threshold

    raw: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, flag in enumerate(in_play):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            raw.append((times[start_idx], times[i - 1]))
            start_idx = None
    if start_idx is not None:
        raw.append((times[start_idx], times[-1]))

    logger.debug("  raw segments: %d, threshold=%.3f", len(raw), threshold)

    merged = _merge_close(raw, gap=settings.merge_gap_s)
    logger.debug("  after gap-merge (%.1fs): %d segments", settings.merge_gap_s, len(merged))

    filtered = [(s, e) for s, e in merged if (e - s) >= settings.min_segment_s]
    logger.debug("  after min-length filter (%.1fs): %d segments", settings.min_segment_s, len(filtered))

    pad = settings.segment_padding_s
    padded = [
        (max(0.0, s - pad), min(video_duration_s, e + pad)) for s, e in filtered
    ]
    return _merge_close(padded, gap=0.0)


def _moving_average_by_time(times: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    if len(times) == 0:
        return values
    half = window_s / 2.0
    out = np.empty_like(values)
    left = 0
    right = 0
    n = len(times)
    csum = np.concatenate(([0.0], np.cumsum(values)))
    for i, t in enumerate(times):
        while left < n and times[left] < t - half:
            left += 1
        while right < n and times[right] <= t + half:
            right += 1
        count = right - left
        out[i] = (csum[right] - csum[left]) / count if count > 0 else 0.0
    return out


def _merge_close(
    segments: list[tuple[float, float]], gap: float
) -> list[tuple[float, float]]:
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda x: x[0])
    out = [sorted_segs[0]]
    for s, e in sorted_segs[1:]:
        last_s, last_e = out[-1]
        if s - last_e <= gap:
            out[-1] = (last_s, max(last_e, e))
        else:
            out.append((s, e))
    return out


def derive_full_timeline(
    in_play_segments: list[tuple[float, float]],
    video_duration_s: float,
) -> list[dict]:
    timeline: list[dict] = []
    cursor = 0.0
    for start, end in in_play_segments:
        if start > cursor:
            timeline.append({"start_s": cursor, "end_s": start, "is_on": False})
        timeline.append({"start_s": start, "end_s": end, "is_on": True})
        cursor = end
    if cursor < video_duration_s:
        timeline.append({"start_s": cursor, "end_s": video_duration_s, "is_on": False})
    return timeline
