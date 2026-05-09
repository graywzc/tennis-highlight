"""Near-player hit study precomputation.

This analysis is intentionally a labeling/evidence workflow, not a final rally
detector. It samples pose, detects audio impact peaks, and builds per-window
features so marked near-player hits can be evaluated against plausible signals.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import cv2

from app.config import settings
from app.pipeline.audio_analysis import analyze_audio_impacts
from app.pipeline.pose_analysis import (
    DEFAULT_POSE_CONF,
    DEFAULT_POSE_MODEL,
    POSE_TARGET_WIDTH,
    _detections_from_result,
    _iter_sampled_frames,
    _probe_video,
    summarize_pose_frames,
)
from app.pipeline.rally_detection import default_rally_knobs
from app.pipeline.tracknet import run_tracknet_window

HIT_STUDY_ALGORITHM = "near_player_hit_study"
HIT_STUDY_ARTIFACT_VERSION = 1

ProgressCallback = Callable[[float, str, float | None], None]


def analyze_hit_study_to_artifact(
    analysis_id: str,
    video_path: Path,
    video_duration_s: float,
    sample_fps: float,
    range_start_s: float,
    range_end_s: float,
    progress_cb: ProgressCallback | None = None,
    model_name: str = DEFAULT_POSE_MODEL,
    conf_threshold: float = DEFAULT_POSE_CONF,
    image_size: int = POSE_TARGET_WIDTH,
    rally_knob_overrides: dict | None = None,
) -> tuple[Path, list[dict], dict]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required for near_player_hit_study. "
            "Install it with `.venv/bin/pip install ultralytics`."
        ) from e

    info = _probe_video(video_path)
    target_w = int(image_size)
    target_h = max(2, ((info["height"] * target_w) // info["width"]) & ~1)
    duration = max(0.0, range_end_s - range_start_s)
    total_samples = max(1, int(round(duration * sample_fps)))

    if progress_cb:
        progress_cb(1.0, f"loading pose model {model_name}", None)
    model = YOLO(model_name)

    if progress_cb:
        progress_cb(5.0, f"sampling near-player pose {total_samples:,} frames", None)
    frames = _iter_sampled_frames(
        video_path,
        target_w,
        target_h,
        sample_fps,
        range_start_s,
        duration,
        total_samples,
        progress_cb,
    )

    pose_frames: list[dict] = []
    started = time.monotonic()
    for sample_idx, (time_s, frame) in enumerate(frames):
        result = model.predict(frame, conf=float(conf_threshold), imgsz=int(image_size), verbose=False)[0]
        detections = _detections_from_result(result, target_w, target_h)
        pose_frames.append({
            "time_s": float(time_s),
            "detections": detections,
            "near_player": _near_player_detection(detections),
        })
        if progress_cb and (sample_idx == 0 or (sample_idx + 1) % 5 == 0):
            elapsed = time.monotonic() - started
            done = sample_idx + 1
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = max(0, total_samples - done) / rate if rate > 0 else None
            progress_cb(min(80.0, 5.0 + (done / total_samples) * 75.0), f"pose {done:,}/{total_samples:,}", eta)

    knobs = {**default_rally_knobs(), **(rally_knob_overrides or {})}
    if progress_cb:
        progress_cb(82.0, "detecting audio impact peaks", None)
    # Keep all peaks for study/debugging; threshold is stored as a feature.
    audio = analyze_audio_impacts(video_path, {**knobs, "min_spectral_centroid_hz": 0.0})
    impacts = [
        imp for imp in audio["impacts"]
        if float(range_start_s) <= float(imp["time_s"]) <= float(range_end_s)
    ]
    threshold = float(knobs.get("min_spectral_centroid_hz", 0.0))
    for imp in impacts:
        centroid = float(imp.get("spectral_centroid_hz", 0.0))
        imp["centroid_pass"] = threshold <= 0 or centroid >= threshold
        imp["centroid_threshold_hz"] = threshold

    if progress_cb:
        progress_cb(92.0, "building near-player feature windows", None)
    windows = build_feature_windows(pose_frames, impacts, range_start_s, range_end_s)

    metadata = {
        "detector": HIT_STUDY_ALGORITHM,
        "detector_version": HIT_STUDY_ARTIFACT_VERSION,
        "model_name": model_name,
        "conf_threshold": float(conf_threshold),
        "image_size": int(image_size),
        "duration_s": float(video_duration_s),
        "range_start_s": float(range_start_s),
        "range_end_s": float(range_end_s),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "sample_fps": float(sample_fps),
        "sample_count": int(len(pose_frames)),
        "rally_knobs": knobs,
    }
    pose_summary = summarize_pose_frames(pose_frames)
    summary = {
        **pose_summary,
        "near_player_visible_frames": sum(1 for f in pose_frames if f.get("near_player")),
        "audio_impact_count": len(impacts),
        "feature_window_count": len(windows),
        "on_count": 0,
    }
    artifact = {
        "metadata": metadata,
        "summary": summary,
        "frames": pose_frames,
        "audio": {
            "summary": {
                "sample_rate": audio["sample_rate"],
                "noise_floor": audio["noise_floor"],
                "duration_s": audio["duration_s"],
                "impact_count": len(impacts),
            },
            "noise_floor": audio["noise_floor"],
            "sample_rate": audio["sample_rate"],
            "impacts": impacts,
            "knobs": knobs,
        },
        "feature_windows": windows,
    }

    timeline = [{
        "start_s": float(range_start_s),
        "end_s": float(range_end_s),
        "is_on": False,
        "source": HIT_STUDY_ALGORITHM,
        "decision_stage": "labeling_needed",
        "sample_count": len(windows),
        "samples": windows[:200],
    }]

    settings.analysis_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = settings.analysis_dir / f"{analysis_id}.hit-study.json.gz"
    with gzip.open(artifact_path, "wt", encoding="utf-8") as f:
        json.dump(artifact, f)

    result = {
        "metadata": metadata,
        "summary": summary,
        "knobs": {"sample_fps": sample_fps, **knobs},
        "config": {
            "sample_fps": sample_fps,
            "pose_model": model_name,
            "pose_conf": float(conf_threshold),
            "pose_imgsz": int(image_size),
            "range_start_s": float(range_start_s),
            "range_end_s": float(range_end_s),
            **knobs,
        },
    }
    if progress_cb:
        progress_cb(100.0, "hit study ready for labels", None)
    return artifact_path, timeline, result


def load_hit_study_artifact(artifact_path: Path) -> dict:
    with gzip.open(artifact_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _near_player_detection(detections: list[dict]) -> dict | None:
    if not detections:
        return None
    # Near player is usually lower/larger in a static whole-court camera.
    def score(det: dict) -> float:
        box = det.get("box") or {}
        y2 = float(box.get("y2", 0.0))
        area = max(0.0, float(box.get("x2", 0.0)) - float(box.get("x1", 0.0))) * max(0.0, y2 - float(box.get("y1", 0.0)))
        return y2 + area * 2.0
    return max(detections, key=score)


def build_feature_windows(
    pose_frames: list[dict],
    impacts: list[dict],
    range_start_s: float,
    range_end_s: float,
    *,
    window_s: float = 1.5,
    stride_s: float = 0.5,
) -> list[dict]:
    out = []
    t = float(range_start_s)
    while t + window_s <= float(range_end_s) + 1e-6:
        end = t + window_s
        frames = [f for f in pose_frames if t <= float(f["time_s"]) < end]
        peaks = [i for i in impacts if t <= float(i["time_s"]) < end]
        out.append(_window_features(t, end, frames, peaks))
        t += stride_s
    return out


def _window_features(start_s: float, end_s: float, frames: list[dict], impacts: list[dict]) -> dict:
    near = [f for f in frames if f.get("near_player")]
    centers = []
    upper_points = []
    for f in near:
        det = f["near_player"]
        box = det.get("box") or {}
        cx = (float(box.get("x1", 0.0)) + float(box.get("x2", 0.0))) / 2.0
        cy = (float(box.get("y1", 0.0)) + float(box.get("y2", 0.0))) / 2.0
        centers.append((float(f["time_s"]), cx, cy))
        pts = []
        for kp in det.get("keypoints", []) or []:
            if int(kp.get("index", -1)) in {5, 6, 7, 8, 9, 10} and float(kp.get("confidence", 0.0)) >= 0.2:
                pts.append((float(kp.get("x", 0.0)), float(kp.get("y", 0.0))))
        upper_points.append((float(f["time_s"]), pts))
    box_speeds = _speeds(centers)
    upper_motion = _upper_body_motion(upper_points)
    centroids = [float(i.get("spectral_centroid_hz", 0.0)) for i in impacts if float(i.get("spectral_centroid_hz", 0.0)) > 0]
    snrs = [float(i.get("snr", 0.0)) for i in impacts]
    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "near_player_visible_ratio": (len(near) / len(frames)) if frames else 0.0,
        "near_player_box_speed_max": max(box_speeds) if box_speeds else 0.0,
        "near_player_box_speed_mean": float(np.mean(box_speeds)) if box_speeds else 0.0,
        "upper_body_motion_max": max(upper_motion) if upper_motion else 0.0,
        "upper_body_motion_mean": float(np.mean(upper_motion)) if upper_motion else 0.0,
        "audio_peak_count": len(impacts),
        "audio_snr_max": max(snrs) if snrs else 0.0,
        "audio_snr_mean": float(np.mean(snrs)) if snrs else 0.0,
        "audio_centroid_max": max(centroids) if centroids else 0.0,
        "audio_centroid_median": float(np.median(centroids)) if centroids else 0.0,
    }


def _speeds(samples: list[tuple[float, float, float]]) -> list[float]:
    speeds = []
    for a, b in zip(samples, samples[1:]):
        dt = max(1e-3, b[0] - a[0])
        speeds.append((((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5) / dt)
    return speeds


def _upper_body_motion(samples: list[tuple[float, list[tuple[float, float]]]]) -> list[float]:
    vals = []
    for a, b in zip(samples, samples[1:]):
        if not a[1] or not b[1]:
            continue
        dt = max(1e-3, b[0] - a[0])
        # Use average nearest-neighbor displacement between visible upper-body points.
        dists = []
        for ax, ay in a[1]:
            dists.append(min(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 for bx, by in b[1]))
        if dists:
            vals.append(float(np.mean(dists)) / dt)
    return vals


def ball_motion_diagnostic(
    video_path: Path,
    artifact: dict,
    time_s: float,
    *,
    window_s: float = 0.8,
    roi_expand_x: float = 3.0,
    roi_expand_up: float = 1.0,
    roi_expand_down: float = 0.8,
    roi_mode: str = "near_player",
    exclude_player: bool = True,
    diff_threshold: float = 12.0,
    ball_detector: str = "motion",
    tracknet_width: int = 640,
    tracknet_height: int = 360,
) -> dict:
    metadata = artifact.get("metadata") or {}
    target_w = int(metadata.get("target_width") or POSE_TARGET_WIDTH)
    target_h = int(metadata.get("target_height") or 360)
    if roi_mode == "full_frame":
        roi = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    else:
        roi = _interaction_roi_with_expansion(
            artifact.get("frames") or [],
            float(time_s),
            expand_x=float(roi_expand_x),
            expand_up=float(roi_expand_up),
            expand_down=float(roi_expand_down),
        )
    if roi is None:
        roi = {"x1": 0.0, "y1": 0.35, "x2": 1.0, "y2": 1.0}
    player_box = _nearest_near_player_box(artifact.get("frames") or [], float(time_s))

    if ball_detector == "tracknet":
        tracknet = run_tracknet_window(
            video_path,
            time_s,
            window_s=window_s,
            roi=roi,
            player_box=player_box,
            exclude_player=exclude_player,
            input_width=int(tracknet_width),
            input_height=int(tracknet_height),
        )
        return {
            "time_s": float(time_s),
            "window_s": float(window_s),
            "fps": tracknet["fps"],
            "roi": roi,
            "roi_mode": roi_mode,
            "player_box": player_box,
            "exclude_player": bool(exclude_player),
            "diff_threshold": float(diff_threshold),
            "ball_detector": "tracknet",
            "candidate_count": tracknet["candidate_count"],
            "before_count": tracknet["before_count"],
            "after_count": tracknet["after_count"],
            "candidates": tracknet["candidates"],
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for ball diagnostic: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    start_s = max(0.0, float(time_s) - window_s)
    end_s = float(time_s) + window_s
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    prev_gray = None
    frame_idx = 0
    candidates: list[dict] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = start_s + frame_idx / fps
        if t > end_s:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if prev_gray is None:
            prev_gray = gray
            continue
        x1 = int(round(roi["x1"] * target_w))
        y1 = int(round(roi["y1"] * target_h))
        x2 = int(round(roi["x2"] * target_w))
        y2 = int(round(roi["y2"] * target_h))
        diff = cv2.absdiff(prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2])
        _, mask = cv2.threshold(diff, float(diff_threshold), 255, cv2.THRESH_BINARY)
        if exclude_player and player_box is not None:
            _clear_box_from_mask(mask, player_box, x1, y1, target_w, target_h)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 1.5 or area > 70.0:
                continue
            px, py, pw, ph = cv2.boundingRect(contour)
            aspect = max(pw, ph) / max(1, min(pw, ph))
            if aspect > 5.0:
                continue
            cx = (x1 + px + pw / 2.0) / target_w
            cy = (y1 + py + ph / 2.0) / target_h
            score = float(area * np.mean(diff[py:py + ph, px:px + pw]))
            frame_candidates.append({
                "time_s": float(t),
                "dt_s": float(t - time_s),
                "x": float(cx),
                "y": float(cy),
                "area": area,
                "score": score,
            })
        frame_candidates.sort(key=lambda c: c["score"], reverse=True)
        candidates.extend(frame_candidates[:8])
        prev_gray = gray
    cap.release()

    before = [c for c in candidates if c["time_s"] < time_s]
    after = [c for c in candidates if c["time_s"] >= time_s]
    return {
        "time_s": float(time_s),
        "window_s": float(window_s),
        "fps": fps,
        "roi": roi,
        "roi_mode": roi_mode,
        "player_box": player_box,
        "exclude_player": bool(exclude_player),
        "diff_threshold": float(diff_threshold),
        "ball_detector": "motion",
        "candidate_count": len(candidates),
        "before_count": len(before),
        "after_count": len(after),
        "candidates": sorted(candidates, key=lambda c: (c["time_s"], -c["score"]))[:300],
    }


def _interaction_roi(frames: list[dict], time_s: float) -> dict | None:
    return _interaction_roi_with_expansion(frames, time_s, expand_x=3.0, expand_up=1.0, expand_down=0.8)


def _interaction_roi_with_expansion(
    frames: list[dict],
    time_s: float,
    *,
    expand_x: float,
    expand_up: float,
    expand_down: float,
) -> dict | None:
    near_frames = [f for f in frames if f.get("near_player")]
    if not near_frames:
        return None
    frame = min(near_frames, key=lambda f: abs(float(f.get("time_s", 0.0)) - time_s))
    box = (frame.get("near_player") or {}).get("box") or {}
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 1.0))
    y2 = float(box.get("y2", 1.0))
    w = max(0.04, x2 - x1)
    h = max(0.08, y2 - y1)
    return {
        "x1": max(0.0, x1 - float(expand_x) * w),
        "y1": max(0.0, y1 - float(expand_up) * h),
        "x2": min(1.0, x2 + float(expand_x) * w),
        "y2": min(1.0, y2 + float(expand_down) * h),
    }


def _nearest_near_player_box(frames: list[dict], time_s: float) -> dict | None:
    near_frames = [f for f in frames if f.get("near_player")]
    if not near_frames:
        return None
    frame = min(near_frames, key=lambda f: abs(float(f.get("time_s", 0.0)) - time_s))
    box = (frame.get("near_player") or {}).get("box") or None
    if not box:
        return None
    return {
        "x1": float(box.get("x1", 0.0)),
        "y1": float(box.get("y1", 0.0)),
        "x2": float(box.get("x2", 1.0)),
        "y2": float(box.get("y2", 1.0)),
    }


def _clear_box_from_mask(
    mask: np.ndarray,
    box: dict,
    roi_x1: int,
    roi_y1: int,
    target_w: int,
    target_h: int,
    *,
    pad_x: float = 0.015,
    pad_y: float = 0.02,
) -> None:
    px1 = int(round((max(0.0, float(box["x1"]) - pad_x) * target_w) - roi_x1))
    py1 = int(round((max(0.0, float(box["y1"]) - pad_y) * target_h) - roi_y1))
    px2 = int(round((min(1.0, float(box["x2"]) + pad_x) * target_w) - roi_x1))
    py2 = int(round((min(1.0, float(box["y2"]) + pad_y) * target_h) - roi_y1))
    px1 = max(0, min(mask.shape[1], px1))
    px2 = max(0, min(mask.shape[1], px2))
    py1 = max(0, min(mask.shape[0], py1))
    py2 = max(0, min(mask.shape[0], py2))
    if px2 > px1 and py2 > py1:
        mask[py1:py2, px1:px2] = 0
