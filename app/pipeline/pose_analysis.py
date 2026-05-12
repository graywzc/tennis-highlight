import gzip
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str, float | None], None]

POSE_ALGORITHM = "pose_skeleton_yolo"
POSE_ARTIFACT_VERSION = 1
POSE_TARGET_WIDTH = 640
DEFAULT_POSE_MODEL = "yolo11n-pose.pt"
DEFAULT_POSE_CONF = 0.25


def analyze_pose_to_artifact(
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
            "Ultralytics is required for pose_skeleton_yolo. "
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
        progress_cb(5.0, f"running pose on {total_samples:,} samples", None)

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
        })
        if progress_cb and (sample_idx == 0 or (sample_idx + 1) % 5 == 0):
            elapsed = time.monotonic() - started
            done = sample_idx + 1
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total_samples - done)
            eta = remaining / rate if rate > 0 else None
            pct = min(99.0, 5.0 + (done / total_samples) * 94.0)
            progress_cb(
                pct,
                f"running pose {done:,}/{total_samples:,} samples — ETA {_fmt_eta(eta)}",
                eta,
            )

    metadata = {
        "detector": POSE_ALGORITHM,
        "detector_version": POSE_ARTIFACT_VERSION,
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
    }
    summary = summarize_pose_frames(pose_frames)

    from app.pipeline.rally_detection import default_rally_knobs, detect_rallies

    rally_knobs = {**default_rally_knobs(), **(rally_knob_overrides or {})}
    if progress_cb:
        progress_cb(99.0, "detecting rallies from audio + pose", None)
    audio_result, timeline = detect_rallies(
        video_path, video_duration_s, pose_frames, rally_knobs, progress_cb,
    )
    metadata["rally_knobs"] = rally_knobs
    metadata["audio"] = audio_result.get("summary") or {}

    summary = {
        **summary,
        "rally_count": audio_result["summary"]["rally_count"],
        "impact_count": audio_result["summary"]["impact_count"],
        "validated_impact_count": audio_result["summary"]["validated_impact_count"],
        "on_count": audio_result["summary"]["rally_count"],
    }

    artifact = {
        "metadata": metadata,
        "summary": summary,
        "frames": pose_frames,
        "rally": {
            "summary": audio_result["summary"],
            "noise_floor": audio_result["noise_floor"],
            "sample_rate": audio_result["sample_rate"],
            "impacts": audio_result["validated_impacts"],
            "segments": timeline,
            "knobs": rally_knobs,
        },
    }

    settings.analysis_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = settings.analysis_dir / f"{analysis_id}.pose.json.gz"
    with gzip.open(artifact_path, "wt", encoding="utf-8") as f:
        json.dump(artifact, f)

    result = {
        "metadata": metadata,
        "summary": summary,
        "knobs": {"sample_fps": sample_fps, **rally_knobs},
        "config": {
            "sample_fps": sample_fps,
            "pose_model": model_name,
            "pose_conf": float(conf_threshold),
            "pose_imgsz": int(image_size),
            "range_start_s": float(range_start_s),
            "range_end_s": float(range_end_s),
            **rally_knobs,
        },
    }
    return artifact_path, timeline, result


def load_pose_artifact(artifact_path: Path) -> dict:
    with gzip.open(artifact_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def summarize_pose_frames(frames: list[dict]) -> dict:
    sample_count = len(frames)
    frames_with_poses = sum(1 for f in frames if f.get("detections"))
    det_counts = [len(f.get("detections", [])) for f in frames]
    keypoint_conf = []
    box_conf = []
    for frame in frames:
        for det in frame.get("detections", []):
            box_conf.append(float(det.get("confidence", 0.0)))
            for kp in det.get("keypoints", []):
                keypoint_conf.append(float(kp.get("confidence", 0.0)))
    return {
        "sample_count": sample_count,
        "frames_with_poses": frames_with_poses,
        "pose_frame_percent": (frames_with_poses / sample_count * 100.0) if sample_count else 0.0,
        "avg_detections_per_frame": float(np.mean(det_counts)) if det_counts else 0.0,
        "max_box_confidence": max(box_conf) if box_conf else None,
        "avg_keypoint_confidence": float(np.mean(keypoint_conf)) if keypoint_conf else None,
    }


def _iter_sampled_frames(
    video_path: Path,
    target_w: int,
    target_h: int,
    sample_fps: float,
    range_start_s: float,
    duration_s: float,
    total_samples: int,
    progress_cb: ProgressCallback | None,
):
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-hwaccel", "videotoolbox",
        "-ss", f"{range_start_s:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_s:.3f}",
        "-an",
        "-vf", f"fps={sample_fps},scale={target_w}:{target_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]
    frame_bytes = target_w * target_h * 3
    sample_idx = 0
    with tempfile.TemporaryFile(mode="w+b") as stderr_tmp:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_tmp,
            bufsize=frame_bytes * 4,
        )
        assert proc.stdout is not None
        try:
            while True:
                buf = _read_exact(proc.stdout, frame_bytes)
                if buf is None:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(target_h, target_w, 3)
                yield range_start_s + (sample_idx / sample_fps), frame
                sample_idx += 1
                if progress_cb and sample_idx % 20 == 0:
                    progress_cb(
                        min(30.0, (sample_idx / total_samples) * 30.0),
                        f"sampled {sample_idx:,}/{total_samples:,} frames",
                        None,
                    )
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait(timeout=30)
        if proc.returncode != 0:
            stderr_tmp.seek(0)
            err = stderr_tmp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg pose sampling failed: {err.strip()[-500:]}")


def _detections_from_result(result, width: int, height: int) -> list[dict]:
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or keypoints is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),))
    kxy = keypoints.xy.cpu().numpy() if keypoints.xy is not None else np.zeros((len(xyxy), 0, 2))
    kconf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.zeros((len(xyxy), kxy.shape[1] if kxy.ndim == 3 else 0))
    detections = []
    for i, box in enumerate(xyxy):
        pts = []
        for j in range(kxy.shape[1]):
            pts.append({
                "index": int(j),
                "x": float(kxy[i, j, 0] / width),
                "y": float(kxy[i, j, 1] / height),
                "confidence": float(kconf[i, j]) if kconf.size else 0.0,
            })
        detections.append({
            "person_index": int(i),
            "confidence": float(conf[i]) if i < len(conf) else 0.0,
            "box": {
                "x1": float(box[0] / width),
                "y1": float(box[1] / height),
                "x2": float(box[2] / width),
                "y2": float(box[3] / height),
            },
            "keypoints": pts,
        })
    return detections


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
    return {"width": int(s["width"]), "height": int(s["height"])}


def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)
        buf.extend(chunk)
    return bytes(buf)


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = seconds / 60
    if mins < 60:
        return f"{mins:.1f}m"
    return f"{mins / 60:.1f}h"
