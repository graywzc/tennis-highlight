"""Modular scan endpoints for independent YOLO Pose and Audio analysis.

Each module can be scanned, saved, and loaded independently. TrackNet (ball-scan)
already exists in hit_study.py. This keeps the three analysis stages decoupled:

  1. YOLO Pose  → pose-scan  (produces pose frames + near-player data)
  2. Audio      → audio-scan (produces impact candidates)
  3. TrackNet   → ball-scan  (existing, uses pose ROI)
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.database import get_analysis_run
from app.pipeline.near_player_hit_study import load_hit_study_artifact

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job stores (same pattern as BALL_SCAN_JOBS)
POSE_SCAN_JOBS: dict[str, dict] = {}
AUDIO_SCAN_JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PoseScanRequest(BaseModel):
    range_start_s: float | None = None
    range_end_s: float | None = None
    sample_fps: float = 4.0
    model_name: str = "yolo11n-pose.pt"
    pose_conf: float = 0.25
    pose_imgsz: int = 640


class AudioScanRequest(BaseModel):
    range_start_s: float | None = None
    range_end_s: float | None = None
    audio_sample_rate: int = 22050
    bandpass_low_hz: float = 1000.0
    bandpass_high_hz: float = 8000.0
    peak_height_mad_k: float = 6.0
    peak_prominence_mult: float = 2.0
    min_impact_separation_s: float = 0.15
    min_spectral_centroid_hz: float = 2500.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _artifacts_dir() -> Path:
    d = settings.analysis_dir.parent / "modular_scans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pose_artifact_path(analysis_id: str, filename: str | None = None) -> Path:
    if filename:
        return _artifacts_dir() / filename
    return _artifacts_dir() / f"{analysis_id}.pose-scan.json.gz"


def _audio_artifact_path(analysis_id: str, filename: str | None = None) -> Path:
    if filename:
        return _artifacts_dir() / filename
    return _artifacts_dir() / f"{analysis_id}.audio-scan.json"


async def _get_hit_study(analysis_id: str):
    """Common validation: analysis must exist."""
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # We no longer strictly enforce HIT_STUDY_ALGORITHM here so modular scans
    # can be added to any analysis (e.g. median_frame).
    return analysis


# =========================================================================
# POSE SCAN
# =========================================================================

@router.post("/hit-study/{analysis_id}/pose-scan")
async def start_pose_scan(analysis_id: str, payload: PoseScanRequest) -> dict:
    analysis = await _get_hit_study(analysis_id)
    video_path = Path(analysis["filepath"])
    if not video_path.exists():
        raise HTTPException(410, "video file missing")

    duration_s = float(analysis["duration_s"])
    start_s = max(0.0, float(payload.range_start_s or 0.0))
    end_s = min(duration_s, float(payload.range_end_s or duration_s))
    if end_s <= start_s:
        raise HTTPException(422, "range_end_s must be greater than range_start_s")

    job_id = uuid4().hex
    POSE_SCAN_JOBS[job_id] = {
        "job_id": job_id,
        "analysis_id": analysis_id,
        "status": "pending",
        "progress_percent": 0.0,
        "progress_message": "queued",
        "progress_eta_s": None,
        "result": None,
        "error": None,
    }
    thread = threading.Thread(
        target=_run_pose_scan,
        args=(job_id, analysis_id, video_path, duration_s, start_s, end_s, payload),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "pending"}


@router.get("/hit-study/pose-scan/{job_id}")
async def pose_scan_status(job_id: str) -> dict:
    job = POSE_SCAN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "pose scan job not found")
    return job


@router.post("/hit-study/{analysis_id}/pose-scan/save")
async def save_pose_scan(analysis_id: str, filename: str | None = None) -> dict:
    """Save the most recent pose scan result for this analysis."""
    analysis = await _get_hit_study(analysis_id)
    # Find the latest completed job for this analysis
    job = _latest_job(POSE_SCAN_JOBS, analysis_id)
    result = job.get("result") if job else None

    if not result and dict(analysis).get("artifact_path"):
        try:
            artifact = load_hit_study_artifact(Path(analysis["artifact_path"]))
            meta = artifact.get("metadata", {})
            config = artifact.get("config", {})
            result = {
                "range_start_s": config.get("range_start_s", 0.0),
                "range_end_s": config.get("range_end_s", meta.get("duration_s", 0.0)),
                "sample_fps": config.get("sample_fps", meta.get("sample_fps", 4.0)),
                "model_name": config.get("pose_model", "yolo11n-pose.pt"),
                "pose_conf": config.get("pose_conf", 0.25),
                "pose_imgsz": config.get("pose_imgsz", 640),
                "target_width": meta.get("target_width", 640),
                "target_height": meta.get("target_height", 360),
                "frames": artifact.get("frames", []),
                "summary": artifact.get("summary", {}),
            }
        except Exception:
            pass

    if not result:
        raise HTTPException(400, "no completed pose scan to save")

    path = _pose_artifact_path(analysis_id, filename)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({
            "analysis_id": analysis_id,
            "saved_at": time.time(),
            "result": result,
        }, f)
    
    from app.database import update_analysis_modular_paths
    await update_analysis_modular_paths(analysis_id, pose_path=str(path))
    
    return {"analysis_id": analysis_id, "saved_path": str(path), "frame_count": len(result.get("frames", []))}


@router.get("/hit-study/{analysis_id}/pose-scan/load")
async def load_pose_scan(analysis_id: str) -> dict:
    analysis = await _get_hit_study(analysis_id)
    path = _pose_artifact_path(analysis_id)
    
    if analysis.get("active_pose_scan_path"):
        p = Path(analysis["active_pose_scan_path"])
        if p.exists():
            path = p

    if not path.exists():
        raise HTTPException(404, f"no saved pose scan at {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
        # Always return a standardized wrapper for modular loads
        result = data.get("result") or data
        return {"result": result, "source_file": path.name}

@router.get("/hit-study/{analysis_id}/pose-scan/download")
async def download_pose_scan(analysis_id: str):
    analysis = await _get_hit_study(analysis_id)
    path = _pose_artifact_path(analysis_id)
    if analysis.get("active_pose_scan_path"):
        p = Path(analysis["active_pose_scan_path"])
        if p.exists():
            path = p
            
    if not path.exists():
        raise HTTPException(404, "no saved pose scan to download")
    return FileResponse(path, media_type="application/gzip", filename=path.name)

@router.post("/hit-study/{analysis_id}/pose-scan/upload")
async def upload_pose_scan(analysis_id: str, file: UploadFile = File(...), filename: str | None = None) -> dict:
    await _get_hit_study(analysis_id)
    target_name = filename or file.filename or f"{analysis_id}.pose-scan.json.gz"
    if not target_name.endswith(".gz"):
        target_name += ".gz"
        
    path = _artifacts_dir() / target_name
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
        
    from app.database import update_analysis_modular_paths
    await update_analysis_modular_paths(analysis_id, pose_path=str(path))
    
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _run_pose_scan(
    job_id: str,
    analysis_id: str,
    video_path: Path,
    video_duration_s: float,
    start_s: float,
    end_s: float,
    payload: PoseScanRequest,
) -> None:
    from app.pipeline.pose_analysis import (
        _detections_from_result,
        _iter_sampled_frames,
        _probe_video,
        summarize_pose_frames,
    )
    from app.pipeline.near_player_hit_study import _near_player_detection

    job = POSE_SCAN_JOBS[job_id]
    job["status"] = "running"
    try:
        from ultralytics import YOLO
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"ultralytics not installed: {e}"
        return

    try:
        info = _probe_video(video_path)
        target_w = int(payload.pose_imgsz)
        target_h = max(2, ((info["height"] * target_w) // info["width"]) & ~1)
        duration = max(0.0, end_s - start_s)
        sample_fps = float(payload.sample_fps)
        total_samples = max(1, int(round(duration * sample_fps)))

        job["progress_percent"] = 2.0
        job["progress_message"] = f"loading model {payload.model_name}"
        model = YOLO(payload.model_name)

        job["progress_percent"] = 5.0
        job["progress_message"] = f"sampling {total_samples:,} frames"
        frames_iter = _iter_sampled_frames(
            video_path, target_w, target_h, sample_fps,
            start_s, duration, total_samples, None,
        )

        pose_frames: list[dict] = []
        started = time.monotonic()
        for sample_idx, (time_s, frame) in enumerate(frames_iter):
            result = model.predict(frame, conf=float(payload.pose_conf), imgsz=int(payload.pose_imgsz), verbose=False)[0]
            detections = _detections_from_result(result, target_w, target_h)
            pose_frames.append({
                "time_s": float(time_s),
                "detections": detections,
                "near_player": _near_player_detection(detections),
            })
            if sample_idx == 0 or (sample_idx + 1) % 5 == 0:
                elapsed = time.monotonic() - started
                done = sample_idx + 1
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_samples - done)
                eta = remaining / rate if rate > 0 else None
                pct = min(99.0, 5.0 + (done / total_samples) * 94.0)
                job["progress_percent"] = pct
                job["progress_message"] = f"pose {done:,}/{total_samples:,}"
                job["progress_eta_s"] = eta

        summary = summarize_pose_frames(pose_frames)
        summary["near_player_visible_frames"] = sum(1 for f in pose_frames if f.get("near_player"))

        job["status"] = "done"
        job["progress_percent"] = 100.0
        job["progress_message"] = f"done: {len(pose_frames)} frames, {summary['frames_with_poses']} with poses"
        job["progress_eta_s"] = None
        job["result"] = {
            "range_start_s": float(start_s),
            "range_end_s": float(end_s),
            "sample_fps": sample_fps,
            "model_name": payload.model_name,
            "pose_conf": float(payload.pose_conf),
            "pose_imgsz": int(payload.pose_imgsz),
            "target_width": target_w,
            "target_height": target_h,
            "frames": pose_frames,
            "summary": summary,
        }
    except Exception as e:
        logger.exception("pose scan failed")
        job["status"] = "error"
        job["error"] = str(e)
        job["progress_message"] = f"error: {e}"


# =========================================================================
# AUDIO SCAN
# =========================================================================

@router.post("/hit-study/{analysis_id}/audio-scan")
async def start_audio_scan(analysis_id: str, payload: AudioScanRequest) -> dict:
    """Run audio impact detection on a range. Returns synchronously (fast)."""
    analysis = await _get_hit_study(analysis_id)
    video_path = Path(analysis["filepath"])
    if not video_path.exists():
        raise HTTPException(410, "video file missing")

    duration_s = float(analysis["duration_s"])
    start_s = max(0.0, float(payload.range_start_s or 0.0))
    end_s = min(duration_s, float(payload.range_end_s or duration_s))
    if end_s <= start_s:
        raise HTTPException(422, "range_end_s must be greater than range_start_s")

    # Audio analysis is CPU-bound but fast enough for typical ranges (<30s).
    # For very long ranges, run in background thread like pose.
    job_id = uuid4().hex
    AUDIO_SCAN_JOBS[job_id] = {
        "job_id": job_id,
        "analysis_id": analysis_id,
        "status": "pending",
        "progress_percent": 0.0,
        "progress_message": "queued",
        "result": None,
        "error": None,
    }
    thread = threading.Thread(
        target=_run_audio_scan,
        args=(job_id, analysis_id, video_path, start_s, end_s, payload),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "pending"}


@router.get("/hit-study/audio-scan/{job_id}")
async def audio_scan_status(job_id: str) -> dict:
    job = AUDIO_SCAN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "audio scan job not found")
    return job


@router.post("/hit-study/{analysis_id}/audio-scan/save")
async def save_audio_scan(analysis_id: str, filename: str | None = None) -> dict:
    analysis = await _get_hit_study(analysis_id)
    job = _latest_job(AUDIO_SCAN_JOBS, analysis_id)
    result = job.get("result") if job else None

    if not result and dict(analysis).get("artifact_path"):
        try:
            artifact = load_hit_study_artifact(Path(analysis["artifact_path"]))
            meta = artifact.get("metadata", {})
            config = artifact.get("config", {})
            audio = artifact.get("audio", {})
            result = {
                "range_start_s": config.get("range_start_s", 0.0),
                "range_end_s": config.get("range_end_s", meta.get("duration_s", 0.0)),
                "knobs": audio.get("knobs", config),
                "sample_rate": audio.get("sample_rate", 22050),
                "noise_floor": audio.get("noise_floor", 0.0),
                "impacts": audio.get("impacts", []),
                "impact_count": len(audio.get("impacts", [])),
            }
        except Exception:
            pass

    if not result:
        raise HTTPException(400, "no completed audio scan to save")

    path = _audio_artifact_path(analysis_id, filename)
    path.write_text(json.dumps({
        "analysis_id": analysis_id,
        "saved_at": time.time(),
        "result": result,
    }, indent=2), encoding="utf-8")
    
    from app.database import update_analysis_modular_paths
    await update_analysis_modular_paths(analysis_id, audio_path=str(path))
    
    return {"analysis_id": analysis_id, "saved_path": str(path), "impact_count": len(result.get("impacts", []))}


@router.get("/hit-study/{analysis_id}/audio-scan/load")
async def load_audio_scan(analysis_id: str) -> dict:
    analysis = await _get_hit_study(analysis_id)
    path = _audio_artifact_path(analysis_id)
    
    if analysis.get("active_audio_scan_path"):
        p = Path(analysis["active_audio_scan_path"])
        if p.exists():
            path = p
            
    if not path.exists():
        raise HTTPException(404, f"no saved audio scan at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("result") or data
    return {"result": result, "source_file": path.name}

@router.get("/hit-study/{analysis_id}/audio-scan/download")
async def download_audio_scan(analysis_id: str):
    analysis = await _get_hit_study(analysis_id)
    path = _audio_artifact_path(analysis_id)
    if analysis.get("active_audio_scan_path"):
        p = Path(analysis["active_audio_scan_path"])
        if p.exists():
            path = p
            
    if not path.exists():
        raise HTTPException(404, "no saved audio scan to download")
    return FileResponse(path, media_type="application/json", filename=path.name)

@router.post("/hit-study/{analysis_id}/audio-scan/upload")
async def upload_audio_scan(analysis_id: str, file: UploadFile = File(...), filename: str | None = None) -> dict:
    await _get_hit_study(analysis_id)
    target_name = filename or file.filename or f"{analysis_id}.audio-scan.json"
    if not target_name.endswith(".json"):
        target_name += ".json"
        
    path = _artifacts_dir() / target_name
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
        
    from app.database import update_analysis_modular_paths
    await update_analysis_modular_paths(analysis_id, audio_path=str(path))
    
    return json.loads(path.read_text(encoding="utf-8"))


def _run_audio_scan(
    job_id: str,
    analysis_id: str,
    video_path: Path,
    start_s: float,
    end_s: float,
    payload: AudioScanRequest,
) -> None:
    from app.pipeline.audio_analysis import analyze_audio_impacts_range

    job = AUDIO_SCAN_JOBS[job_id]
    job["status"] = "running"
    job["progress_message"] = "running audio analysis"
    job["progress_percent"] = 10.0

    try:
        knobs = {
            "audio_sample_rate": int(payload.audio_sample_rate),
            "bandpass_low_hz": float(payload.bandpass_low_hz),
            "bandpass_high_hz": float(payload.bandpass_high_hz),
            "peak_height_mad_k": float(payload.peak_height_mad_k),
            "peak_prominence_mult": float(payload.peak_prominence_mult),
            "min_impact_separation_s": float(payload.min_impact_separation_s),
            "min_spectral_centroid_hz": float(payload.min_spectral_centroid_hz),
        }

        result = analyze_audio_impacts_range(video_path, start_s, end_s, knobs)

        job["status"] = "done"
        job["progress_percent"] = 100.0
        job["progress_message"] = f"done: {len(result['impacts'])} impacts detected"
        job["result"] = {
            "range_start_s": float(start_s),
            "range_end_s": float(end_s),
            "knobs": knobs,
            "sample_rate": result["sample_rate"],
            "noise_floor": result["noise_floor"],
            "impacts": result["impacts"],
            "impact_count": len(result["impacts"]),
        }
    except Exception as e:
        logger.exception("audio scan failed")
        job["status"] = "error"
        job["error"] = str(e)
        job["progress_message"] = f"error: {e}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _latest_job(store: dict[str, dict], analysis_id: str) -> dict | None:
    """Find the most recent completed job for an analysis."""
    candidates = [
        j for j in store.values()
        if j.get("analysis_id") == analysis_id and j.get("status") == "done"
    ]
    return candidates[-1] if candidates else None
