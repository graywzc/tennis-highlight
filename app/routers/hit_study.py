import gzip
import json
import threading
import time
from uuid import uuid4
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.database import delete_strike_label, get_analysis_run, list_strike_labels, upsert_strike_label
from app.pipeline.near_player_hit_study import HIT_STUDY_ALGORITHM, ball_motion_diagnostic, load_hit_study_artifact

router = APIRouter()
BALL_SCAN_JOBS: dict[str, dict] = {}


class BallScanRequest(BaseModel):
    range_start_s: float | None = None
    range_end_s: float | None = None
    scan_range: str = "analysis_range"
    scan_target: str = "audio_candidates"
    target_times_s: list[float] = []
    scan_mode: str = "range"
    mark_before_s: float = 1.0
    mark_after_s: float = 1.0
    scan_fps: str = "2"
    ball_detector: str = "motion"
    roi_mode: str = "near_player"
    roi_expand_x: float = 3.0
    roi_expand_up: float = 1.0
    roi_expand_down: float = 0.8
    exclude_player: bool = True
    diff_threshold: float = 12.0
    tracknet_width: int = 640
    tracknet_height: int = 360
    tracknet_stack_frames: int = 3
    step_s: float | None = None


class BallScanSaveRequest(BaseModel):
    job_id: str | None = None
    result: dict | None = None


def _label_dir() -> Path:
    path = settings.analysis_dir.parent / "hit_labels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _label_path(analysis_id: str, filename: str | None = None) -> Path:
    if filename:
        safe_name = Path(filename).name
        if not safe_name.endswith(".json"):
            safe_name += ".json"
        return _label_dir() / safe_name
    return _label_dir() / f"{analysis_id}.near-player-hit-labels.json"


def _row_to_label(row) -> dict:
    return {
        "id": row["id"],
        "analysis_id": row["analysis_id"],
        "time_s": float(row["time_s"]),
        "source": row["source"],
        "is_strike": bool(row["is_strike"]),
        "algorithm_validated": None if row["algorithm_validated"] is None else bool(row["algorithm_validated"]),
        "comment": row["comment"],
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
    }


@router.get("/hit-study/{analysis_id}/data")
async def hit_study_data(analysis_id: str) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # Allowed for any algorithm now
    return _load_hit_study_context(analysis)



@router.post("/hit-study/{analysis_id}/evaluate")
async def evaluate_hit_study(analysis_id: str) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # Allowed for any algorithm now
    artifact = load_hit_study_artifact(Path(analysis["artifact_path"]))
    labels = [
        r for r in await list_strike_labels(analysis_id)
        if r["source"] == "near_player_hit" and bool(r["is_strike"])
    ]
    hit_times = [float(r["time_s"]) for r in labels]
    windows = artifact.get("feature_windows") or []
    if not hit_times:
        return {
            "analysis_id": analysis_id,
            "hit_count": 0,
            "window_count": len(windows),
            "message": "Mark near-player hit times before evaluating labels.",
            "features": [],
        }
    positives = [
        w for w in windows
        if any(float(w["start_s"]) - 0.25 <= t <= float(w["end_s"]) + 0.25 for t in hit_times)
    ]
    negatives = [
        w for w in windows
        if all(abs(((float(w["start_s"]) + float(w["end_s"])) / 2.0) - t) > 2.0 for t in hit_times)
    ]
    feature_keys = [
        "upper_body_motion_max",
        "upper_body_motion_mean",
        "audio_peak_count",
        "audio_snr_max",
        "audio_snr_mean",
        "audio_centroid_max",
        "audio_centroid_median",
    ]
    rows = []
    for key in feature_keys:
        pos = [float(w.get(key, 0.0)) for w in positives]
        neg = [float(w.get(key, 0.0)) for w in negatives]
        pos_mean = float(np.mean(pos)) if pos else 0.0
        neg_mean = float(np.mean(neg)) if neg else 0.0
        neg_std = float(np.std(neg)) if len(neg) > 1 else 0.0
        separation = (pos_mean - neg_mean) / (neg_std + 1e-6)
        rows.append({
            "feature": key,
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "separation": float(separation),
        })
    rows.sort(key=lambda r: abs(r["separation"]), reverse=True)
    return {
        "analysis_id": analysis_id,
        "hit_count": len(hit_times),
        "positive_window_count": len(positives),
        "negative_window_count": len(negatives),
        "window_count": len(windows),
        "features": rows,
    }


@router.get("/hit-study/{analysis_id}/ball-diagnostic")
async def hit_study_ball_diagnostic(
    analysis_id: str,
    time_s: float,
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
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # Allowed for any algorithm now
    video_path = Path(analysis["filepath"])
    if not video_path.exists():
        raise HTTPException(410, "video file missing")
    try:
        diagnostic = ball_motion_diagnostic(
            video_path,
            _load_hit_study_context(analysis),
            float(time_s),
            roi_expand_x=float(roi_expand_x),
            roi_expand_up=float(roi_expand_up),
            roi_expand_down=float(roi_expand_down),
            roi_mode=roi_mode,
            exclude_player=bool(exclude_player),
            diff_threshold=float(diff_threshold),
            ball_detector=ball_detector,
            tracknet_width=int(tracknet_width),
            tracknet_height=int(tracknet_height),
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
    return {"analysis_id": analysis_id, **diagnostic}


@router.post("/hit-study/{analysis_id}/ball-scan")
async def start_ball_scan(
    analysis_id: str,
    payload: BallScanRequest,
) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # Allowed for any algorithm now
    video_path = Path(analysis["filepath"])
    if not video_path.exists():
        raise HTTPException(410, "video file missing")

    artifact = _load_hit_study_context(analysis)
    metadata = artifact.get("metadata") or {}
    analysis_start = float(metadata.get("range_start_s", 0.0))
    analysis_end = float(metadata.get("range_end_s", analysis["duration_s"]))
    start_s = max(analysis_start, float(payload.range_start_s or analysis_start))
    end_s = min(analysis_end, float(payload.range_end_s or analysis_end))
    target_times = [
        float(t) for t in (payload.target_times_s or [])
        if start_s <= float(t) <= end_s
    ]
    if not target_times and payload.scan_target == "marked_hits":
        labels = [
            r for r in await list_strike_labels(analysis_id)
            if r["source"] == "near_player_hit" and bool(r["is_strike"])
        ]
        target_times = [float(r["time_s"]) for r in labels if start_s <= float(r["time_s"]) <= end_s]
    ranges = _scan_ranges(payload, start_s, end_s, target_times)
    if not ranges:
        raise HTTPException(400, "scan range is empty")
    start_s = min(r[0] for r in ranges)
    end_s = max(r[1] for r in ranges)
    job_id = uuid4().hex
    BALL_SCAN_JOBS[job_id] = {
        "job_id": job_id,
        "analysis_id": analysis_id,
        "status": "pending",
        "progress_percent": 0.0,
        "progress_message": "queued",
        "range_start_s": start_s,
        "range_end_s": end_s,
        "cancel_requested": False,
        "result": None,
        "error": None,
    }
    thread = threading.Thread(
        target=_run_ball_scan_job,
        args=(job_id, video_path, artifact, ranges, target_times, payload),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "pending"}


@router.get("/hit-study/ball-scan/{job_id}")
async def ball_scan_status(job_id: str) -> dict:
    job = BALL_SCAN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "ball scan job not found")
    return job


@router.post("/hit-study/ball-scan/{job_id}/cancel")
async def cancel_ball_scan(job_id: str) -> dict:
    job = BALL_SCAN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "ball scan job not found")
    job["cancel_requested"] = True
    if job["status"] in {"pending", "running"}:
        job["progress_message"] = "cancel requested"
    return {"job_id": job_id, "status": job["status"], "cancel_requested": True}


@router.post("/hit-study/{analysis_id}/ball-scan/save")
async def save_ball_scan(analysis_id: str, payload: BallScanSaveRequest) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    result = payload.result
    if payload.job_id:
        job = BALL_SCAN_JOBS.get(payload.job_id)
        if job:
            result = job.get("result")
    if not result:
        raise HTTPException(400, "no scan result to save")
    path = _ball_scan_path(analysis_id)
    path.write_text(json.dumps({
        "analysis_id": analysis_id,
        "saved_at": time.time(),
        "result": result,
    }, indent=2), encoding="utf-8")
    
    from app.database import update_analysis_modular_paths
    await update_analysis_modular_paths(analysis_id, ball_path=str(path))
    
    return {"analysis_id": analysis_id, "saved_path": str(path), "result": result}


@router.get("/hit-study/{analysis_id}/ball-scan/load")
async def load_ball_scan(analysis_id: str) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    path = _ball_scan_path(analysis_id)
    
    if analysis.get("active_ball_scan_path"):
        p = Path(analysis["active_ball_scan_path"])
        if p.exists():
            path = p

    if not path.exists():
        raise HTTPException(404, f"saved scan not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("result") or data
    return {"result": result, "source_file": path.name}


def _run_ball_scan_job(
    job_id: str,
    video_path: Path,
    artifact: dict,
    ranges: list[tuple[float, float]],
    target_times: list[float],
    payload: BallScanRequest,
) -> None:
    job = BALL_SCAN_JOBS[job_id]
    job["status"] = "running"
    detector = payload.ball_detector
    source_fps = _video_fps(video_path)
    step_s = max(0.05, _scan_step_s(payload, detector, source_fps))
    window_s = max(0.08, min(0.3, step_s * 0.75))
    centers = []
    for start_s, end_s in ranges:
        t = start_s
        while t <= end_s + 1e-6:
            centers.append(t)
            t += step_s
    started = time.monotonic()
    seen = set()
    candidates = []
    fps = 0.0
    result = {
        "range_start_s": min(r[0] for r in ranges),
        "range_end_s": max(r[1] for r in ranges),
        "ranges": [{"start_s": r[0], "end_s": r[1]} for r in ranges],
        "target_times_s": sorted({float(t) for t in target_times}),
        "target_count": len(set(target_times)),
        "scan_range": payload.scan_range,
        "scan_target": payload.scan_target,
        "scan_mode": payload.scan_target,
        "scan_fps": payload.scan_fps,
        "scan_step_s": step_s,
        "ball_detector": detector,
        "roi_mode": payload.roi_mode,
        "exclude_player": payload.exclude_player,
        "diff_threshold": payload.diff_threshold,
        "tracknet_width": int(payload.tracknet_width),
        "tracknet_height": int(payload.tracknet_height),
        "tracknet_stack_frames": 3,
        "fps": source_fps,
        "candidate_count": 0,
        "candidates": [],
        "partial": True,
    }
    job["result"] = result
    try:
        for idx, center in enumerate(centers):
            if job.get("cancel_requested"):
                break
            pct = (idx / max(1, len(centers))) * 100.0
            elapsed = time.monotonic() - started
            rate = idx / elapsed if elapsed > 0 and idx > 0 else 0.0
            eta = (len(centers) - idx) / rate if rate > 0 else None
            job["progress_percent"] = pct
            job["progress_message"] = f"{detector} scan {idx + 1:,}/{len(centers):,}"
            job["progress_eta_s"] = eta
            data = ball_motion_diagnostic(
                video_path,
                artifact,
                center,
                window_s=window_s,
                roi_expand_x=payload.roi_expand_x,
                roi_expand_up=payload.roi_expand_up,
                roi_expand_down=payload.roi_expand_down,
                roi_mode=payload.roi_mode,
                exclude_player=payload.exclude_player,
                diff_threshold=payload.diff_threshold,
                ball_detector=detector,
                tracknet_width=int(payload.tracknet_width),
                tracknet_height=int(payload.tracknet_height),
            )
            fps = float(data.get("fps") or fps)
            for c in data.get("candidates") or []:
                ct = float(c.get("time_s", 0.0))
                if not any(start <= ct <= end for start, end in ranges):
                    continue
                key = (round(ct, 2), round(float(c.get("x", 0.0)), 3), round(float(c.get("y", 0.0)), 3), c.get("source", detector))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(c)
            candidates.sort(key=lambda c: (float(c.get("time_s", 0.0)), -float(c.get("score", 0.0))))
            result["fps"] = fps or source_fps
            result["candidate_count"] = len(candidates)
            result["candidates"] = candidates[:5000]
            job["result"] = dict(result)
        if job.get("cancel_requested"):
            result["partial"] = True
            result["cancelled"] = True
            job["status"] = "canceled"
            job["progress_message"] = f"canceled: {len(candidates):,} candidates"
            job["result"] = result
            return
        candidates.sort(key=lambda c: (float(c.get("time_s", 0.0)), -float(c.get("score", 0.0))))
        job["status"] = "done"
        job["progress_percent"] = 100.0
        job["progress_message"] = f"done: {len(candidates):,} candidates"
        job["progress_eta_s"] = None
        result["partial"] = False
        result["cancelled"] = False
        result["fps"] = fps or source_fps
        result["candidate_count"] = len(candidates)
        result["candidates"] = candidates[:5000]
        job["result"] = result
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["progress_message"] = f"error: {e}"


def _scan_ranges(payload: BallScanRequest, start_s: float, end_s: float, target_times: list[float]) -> list[tuple[float, float]]:
    before = max(0.0, float(payload.mark_before_s))
    after = max(0.0, float(payload.mark_after_s))
    ranges = [
        (max(start_s, t - before), min(end_s, t + after))
        for t in target_times
        if start_s <= t <= end_s
    ]
    ranges = [(a, b) for a, b in ranges if b > a]
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 0.05:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _scan_step_s(payload: BallScanRequest, detector: str, source_fps: float) -> float:
    if payload.step_s is not None:
        return float(payload.step_s)
    if payload.scan_fps == "original":
        return 1.0 / max(1.0, source_fps)
    try:
        fps = float(payload.scan_fps)
    except (TypeError, ValueError):
        fps = 2.0 if detector == "tracknet" else 4.0
    return 1.0 / max(0.1, min(max(1.0, source_fps), fps))


def _video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    finally:
        cap.release()


def _ball_scan_path(analysis_id: str) -> Path:
    path = settings.analysis_dir.parent / "ball_scans"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{analysis_id}.ball-scan.json"


def _modular_pose_scan_path(analysis_id: str) -> Path:
    return settings.analysis_dir.parent / "modular_scans" / f"{analysis_id}.pose-scan.json.gz"


def _modular_audio_scan_path(analysis_id: str) -> Path:
    return settings.analysis_dir.parent / "modular_scans" / f"{analysis_id}.audio-scan.json"


def _load_hit_study_context(analysis) -> dict:
    analysis = dict(analysis)
    pose_path = _modular_pose_scan_path(analysis["id"])
    audio_path = _modular_audio_scan_path(analysis["id"])
    
    # If the database has active paths, use them if they exist
    if analysis.get("active_pose_scan_path"):
        p = Path(analysis["active_pose_scan_path"])
        if p.exists():
            pose_path = p
    if analysis.get("active_audio_scan_path"):
        p = Path(analysis["active_audio_scan_path"])
        if p.exists():
            audio_path = p

    # If no modular scans exist, fallback to old monolithic artifact if present
    if not pose_path.exists() and not audio_path.exists() and analysis["artifact_path"]:
        path = Path(analysis["artifact_path"])
        if not path.exists():
            raise HTTPException(410, "hit study artifact file missing")
        art = load_hit_study_artifact(path)
        art["pose_source_file"] = path.name
        art["audio_source_file"] = path.name
        return art

    audio = {"impacts": []}
    audio_source = None
    if audio_path.exists():
        payload = json.loads(audio_path.read_text(encoding="utf-8"))
        audio_result = payload.get("result") or payload
        audio = {
            "impacts": audio_result.get("impacts") or [],
            "knobs": audio_result.get("knobs") or {},
            "sample_rate": audio_result.get("sample_rate"),
            "noise_floor": audio_result.get("noise_floor"),
            "range_start_s": audio_result.get("range_start_s"),
            "range_end_s": audio_result.get("range_end_s"),
            "impact_count": audio_result.get("impact_count", len(audio_result.get("impacts") or [])),
        }
        audio_source = audio_path.name

    if pose_path.exists():
        with gzip.open(pose_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        result = payload.get("result") or payload
        frames = result.get("frames") or []
        metadata = {
            "detector": HIT_STUDY_ALGORITHM,
            "detector_version": 1,
            "duration_s": float(analysis["duration_s"]),
            "range_start_s": float(result.get("range_start_s", analysis["range_start_s"] or 0.0)),
            "range_end_s": float(result.get("range_end_s", analysis["range_end_s"] or analysis["duration_s"])),
            "target_width": int(result.get("target_width", 640)),
            "target_height": int(result.get("target_height", 360)),
            "sample_fps": float(result.get("sample_fps", 4.0)),
            "sample_count": len(frames),
        }
        return {
            "metadata": metadata,
            "summary": result.get("summary") or {},
            "frames": frames,
            "audio": audio,
            "feature_windows": [],
            "pose_source_file": pose_path.name,
            "audio_source_file": audio_source,
        }

    return {
        "metadata": {
            "detector": HIT_STUDY_ALGORITHM,
            "detector_version": 1,
            "duration_s": float(analysis["duration_s"]),
            "range_start_s": float(analysis["range_start_s"] or 0.0),
            "range_end_s": float(analysis["range_end_s"] or analysis["duration_s"]),
            "target_width": 640,
            "target_height": 360,
            "sample_fps": 4.0,
            "sample_count": 0,
        },
        "summary": {},
        "frames": [],
        "audio": audio,
        "feature_windows": [],
        "audio_source_file": audio_source,
    }


@router.post("/hit-study/{analysis_id}/labels/save")
async def save_hit_labels(analysis_id: str, filename: str | None = None) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # Allowed for any algorithm now
    rows = [
        r for r in await list_strike_labels(analysis_id)
        if r["source"] == "near_player_hit" and bool(r["is_strike"])
    ]
    labels = [_row_to_label(r) for r in rows]
    labels.sort(key=lambda r: r["time_s"])
    path = _label_path(analysis_id, filename)
    payload = {
        "analysis_id": analysis_id,
        "video_id": analysis["video_id"],
        "filename": analysis["filename"],
        "algorithm": analysis["algorithm"],
        "saved_at": time.time(),
        "labels": [
            {
                "time_s": label["time_s"],
                "source": "near_player_hit",
                "is_strike": True,
                "comment": label.get("comment"),
            }
            for label in labels
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"analysis_id": analysis_id, "saved_path": str(path), "labels": labels}


@router.post("/hit-study/{analysis_id}/labels/load")
async def load_hit_labels(analysis_id: str, filename: str | None = None) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # Allowed for any algorithm now
    path = _label_path(analysis_id, filename)
    if not path.exists():
        raise HTTPException(404, f"saved labels not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"saved label file is invalid JSON: {e}") from e

    existing = [
        r for r in await list_strike_labels(analysis_id)
        if r["source"] == "near_player_hit"
    ]
    for row in existing:
        await delete_strike_label(row["id"])

    for label in payload.get("labels") or []:
        await upsert_strike_label(
            analysis_id,
            time_s=float(label["time_s"]),
            source="near_player_hit",
            is_strike=bool(label.get("is_strike", True)),
            algorithm_validated=None,
            comment=label.get("comment"),
        )
    rows = [
        r for r in await list_strike_labels(analysis_id)
        if r["source"] == "near_player_hit" and bool(r["is_strike"])
    ]
    labels = [_row_to_label(r) for r in rows]
    labels.sort(key=lambda r: r["time_s"])
    return {"analysis_id": analysis_id, "loaded_path": str(path), "labels": labels}


@router.post("/hit-study/{analysis_id}/labels/upload")
async def upload_hit_labels(analysis_id: str, payload: dict) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    # Allowed for any algorithm now

    existing = [
        r for r in await list_strike_labels(analysis_id)
        if r["source"] == "near_player_hit"
    ]
    for row in existing:
        await delete_strike_label(row["id"])

    for label in payload.get("labels") or []:
        await upsert_strike_label(
            analysis_id,
            time_s=float(label["time_s"]),
            source="near_player_hit",
            is_strike=bool(label.get("is_strike", True)),
            algorithm_validated=None,
            comment=label.get("comment"),
        )
    rows = [
        r for r in await list_strike_labels(analysis_id)
        if r["source"] == "near_player_hit" and bool(r["is_strike"])
    ]
    labels = [_row_to_label(r) for r in rows]
    labels.sort(key=lambda r: r["time_s"])
    return {"analysis_id": analysis_id, "labels": labels}
