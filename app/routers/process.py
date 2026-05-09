import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.database import (
    create_analysis,
    delete_analysis_run,
    get_analysis_run,
    get_video,
    list_analysis_runs,
)
from app.models import AnalysisRunSummary, DetectorKnobs
from app.pipeline.motion_analysis import default_knobs, normalize_knobs
from app.pipeline.near_player_hit_study import HIT_STUDY_ALGORITHM
from app.pipeline.pose_analysis import POSE_ALGORITHM
from app.pipeline.orchestrator import run_analysis

router = APIRouter()


class StartAnalysisRequest(BaseModel):
    algorithm: str = "median_frame"
    knobs: DetectorKnobs | None = None
    config: dict | None = None
    range_start_s: float | None = None
    range_end_s: float | None = None


@router.post("/process/{video_id}")
async def start_processing(
    video_id: str,
    background_tasks: BackgroundTasks,
    payload: StartAnalysisRequest | None = None,
) -> dict:
    video = await get_video(video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    req = payload or StartAnalysisRequest()
    if req.algorithm not in {"median_frame", "median_court_roi", POSE_ALGORITHM, HIT_STUDY_ALGORITHM}:
        raise HTTPException(400, "unsupported algorithm")
    if req.algorithm == "median_court_roi" and not video["court_calibration_json"]:
        raise HTTPException(400, "court calibration is required for median_court_roi")

    knobs = normalize_knobs(req.knobs.model_dump() if req.knobs else default_knobs())
    duration = float(video["duration_s"])
    range_start, range_end = _analysis_range(req.range_start_s, req.range_end_s, duration)
    analysis_id = await create_analysis(
        video_id,
        algorithm=req.algorithm,
        noninstant_knobs_json=json.dumps({
            "sample_fps": knobs["sample_fps"],
            "median_bg_samples": knobs["median_bg_samples"],
            "court_weight": knobs["court_weight"],
            "outside_weight": knobs["outside_weight"],
            "near_camera_weight": knobs["near_camera_weight"],
            **(req.config or {}),
            "range_start_s": range_start,
            "range_end_s": range_end,
        }),
        instant_knobs_json=json.dumps(knobs),
        range_start_s=range_start,
        range_end_s=range_end,
    )
    background_tasks.add_task(run_analysis, analysis_id)
    return {"video_id": video_id, "analysis_id": analysis_id, "status": "pending"}


@router.get("/analyses", response_model=list[AnalysisRunSummary])
async def analyses() -> list[AnalysisRunSummary]:
    rows = await list_analysis_runs()
    return [_analysis_summary(r) for r in rows]


@router.get("/analysis-status/{analysis_id}", response_model=AnalysisRunSummary)
async def analysis_status(analysis_id: str) -> AnalysisRunSummary:
    row = await get_analysis_run(analysis_id)
    if row is None:
        raise HTTPException(404, "analysis not found")
    return _analysis_summary(row)


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str) -> dict:
    row = await get_analysis_run(analysis_id)
    if row is None:
        raise HTTPException(404, "analysis not found")
    if row["status"] == "analyzing":
        raise HTTPException(409, "cannot delete an actively running analysis")
    result = await delete_analysis_run(analysis_id)
    removed = []
    if result and result.get("artifact_path"):
        path = Path(result["artifact_path"])
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        except OSError:
            pass
    return {"analysis_id": analysis_id, "removed_files": removed}


def _analysis_summary(r) -> AnalysisRunSummary:
    knobs = None
    if r["noninstant_knobs_json"]:
        try:
            knobs = json.loads(r["noninstant_knobs_json"])
        except json.JSONDecodeError:
            knobs = None
    return AnalysisRunSummary(
        analysis_id=r["id"],
        video_id=r["video_id"],
        filename=r["filename"],
        algorithm=r["algorithm"],
        status=r["status"],
        error_msg=r["error_msg"],
        duration_s=r["duration_s"],
        segment_count=r["segment_count"] if "segment_count" in r.keys() else 0,
        on_segment_count=r["on_segment_count"] if "on_segment_count" in r.keys() else 0,
        progress_percent=r["progress_percent"],
        progress_message=r["progress_message"],
        progress_eta_s=r["progress_eta_s"] if "progress_eta_s" in r.keys() else None,
        range_start_s=r["range_start_s"] if "range_start_s" in r.keys() else None,
        range_end_s=r["range_end_s"] if "range_end_s" in r.keys() else None,
        noninstant_knobs=knobs,
        created_at=r["created_at"],
    )


def _analysis_range(
    start_s: float | None,
    end_s: float | None,
    duration_s: float,
) -> tuple[float, float]:
    start = 0.0 if start_s is None else max(0.0, min(duration_s, float(start_s)))
    end = duration_s if end_s is None else max(0.0, min(duration_s, float(end_s)))
    if end <= start:
        raise HTTPException(400, "analysis range must have end after start")
    return start, end
