import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.database import get_analysis_run, get_segments, get_segments_for_analysis, get_video, replace_segments
from app.models import Segment, SegmentsResponse
from app.pipeline.ffmpeg_utils import probe_video_fps

router = APIRouter()


def segment_from_row(r) -> Segment:
    samples = None
    if "samples_json" in r.keys() and r["samples_json"]:
        try:
            samples = json.loads(r["samples_json"])
        except json.JSONDecodeError:
            samples = []
    return Segment(
        start_s=float(r["start_s"]),
        end_s=float(r["end_s"]),
        is_on=bool(r["is_on"]),
        source=r["source"] if "source" in r.keys() else "manual",
        raw_start_s=r["raw_start_s"] if "raw_start_s" in r.keys() else None,
        raw_end_s=r["raw_end_s"] if "raw_end_s" in r.keys() else None,
        avg_score=r["avg_score"] if "avg_score" in r.keys() else None,
        max_score=r["max_score"] if "max_score" in r.keys() else None,
        min_score=r["min_score"] if "min_score" in r.keys() else None,
        sample_count=r["sample_count"] if "sample_count" in r.keys() else None,
        decision_stage=r["decision_stage"] if "decision_stage" in r.keys() else None,
        samples=samples,
    )


@router.get("/segments/{video_id}", response_model=SegmentsResponse)
async def list_segments(video_id: str) -> SegmentsResponse:
    video = await get_video(video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    rows = await get_segments(video_id)
    return SegmentsResponse(
        video_id=video_id,
        duration_s=float(video["duration_s"]),
        segments=[segment_from_row(r) for r in rows],
    )


@router.get("/analysis-segments/{analysis_id}", response_model=SegmentsResponse)
async def list_analysis_segments(analysis_id: str) -> SegmentsResponse:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    rows = await get_segments_for_analysis(analysis_id)
    return SegmentsResponse(
        video_id=analysis["video_id"],
        duration_s=float(analysis["duration_s"]),
        segments=[segment_from_row(r) for r in rows],
    )


@router.patch("/segments/{video_id}", response_model=SegmentsResponse)
async def save_segments(video_id: str, payload: SegmentsResponse) -> SegmentsResponse:
    video = await get_video(video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    duration = float(video["duration_s"])

    segs = []
    for s in payload.segments:
        start = max(0.0, min(duration, float(s.start_s)))
        end = max(0.0, min(duration, float(s.end_s)))
        if end <= start:
            continue
        segs.append({
            "start_s": start,
            "end_s": end,
            "is_on": s.is_on,
            "source": s.source,
            "raw_start_s": s.raw_start_s,
            "raw_end_s": s.raw_end_s,
            "avg_score": s.avg_score,
            "max_score": s.max_score,
            "min_score": s.min_score,
            "sample_count": s.sample_count,
            "decision_stage": s.decision_stage,
            "samples": s.samples or [],
        })
    segs.sort(key=lambda x: x["start_s"])

    await replace_segments(video_id, segs)
    rows = await get_segments(video_id)
    return SegmentsResponse(
        video_id=video_id,
        duration_s=duration,
        segments=[segment_from_row(r) for r in rows],
    )


@router.get("/video-file/{video_id}")
async def video_file_path(video_id: str) -> dict:
    """Return the public URL of the original video file for the player."""
    video = await get_video(video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    path = Path(video["filepath"])
    name = path.name
    source_fps = None
    try:
        source_fps = await probe_video_fps(path)
    except Exception:
        source_fps = None
    return {"url": f"/uploads/{name}", "source_fps": source_fps}
