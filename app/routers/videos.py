import logging
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database import delete_video_cascade, get_video, list_media, list_videos, update_video_court_calibration
from app.models import MediaSummary, VideoSummary

logger = logging.getLogger(__name__)
router = APIRouter()


class CourtCalibrationRequest(BaseModel):
    points: list[dict]


@router.get("/videos", response_model=list[VideoSummary])
async def list_all_videos() -> list[VideoSummary]:
    rows = await list_videos()
    return [
        VideoSummary(
            video_id=r["id"],
            filename=r["filename"],
            status=r["status"],
            error_msg=r["error_msg"],
            duration_s=r["duration_s"],
            segment_count=r["segment_count"] or 0,
            on_segment_count=r["on_segment_count"] or 0,
            created_at=r["created_at"],
            progress_percent=r["progress_percent"] if "progress_percent" in r.keys() else None,
            progress_message=r["progress_message"] if "progress_message" in r.keys() else None,
        )
        for r in rows
    ]


@router.get("/media", response_model=list[MediaSummary])
async def media_pool() -> list[MediaSummary]:
    rows = await list_media()
    return [
        MediaSummary(
            video_id=r["id"],
            filename=r["filename"],
            filepath=r["filepath"],
            duration_s=r["duration_s"],
            content_hash=r["content_hash"],
            size_bytes=r["size_bytes"],
            created_at=r["created_at"],
            analysis_count=r["analysis_count"] or 0,
            has_court_calibration=bool(r["court_calibration_json"]),
            court_calibration=json.loads(r["court_calibration_json"]) if r["court_calibration_json"] else None,
        )
        for r in rows
    ]


@router.put("/media/{video_id}/court-calibration")
async def save_court_calibration(video_id: str, payload: CourtCalibrationRequest) -> dict:
    video = await get_video(video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    if len(payload.points) != 4:
        raise HTTPException(400, "court calibration requires exactly 4 points")
    clean = []
    for p in payload.points:
        x = float(p.get("x", -1))
        y = float(p.get("y", -1))
        if x < 0 or x > 1 or y < 0 or y > 1:
            raise HTTPException(400, "points must be normalized between 0 and 1")
        clean.append({"x": x, "y": y})
    calibration = {"points": clean}
    await update_video_court_calibration(video_id, json.dumps(calibration))
    return {"video_id": video_id, "court_calibration": calibration}


@router.delete("/videos/{video_id}")
async def delete_video(video_id: str) -> dict:
    paths = await delete_video_cascade(video_id)
    if paths["upload"] is None:
        raise HTTPException(404, "video not found")

    removed = []
    upload_paths = [paths["upload"]] if paths.get("delete_upload", True) else []
    for p in [*upload_paths, *paths["exports"]]:
        if not p:
            continue
        path = Path(p)
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        except OSError as e:
            logger.warning("could not delete %s: %s", p, e)

    logger.info("deleted video_id=%s removed=%d files", video_id, len(removed))
    return {"video_id": video_id, "removed_files": removed}
