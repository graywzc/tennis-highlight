from fastapi import APIRouter, HTTPException

from app.database import get_video
from app.models import StatusResponse

router = APIRouter()


@router.get("/status/{video_id}", response_model=StatusResponse)
async def get_status(video_id: str) -> StatusResponse:
    video = await get_video(video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    return StatusResponse(
        video_id=video_id,
        status=video["status"],
        error_msg=video["error_msg"],
        duration_s=video["duration_s"],
        progress_percent=video["progress_percent"] if "progress_percent" in video.keys() else None,
        progress_message=video["progress_message"] if "progress_message" in video.keys() else None,
    )
