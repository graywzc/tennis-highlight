from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.database import get_analysis_run
from app.pipeline.pose_analysis import POSE_ALGORITHM, load_pose_artifact

router = APIRouter()


@router.get("/pose-data/{analysis_id}")
async def pose_data(analysis_id: str) -> dict:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    if analysis["algorithm"] != POSE_ALGORITHM:
        raise HTTPException(400, "analysis is not a pose analysis")
    if not analysis["artifact_path"]:
        raise HTTPException(409, "pose artifact is not ready")
    path = Path(analysis["artifact_path"])
    if not path.exists():
        raise HTTPException(410, "pose artifact file missing")
    return load_pose_artifact(path)
