from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.database import get_export

router = APIRouter()


@router.get("/download/{export_id}")
async def download_export(export_id: str) -> FileResponse:
    row = await get_export(export_id)
    if row is None:
        raise HTTPException(404, "export not found")
    if row["status"] != "done" or not row["output_filepath"]:
        raise HTTPException(409, "export not ready")
    path = Path(row["output_filepath"])
    if not path.exists():
        raise HTTPException(410, "export file missing")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"highlights_{export_id[:8]}.mp4",
    )
