import hashlib
import logging
import re
import time
from uuid import uuid4
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.database import (
    create_video,
    get_video_by_content_hash,
    get_video_by_filename,
    update_video_asset_info,
)
from app.models import UploadResponse
from app.pipeline.ffmpeg_utils import probe_duration

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXT = {".mp4", ".mkv", ".mov", ".m4v"}


def _safe_filename(raw: str) -> str:
    """Strip directory traversal and fs-unfriendly characters."""
    name = Path(raw).name  # drops any path components
    name = re.sub(r"[^\w.\-\s]", "_", name).strip()
    return name or "upload.mp4"


def _available_original_name(safe_name: str, content_hash: str) -> Path:
    original = settings.upload_dir / safe_name
    if not original.exists():
        return original
    return settings.upload_dir / f"{content_hash[:12]}_{safe_name}"


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    safe_name = _safe_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    temp_dest = settings.upload_dir / f".upload_{uuid4().hex}_{safe_name}"
    existing = await get_video_by_filename(safe_name)

    logger.info(
        "upload start: %s (existing=%s)",
        safe_name, existing["id"] if existing else "none",
    )
    started = time.monotonic()
    bytes_written = 0
    last_log = started
    digest = hashlib.sha256()

    async with aiofiles.open(temp_dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            await f.write(chunk)
            bytes_written += len(chunk)
            now = time.monotonic()
            if now - last_log >= 5.0:
                mb = bytes_written / (1024 * 1024)
                rate = mb / (now - started) if now - started > 0 else 0
                logger.info("  upload progress: %.1f MB written (%.1f MB/s)", mb, rate)
                last_log = now

    elapsed = time.monotonic() - started
    mb = bytes_written / (1024 * 1024)
    logger.info(
        "upload finished: %.1f MB in %.1fs (%.1f MB/s)",
        mb, elapsed, mb / elapsed if elapsed > 0 else 0,
    )

    content_hash = digest.hexdigest()
    duplicate = await get_video_by_content_hash(content_hash)
    if duplicate is not None:
        dest = Path(duplicate["filepath"])
        temp_dest.unlink(missing_ok=True)
        duration_s = float(duplicate["duration_s"])
        logger.info(
            "upload deduped by content hash: using %s from video_id=%s",
            dest.name, duplicate["id"],
        )
    else:
        dest = _available_original_name(safe_name, content_hash)
        if dest.exists():
            temp_dest.unlink(missing_ok=True)
        else:
            temp_dest.replace(dest)
        try:
            duration_s = await probe_duration(dest)
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not probe video: {e}")

    try:
        size_bytes = dest.stat().st_size
    except Exception as e:
        raise HTTPException(400, f"Could not stat video: {e}")

    if existing is not None:
        video_id = existing["id"]
        await update_video_asset_info(
            video_id,
            filepath=str(dest),
            duration_s=duration_s,
            content_hash=content_hash,
            size_bytes=size_bytes,
        )
        logger.info(
            "upload reused existing row: video_id=%s file=%s", video_id, safe_name
        )
    else:
        video_id = await create_video(
            safe_name,
            str(dest),
            duration_s,
            content_hash=content_hash,
            size_bytes=size_bytes,
        )
        logger.info(
            "upload registered: video_id=%s duration=%.1fs file=%s",
            video_id, duration_s, dest.name,
        )

    return UploadResponse(
        video_id=video_id, filename=safe_name, duration_s=duration_s
    )
