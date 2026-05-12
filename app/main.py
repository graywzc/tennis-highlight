import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import analysis, download, export, hit_study, modular_scan, pose, process, segments, status, strike_labels, upload, videos


def _configure_logging() -> None:
    level = logging.DEBUG if settings.verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quiet down noisy libs unless verbose
    if not settings.verbose:
        logging.getLogger("multipart").setLevel(logging.WARNING)
        logging.getLogger("python_multipart").setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger("app")


def _ensure_runtime_dirs() -> None:
    for d in (
        settings.upload_dir,
        settings.exports_dir,
        settings.segments_tmp_dir,
        settings.analysis_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_runtime_dirs()
    await init_db()
    logger.info(
        "startup complete (verbose=%s, motion_threshold=%.3f, sample_fps=%.1f)",
        settings.verbose, settings.motion_threshold, settings.sample_fps,
    )
    yield


app = FastAPI(title="Tennis Highlight Extractor", lifespan=lifespan)

app.include_router(upload.router)
app.include_router(videos.router)
app.include_router(process.router)
app.include_router(status.router)
app.include_router(analysis.router)
app.include_router(pose.router)
app.include_router(hit_study.router)
app.include_router(modular_scan.router)
app.include_router(strike_labels.router)
app.include_router(segments.router)
app.include_router(export.router)
app.include_router(download.router)

_ensure_runtime_dirs()
app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/{analysis_id}")
async def index_analysis(analysis_id: str):
    # This acts as a catch-all for our SPA so the frontend can read the URL path
    return FileResponse("static/index.html")
