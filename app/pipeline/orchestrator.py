import asyncio
import json
import logging
import time
from pathlib import Path

from app.database import (
    complete_analysis,
    get_analysis_run,
    get_export,
    get_video,
    replace_segments,
    update_analysis_progress_sync,
    update_analysis_status,
    update_export,
)
from app.config import settings
from app.pipeline.ffmpeg_utils import probe_duration
from app.pipeline.motion_analysis import (
    analyze_video_to_artifact,
    default_knobs,
    normalize_knobs,
)
from app.pipeline.near_player_hit_study import HIT_STUDY_ALGORITHM, analyze_hit_study_to_artifact
from app.pipeline.pose_analysis import POSE_ALGORITHM, analyze_pose_to_artifact
from app.pipeline.video_editor import export_segments

logger = logging.getLogger(__name__)


async def run_analysis(analysis_id: str) -> None:
    """Background task: analyze a video for in-play segments and store them."""
    started = time.monotonic()
    try:
        await update_analysis_status(analysis_id, "analyzing")
        analysis = await get_analysis_run(analysis_id)
        if analysis is None:
            raise RuntimeError("analysis not found")

        video_id = analysis["video_id"]
        video_path = Path(analysis["filepath"])
        duration = float(analysis["duration_s"])
        logger.info(
            "analysis start: analysis_id=%s video_id=%s file=%s duration=%.1fs",
            analysis_id, video_id, video_path.name, duration,
        )

        def progress_cb(percent: float, message: str, eta_s: float | None = None) -> None:
            update_analysis_progress_sync(analysis_id, percent, message, eta_s)

        try:
            knobs = normalize_knobs(json.loads(analysis["instant_knobs_json"]), default_knobs())
        except Exception:
            knobs = default_knobs()
        try:
            config = json.loads(analysis["noninstant_knobs_json"] or "{}")
        except Exception:
            config = {}
        if analysis["algorithm"] in {POSE_ALGORITHM, HIT_STUDY_ALGORITHM}:
            rally_keys = (
                "audio_sample_rate", "bandpass_low_hz", "bandpass_high_hz",
                "peak_height_mad_k", "peak_prominence_mult", "min_impact_separation_s",
                "min_spectral_centroid_hz",
                "pose_window_s", "wrist_conf_min", "min_wrist_velocity",
                "max_gap_s", "min_hits_per_rally", "rally_padding_s",
            )
            rally_overrides = {k: config[k] for k in rally_keys if k in config}
            analyzer = analyze_pose_to_artifact if analysis["algorithm"] == POSE_ALGORITHM else analyze_hit_study_to_artifact
            artifact_path, timeline, analysis_result = await asyncio.to_thread(
                analyzer,
                analysis_id,
                video_path,
                duration,
                float(knobs["sample_fps"]),
                float(analysis["range_start_s"] or 0.0),
                float(analysis["range_end_s"] or duration),
                progress_cb,
                str(config.get("pose_model", "yolo11n-pose.pt")),
                float(config.get("pose_conf", 0.25)),
                int(config.get("pose_imgsz", 640)),
                rally_overrides or None,
            )
        else:
            court_calibration = None
            if analysis["algorithm"] == "median_court_roi":
                video = await get_video(video_id)
                if video is None or not video["court_calibration_json"]:
                    raise RuntimeError("court calibration is required for median_court_roi")
                court_calibration = json.loads(video["court_calibration_json"])
            artifact_path, timeline, analysis_result = await asyncio.to_thread(
                analyze_video_to_artifact,
                analysis_id,
                video_path,
                duration,
                lambda pct, msg: progress_cb(pct, msg, None),
                knobs,
                analysis["algorithm"],
                court_calibration,
            )
        summary = analysis_result.get("summary", analysis_result)

        await replace_segments(video_id, timeline, analysis_id=analysis_id)
        await complete_analysis(
            analysis_id,
            artifact_path=str(artifact_path),
            instant_knobs_json=json.dumps(analysis_result.get("config") or analysis_result.get("knobs") or {}),
        )
        elapsed = time.monotonic() - started
        logger.info(
            "analysis complete: analysis_id=%s in_play_segments=%d total_segments=%d elapsed=%.1fs",
            analysis_id, summary.get("on_count", 0), len(timeline), elapsed,
        )
    except Exception as e:
        logger.exception("analysis failed for %s", analysis_id)
        await update_analysis_status(analysis_id, "error", error_msg=str(e))


async def run_export(export_id: str, video_id: str, segments: list[dict]) -> None:
    """Background task: stitch user-selected segments into a final video."""
    started = time.monotonic()
    try:
        await update_export(export_id, status="exporting")
        video = await get_video(video_id)
        if video is None:
            raise RuntimeError("video not found")

        on_pairs = [
            (float(s["start_s"]), float(s["end_s"]))
            for s in segments
            if s.get("is_on") and float(s["end_s"]) > float(s["start_s"])
        ]
        if not on_pairs:
            raise RuntimeError("no segments selected for export")

        total_kept = sum(e - s for s, e in on_pairs)
        logger.info(
            "export start: export_id=%s video_id=%s kept_segments=%d kept_duration=%.1fs",
            export_id, video_id, len(on_pairs), total_kept,
        )

        output_path = settings.exports_dir / f"{export_id}.mp4"
        await export_segments(
            Path(video["filepath"]), on_pairs, output_path, export_id
        )

        duration_s = await probe_duration(output_path)
        await update_export(
            export_id,
            status="done",
            output_filepath=str(output_path),
            duration_s=duration_s,
        )
        elapsed = time.monotonic() - started
        logger.info(
            "export complete: export_id=%s output=%s duration=%.1fs elapsed=%.1fs",
            export_id, output_path.name, duration_s, elapsed,
        )
    except Exception as e:
        logger.exception("export failed for %s", export_id)
        await update_export(export_id, status="error", error_msg=str(e))
