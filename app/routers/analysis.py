import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.database import (
    get_analysis_run,
    get_segments_for_analysis,
    replace_segments,
    update_analysis_knobs,
)
from app.models import (
    AudioPreviewRequest,
    AudioPreviewResponse,
    AnalysisMetadata,
    AnalysisPreviewRequest,
    AnalysisPreviewResponse,
    AnalysisResponse,
    DetectorKnobs,
    Segment,
)
from app.pipeline.motion_analysis import default_knobs, load_artifact, normalize_knobs, recompute_from_artifact
from app.pipeline.near_player_hit_study import HIT_STUDY_ALGORITHM, load_hit_study_artifact
from app.pipeline.pose_analysis import POSE_ALGORITHM, load_pose_artifact
from app.pipeline.rally_detection import default_rally_knobs, preview_audio_range
from app.routers.segments import segment_from_row

router = APIRouter()


def _knobs_model(knobs: dict) -> DetectorKnobs:
    return DetectorKnobs(**normalize_knobs(knobs))


def _metadata_model(metadata: dict, analysis_id: str | None = None) -> AnalysisMetadata:
    return AnalysisMetadata(
        analysis_id=analysis_id,
        detector=metadata["detector"],
        detector_version=int(metadata["detector_version"]),
        duration_s=float(metadata["duration_s"]),
        target_width=int(metadata["target_width"]),
        target_height=int(metadata["target_height"]),
        sample_fps=float(metadata["sample_fps"]),
        median_bg_samples=int(metadata["median_bg_samples"]),
        sample_count=int(metadata["sample_count"]),
    )


def _segment_models(segments: list[dict]) -> list[Segment]:
    return [Segment(**seg) for seg in segments]


def _artifact_path(analysis) -> Path:
    path = analysis["artifact_path"]
    if not path:
        raise HTTPException(409, "analysis artifact missing; wait for this analysis to finish")
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise HTTPException(410, "analysis artifact file missing; run the analysis again")
    return artifact_path


def _current_knobs(analysis, defaults: dict) -> dict:
    if analysis["instant_knobs_json"]:
        try:
            return normalize_knobs(json.loads(analysis["instant_knobs_json"]), defaults)
        except json.JSONDecodeError:
            pass
    return normalize_knobs(defaults, defaults)


def _lock_reanalysis_only(knobs: dict, metadata: dict) -> dict:
    locked = dict(knobs)
    locked["sample_fps"] = float(metadata["sample_fps"])
    locked["median_bg_samples"] = int(metadata["median_bg_samples"])
    return locked


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str) -> AnalysisResponse:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")

    metadata = None
    defaults = default_knobs()
    summary = {
        "sample_count": 0,
        "on_count": 0,
        "total_count": 0,
        "on_duration_s": 0.0,
        "on_percent": 0.0,
        "foreground_percentiles": {},
        "smoothed_percentiles": {},
        "changed_segment_count": None,
    }

    if analysis["artifact_path"]:
        artifact_path = _artifact_path(analysis)
        if analysis["algorithm"] in {POSE_ALGORITHM, HIT_STUDY_ALGORITHM}:
            artifact = load_pose_artifact(artifact_path) if analysis["algorithm"] == POSE_ALGORITHM else load_hit_study_artifact(artifact_path)
            metadata = {
                **artifact["metadata"],
                "median_bg_samples": 0,
            }
            defaults = normalize_knobs(defaults, defaults)
            knobs = _current_knobs(analysis, defaults)
            summary = artifact.get("summary", {})
            summary["config"] = artifact.get("metadata", {})
        else:
            artifact = load_artifact(artifact_path)
            metadata = artifact["metadata"]
            defaults = normalize_knobs(metadata.get("default_knobs") or defaults, defaults)
            knobs = _current_knobs(analysis, defaults)
            result = recompute_from_artifact(artifact_path, knobs, include_samples=False)
            summary = result["summary"]
    else:
        if analysis["instant_knobs_json"]:
            try:
                knobs = normalize_knobs(json.loads(analysis["instant_knobs_json"]), defaults)
            except json.JSONDecodeError:
                knobs = normalize_knobs(defaults, defaults)
        else:
            knobs = normalize_knobs(defaults, defaults)

    return AnalysisResponse(
        video_id=analysis["video_id"],
        analysis_id=analysis_id,
        duration_s=float(analysis["duration_s"]),
        defaults=_knobs_model(defaults),
        knobs=_knobs_model(knobs),
        metadata=_metadata_model(metadata, analysis_id) if metadata else None,
        summary=summary,
    )


@router.post("/analysis/{analysis_id}/preview", response_model=AnalysisPreviewResponse)
async def preview_analysis(analysis_id: str, payload: AnalysisPreviewRequest) -> AnalysisPreviewResponse:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    artifact_path = _artifact_path(analysis)
    if analysis["algorithm"] == POSE_ALGORITHM:
        raise HTTPException(400, "pose analyses do not support instant segment preview yet")
    metadata = load_artifact(artifact_path)["metadata"]

    requested = payload.knobs.model_dump()
    requested = _lock_reanalysis_only(requested, metadata)
    result = recompute_from_artifact(artifact_path, requested, include_samples=True)

    previous_rows = await get_segments_for_analysis(analysis_id)
    previous = [(round(float(r["start_s"]), 2), round(float(r["end_s"]), 2), bool(r["is_on"])) for r in previous_rows]
    current = [
        (round(float(s["start_s"]), 2), round(float(s["end_s"]), 2), bool(s["is_on"]))
        for s in result["segments"]
    ]
    changed_count = sum(1 for a, b in zip(previous, current) if a != b) + abs(len(previous) - len(current))
    result["summary"]["changed_segment_count"] = int(changed_count)

    await replace_segments(analysis["video_id"], result["segments"], analysis_id=analysis_id)
    await update_analysis_knobs(analysis_id, json.dumps(result["knobs"]))

    rows = await get_segments_for_analysis(analysis_id)
    return AnalysisPreviewResponse(
        video_id=analysis["video_id"],
        analysis_id=analysis_id,
        duration_s=float(analysis["duration_s"]),
        knobs=_knobs_model(result["knobs"]),
        segments=[segment_from_row(r) for r in rows],
        summary=result["summary"],
    )


@router.post("/analysis/{analysis_id}/audio-preview", response_model=AudioPreviewResponse)
async def preview_audio(analysis_id: str, payload: AudioPreviewRequest) -> AudioPreviewResponse:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    if analysis["algorithm"] not in {POSE_ALGORITHM, HIT_STUDY_ALGORITHM}:
        raise HTTPException(400, "audio preview is only available for pose-based analyses")
    artifact_path = _artifact_path(analysis)
    artifact = load_pose_artifact(artifact_path) if analysis["algorithm"] == POSE_ALGORITHM else load_hit_study_artifact(artifact_path)

    start_s = max(0.0, float(payload.range_start_s))
    end_s = min(float(analysis["duration_s"]), float(payload.range_end_s))
    if end_s <= start_s:
        raise HTTPException(422, "range_end_s must be greater than range_start_s")

    requested = payload.knobs.model_dump()
    knobs = normalize_knobs({**default_rally_knobs(), **requested})
    result = preview_audio_range(
        Path(analysis["filepath"]),
        start_s,
        end_s,
        artifact.get("frames") or [],
        knobs,
    )
    return AudioPreviewResponse(
        video_id=analysis["video_id"],
        analysis_id=analysis_id,
        duration_s=float(analysis["duration_s"]),
        range_start_s=start_s,
        range_end_s=end_s,
        knobs=_knobs_model(knobs),
        impacts=result["validated_impacts"],
        summary=result["summary"],
    )
