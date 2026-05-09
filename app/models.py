from typing import Literal

from pydantic import BaseModel, model_validator


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    duration_s: float


class StatusResponse(BaseModel):
    video_id: str
    status: str
    error_msg: str | None = None
    duration_s: float | None = None
    progress_percent: float | None = None
    progress_message: str | None = None


class VideoSummary(BaseModel):
    video_id: str
    filename: str
    status: str
    error_msg: str | None = None
    duration_s: float | None = None
    segment_count: int = 0
    on_segment_count: int = 0
    created_at: float
    progress_percent: float | None = None
    progress_message: str | None = None


class MediaSummary(BaseModel):
    video_id: str
    filename: str
    filepath: str
    duration_s: float | None = None
    content_hash: str | None = None
    size_bytes: int | None = None
    created_at: float
    analysis_count: int = 0
    has_court_calibration: bool = False
    court_calibration: dict | None = None


class AnalysisRunSummary(BaseModel):
    analysis_id: str
    video_id: str
    filename: str
    algorithm: str
    status: str
    error_msg: str | None = None
    duration_s: float | None = None
    segment_count: int = 0
    on_segment_count: int = 0
    progress_percent: float | None = None
    progress_message: str | None = None
    progress_eta_s: float | None = None
    range_start_s: float | None = None
    range_end_s: float | None = None
    noninstant_knobs: dict | None = None
    created_at: float


class Segment(BaseModel):
    start_s: float
    end_s: float
    is_on: bool = True
    source: str = "manual"
    raw_start_s: float | None = None
    raw_end_s: float | None = None
    avg_score: float | None = None
    max_score: float | None = None
    min_score: float | None = None
    sample_count: int | None = None
    decision_stage: str | None = None
    samples: list[dict] | None = None


class SegmentsResponse(BaseModel):
    video_id: str
    duration_s: float
    segments: list[Segment]


class ExportRequest(BaseModel):
    segments: list[Segment]


class ExportResponse(BaseModel):
    export_id: str
    status: str


class ExportStatusResponse(BaseModel):
    export_id: str
    status: str
    error_msg: str | None = None
    duration_s: float | None = None


class DetectorKnobs(BaseModel):
    diff_threshold: int
    motion_threshold: float
    merge_gap_s: float
    min_segment_s: float
    segment_padding_s: float
    sample_fps: float
    median_bg_samples: int
    enable_merge_gap: bool = True
    enable_min_segment: bool = True
    enable_padding: bool = True
    court_weight: float = 1.0
    outside_weight: float = 0.15
    near_camera_weight: float = 0.0
    # Rally detection (used by pose_skeleton_yolo)
    audio_sample_rate: int = 22050
    bandpass_low_hz: float = 1000.0
    bandpass_high_hz: float = 8000.0
    peak_height_mad_k: float = 6.0
    peak_prominence_mult: float = 2.0
    min_impact_separation_s: float = 0.15
    min_spectral_centroid_hz: float = 2500.0
    pose_window_s: float = 0.75
    wrist_conf_min: float = 0.3
    min_wrist_velocity: float = 0.4
    max_gap_s: float = 5.0
    min_hits_per_rally: int = 2
    rally_padding_s: float = 1.0


class AnalysisMetadata(BaseModel):
    analysis_id: str | None = None
    detector: str
    detector_version: int
    duration_s: float
    target_width: int
    target_height: int
    sample_fps: float
    median_bg_samples: int
    sample_count: int


class AnalysisResponse(BaseModel):
    video_id: str
    analysis_id: str | None = None
    duration_s: float
    defaults: DetectorKnobs
    knobs: DetectorKnobs
    metadata: AnalysisMetadata | None = None
    summary: dict


class AnalysisPreviewRequest(BaseModel):
    knobs: DetectorKnobs


class AnalysisPreviewResponse(BaseModel):
    video_id: str
    analysis_id: str | None = None
    duration_s: float
    knobs: DetectorKnobs
    segments: list[Segment]
    summary: dict


class AudioPreviewRequest(BaseModel):
    range_start_s: float
    range_end_s: float
    knobs: DetectorKnobs


class AudioPreviewResponse(BaseModel):
    video_id: str
    analysis_id: str
    duration_s: float
    range_start_s: float
    range_end_s: float
    knobs: DetectorKnobs
    impacts: list[dict]
    summary: dict


class StrikeLabelRequest(BaseModel):
    time_s: float
    source: Literal["candidate", "manual", "near_player_hit"]
    is_strike: bool
    algorithm_validated: bool | None = None
    comment: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "StrikeLabelRequest":
        if self.source == "candidate" and self.algorithm_validated is None:
            raise ValueError("algorithm_validated is required when source='candidate'")
        return self


class StrikeLabel(BaseModel):
    id: str
    analysis_id: str
    time_s: float
    source: Literal["candidate", "manual", "near_player_hit"]
    is_strike: bool
    algorithm_validated: bool | None
    comment: str | None
    created_at: float
    updated_at: float


class StrikeLabelsResponse(BaseModel):
    analysis_id: str
    labels: list[StrikeLabel]
