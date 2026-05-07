from pydantic import BaseModel


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
