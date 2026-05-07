from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    upload_dir: Path = Path("data/uploads")
    exports_dir: Path = Path("data/exports")
    segments_tmp_dir: Path = Path("data/segments_tmp")
    analysis_dir: Path = Path("data/analysis")
    db_path: Path = Path("data/app.db")

    motion_threshold: float = 0.02
    merge_gap_s: float = 3.0
    min_segment_s: float = 5.0
    segment_padding_s: float = 1.0
    sample_fps: float = 2.0

    # Median-background detector
    diff_threshold: int = 25            # 0-255; per-pixel intensity diff to count as foreground
    median_bg_samples: int = 80         # frames spread across the video to compute median background

    verbose: bool = False
    progress_report_interval_s: float = 5.0


settings = Settings()
