import logging
from pathlib import Path

from app.config import settings
from app.pipeline.ffmpeg_utils import run

logger = logging.getLogger(__name__)


async def export_segments(
    video_path: Path,
    segments: list[tuple[float, float]],
    output_path: Path,
    job_id: str,
) -> None:
    """Extract each (start, end) segment from video_path and stitch into output_path."""
    if not segments:
        raise ValueError("No segments to export")

    work_dir = settings.segments_tmp_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    seg_paths: list[Path] = []
    try:
        for i, (start, end) in enumerate(segments):
            seg_path = work_dir / f"seg_{i:04d}.mp4"
            duration = max(0.0, end - start)
            logger.info(
                "export seg %d/%d: %.2fs - %.2fs (%.2fs)",
                i + 1, len(segments), start, end, duration,
            )
            await run([
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                str(seg_path),
            ])
            seg_paths.append(seg_path)

        concat_list = work_dir / "concat.txt"
        # Each line must be: file '/abs/path'
        concat_list.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in seg_paths) + "\n",
            encoding="utf-8",
        )

        logger.info("export concat: %d segments → %s", len(seg_paths), output_path.name)
        await run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ])
    finally:
        for p in seg_paths:
            p.unlink(missing_ok=True)
        concat_list = work_dir / "concat.txt"
        concat_list.unlink(missing_ok=True)
        try:
            work_dir.rmdir()
        except OSError:
            pass
