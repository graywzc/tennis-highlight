import asyncio
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id               TEXT PRIMARY KEY,
    filename         TEXT NOT NULL,
    filepath         TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    error_msg        TEXT,
    duration_s       REAL,
    progress_percent REAL DEFAULT 0,
    progress_message TEXT,
    content_hash     TEXT,
    size_bytes       INTEGER,
    analysis_id      TEXT,
    court_calibration_json TEXT,
    analysis_artifact_path TEXT,
    analysis_knobs_json TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    analysis_id TEXT,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    is_on       INTEGER NOT NULL DEFAULT 1,
    source      TEXT NOT NULL DEFAULT 'manual',
    raw_start_s REAL,
    raw_end_s   REAL,
    avg_score   REAL,
    max_score   REAL,
    min_score   REAL,
    sample_count INTEGER,
    decision_stage TEXT,
    samples_json TEXT,
    sort_order  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS exports (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id),
    status          TEXT NOT NULL DEFAULT 'pending',
    error_msg       TEXT,
    output_filepath TEXT,
    duration_s      REAL,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id),
    algorithm       TEXT NOT NULL DEFAULT 'median_frame',
    status          TEXT NOT NULL DEFAULT 'pending',
    error_msg       TEXT,
    artifact_path   TEXT,
    noninstant_knobs_json TEXT,
    instant_knobs_json TEXT,
    progress_percent REAL DEFAULT 0,
    progress_message TEXT,
    progress_eta_s REAL,
    range_start_s  REAL DEFAULT 0,
    range_end_s    REAL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_exports_video ON exports(video_id);
CREATE INDEX IF NOT EXISTS idx_analyses_video ON analyses(video_id, created_at);
"""


@contextmanager
def _connect():
    """Open a SQLite connection and guarantee it gets closed.

    Note: sqlite3's own `with conn:` syntax commits/rolls back but does NOT
    close the connection — using that alone leaks file descriptors.
    """
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


async def init_db() -> None:
    def _init():
        Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
        with _connect() as conn:
            conn.executescript(SCHEMA)
            # Idempotent migrations for existing DBs.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(videos)").fetchall()}
            if "progress_percent" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN progress_percent REAL DEFAULT 0")
            if "progress_message" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN progress_message TEXT")
            if "analysis_artifact_path" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN analysis_artifact_path TEXT")
            if "analysis_knobs_json" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN analysis_knobs_json TEXT")
            if "content_hash" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN content_hash TEXT")
            if "size_bytes" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN size_bytes INTEGER")
            if "analysis_id" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN analysis_id TEXT")
            if "court_calibration_json" not in cols:
                conn.execute("ALTER TABLE videos ADD COLUMN court_calibration_json TEXT")
            seg_cols = {r[1] for r in conn.execute("PRAGMA table_info(segments)").fetchall()}
            segment_migrations = {
                "analysis_id": "ALTER TABLE segments ADD COLUMN analysis_id TEXT",
                "source": "ALTER TABLE segments ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'",
                "raw_start_s": "ALTER TABLE segments ADD COLUMN raw_start_s REAL",
                "raw_end_s": "ALTER TABLE segments ADD COLUMN raw_end_s REAL",
                "avg_score": "ALTER TABLE segments ADD COLUMN avg_score REAL",
                "max_score": "ALTER TABLE segments ADD COLUMN max_score REAL",
                "min_score": "ALTER TABLE segments ADD COLUMN min_score REAL",
                "sample_count": "ALTER TABLE segments ADD COLUMN sample_count INTEGER",
                "decision_stage": "ALTER TABLE segments ADD COLUMN decision_stage TEXT",
                "samples_json": "ALTER TABLE segments ADD COLUMN samples_json TEXT",
            }
            for col, sql in segment_migrations.items():
                if col not in seg_cols:
                    conn.execute(sql)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id              TEXT PRIMARY KEY,
                    video_id        TEXT NOT NULL REFERENCES videos(id),
                    algorithm       TEXT NOT NULL DEFAULT 'median_frame',
                    status          TEXT NOT NULL DEFAULT 'pending',
                    error_msg       TEXT,
                    artifact_path   TEXT,
                    noninstant_knobs_json TEXT,
                    instant_knobs_json TEXT,
                    progress_percent REAL DEFAULT 0,
                    progress_message TEXT,
                    progress_eta_s REAL,
                    range_start_s  REAL DEFAULT 0,
                    range_end_s    REAL,
                    created_at      REAL NOT NULL,
                    updated_at      REAL NOT NULL
                )
                """
            )
            analysis_cols = {r[1] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()}
            analysis_migrations = {
                "algorithm": "ALTER TABLE analyses ADD COLUMN algorithm TEXT NOT NULL DEFAULT 'median_frame'",
                "error_msg": "ALTER TABLE analyses ADD COLUMN error_msg TEXT",
                "artifact_path": "ALTER TABLE analyses ADD COLUMN artifact_path TEXT",
                "noninstant_knobs_json": "ALTER TABLE analyses ADD COLUMN noninstant_knobs_json TEXT",
                "instant_knobs_json": "ALTER TABLE analyses ADD COLUMN instant_knobs_json TEXT",
                "progress_percent": "ALTER TABLE analyses ADD COLUMN progress_percent REAL DEFAULT 0",
                "progress_message": "ALTER TABLE analyses ADD COLUMN progress_message TEXT",
                "progress_eta_s": "ALTER TABLE analyses ADD COLUMN progress_eta_s REAL",
                "range_start_s": "ALTER TABLE analyses ADD COLUMN range_start_s REAL DEFAULT 0",
                "range_end_s": "ALTER TABLE analyses ADD COLUMN range_end_s REAL",
                "updated_at": "ALTER TABLE analyses ADD COLUMN updated_at REAL",
            }
            for col, sql in analysis_migrations.items():
                if col not in analysis_cols:
                    conn.execute(sql)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_analysis ON segments(analysis_id, sort_order)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_video ON analyses(video_id, created_at)")
            _migrate_existing_video_segments_to_analyses(conn)
            conn.execute(
                "UPDATE videos SET status='pending' WHERE status IN ('analyzing')"
            )
            conn.execute(
                "UPDATE exports SET status='error', error_msg='Server restarted' "
                "WHERE status IN ('pending', 'exporting')"
            )
            conn.commit()
    await asyncio.to_thread(_init)


def _now() -> float:
    return time.time()


def _migrate_existing_video_segments_to_analyses(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT v.id, v.analysis_id, v.analysis_artifact_path, v.analysis_knobs_json,
               v.status, v.created_at, v.updated_at,
               COUNT(s.id) AS segment_count
        FROM videos v
        JOIN segments s ON s.video_id = v.id
        WHERE s.analysis_id IS NULL
        GROUP BY v.id
        """
    ).fetchall()
    for row in rows:
        analysis_id = row["analysis_id"] or uuid4().hex
        exists = conn.execute("SELECT id FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        if exists is None:
            status = "done" if row["status"] == "done" else "pending"
            conn.execute(
                """
                INSERT INTO analyses (
                    id, video_id, algorithm, status, artifact_path,
                    instant_knobs_json, progress_percent, progress_message,
                    created_at, updated_at
                )
                VALUES (?, ?, 'median_frame', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    row["id"],
                    status,
                    row["analysis_artifact_path"],
                    row["analysis_knobs_json"],
                    100.0 if status == "done" else 0.0,
                    "migrated" if status == "done" else None,
                    row["created_at"],
                    row["updated_at"] or row["created_at"],
                ),
            )
        conn.execute(
            "UPDATE segments SET analysis_id=? WHERE video_id=? AND analysis_id IS NULL",
            (analysis_id, row["id"]),
        )
        conn.execute(
            "UPDATE videos SET analysis_id=? WHERE id=? AND analysis_id IS NULL",
            (analysis_id, row["id"]),
        )


# ---- Videos --------------------------------------------------------------

async def create_video(
    filename: str,
    filepath: str,
    duration_s: float,
    *,
    content_hash: str | None = None,
    size_bytes: int | None = None,
) -> str:
    video_id = uuid4().hex

    def _q():
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO videos (
                    id, filename, filepath, status, duration_s,
                    content_hash, size_bytes, created_at, updated_at
                )
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    filename,
                    filepath,
                    duration_s,
                    content_hash,
                    size_bytes,
                    _now(),
                    _now(),
                ),
            )
            conn.commit()
    await asyncio.to_thread(_q)
    return video_id


async def get_video(video_id: str):
    def _q():
        with _connect() as conn:
            return conn.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
    return await asyncio.to_thread(_q)


async def get_video_by_filename(filename: str):
    def _q():
        with _connect() as conn:
            return conn.execute(
                "SELECT * FROM videos WHERE filename = ?", (filename,)
            ).fetchone()
    return await asyncio.to_thread(_q)


async def get_video_by_content_hash(content_hash: str):
    def _q():
        with _connect() as conn:
            return conn.execute(
                """
                SELECT * FROM videos
                WHERE content_hash = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
    return await asyncio.to_thread(_q)


async def list_videos() -> list[sqlite3.Row]:
    """All videos with attached segment counts, newest first."""
    def _q():
        with _connect() as conn:
            return conn.execute(
                """
                SELECT
                    v.*,
                    (SELECT COUNT(*) FROM segments WHERE video_id = v.id) AS segment_count,
                    (SELECT COUNT(*) FROM segments WHERE video_id = v.id AND is_on = 1) AS on_segment_count
                FROM videos v
                ORDER BY v.created_at DESC
                """
            ).fetchall()
    return await asyncio.to_thread(_q)


async def list_media() -> list[sqlite3.Row]:
    """Distinct uploaded media assets, newest representative first."""
    def _q():
        with _connect() as conn:
            return conn.execute(
                """
                SELECT
                    MIN(v.id) AS id,
                    v.filename,
                    v.filepath,
                    v.content_hash,
                    v.size_bytes,
                    v.court_calibration_json,
                    MAX(v.duration_s) AS duration_s,
                    MIN(v.created_at) AS created_at,
                    COUNT(*) AS row_count,
                    (
                        SELECT COUNT(*) FROM analyses a
                        JOIN videos vv ON vv.id = a.video_id
                        WHERE vv.filepath = v.filepath
                    ) AS analysis_count
                FROM videos v
                GROUP BY COALESCE(v.content_hash, v.filepath), v.filepath
                ORDER BY MIN(v.created_at) DESC
                """
            ).fetchall()
    return await asyncio.to_thread(_q)


async def update_video_duration(video_id: str, duration_s: float) -> None:
    def _q():
        with _connect() as conn:
            conn.execute(
                "UPDATE videos SET duration_s=?, updated_at=? WHERE id=?",
                (duration_s, _now(), video_id),
            )
            conn.commit()
    await asyncio.to_thread(_q)


async def update_video_asset_info(
    video_id: str,
    *,
    filepath: str,
    duration_s: float,
    content_hash: str,
    size_bytes: int,
) -> None:
    def _q():
        with _connect() as conn:
            conn.execute(
                """
                UPDATE videos
                SET filepath=?, duration_s=?, content_hash=?, size_bytes=?, updated_at=?
                WHERE id=?
                """,
                (filepath, duration_s, content_hash, size_bytes, _now(), video_id),
            )
            conn.commit()
    await asyncio.to_thread(_q)


async def update_video_court_calibration(video_id: str, calibration_json: str | None) -> None:
    def _q():
        with _connect() as conn:
            conn.execute(
                "UPDATE videos SET court_calibration_json=?, updated_at=? WHERE id=?",
                (calibration_json, _now(), video_id),
            )
            conn.commit()
    await asyncio.to_thread(_q)


async def delete_video_cascade(video_id: str) -> dict:
    """Delete a video row plus its segments/exports. Returns paths to clean from disk."""
    def _q():
        with _connect() as conn:
            row = conn.execute(
                "SELECT filepath FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
            if row is None:
                return {"upload": None, "exports": []}
            upload = row["filepath"]
            export_rows = conn.execute(
                "SELECT output_filepath FROM exports "
                "WHERE video_id = ? AND output_filepath IS NOT NULL",
                (video_id,),
            ).fetchall()
            exports = [r["output_filepath"] for r in export_rows]
            conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM exports WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            conn.commit()
            shared = conn.execute(
                "SELECT COUNT(*) AS n FROM videos WHERE filepath = ?",
                (upload,),
            ).fetchone()
            return {
                "upload": upload,
                "delete_upload": (shared["n"] == 0),
                "exports": exports,
            }
    return await asyncio.to_thread(_q)


async def update_video_status(video_id: str, status: str, error_msg: str | None = None) -> None:
    def _q():
        with _connect() as conn:
            conn.execute(
                "UPDATE videos SET status=?, error_msg=?, updated_at=? WHERE id=?",
                (status, error_msg, _now(), video_id),
            )
            conn.commit()
    await asyncio.to_thread(_q)


async def update_video_analysis(
    video_id: str,
    analysis_id: str | None = None,
    artifact_path: str | None = None,
    knobs_json: str | None = None,
) -> None:
    def _q():
        fields = ["updated_at=?"]
        values: list = [_now()]
        if analysis_id is not None:
            fields.append("analysis_id=?")
            values.append(analysis_id)
        if artifact_path is not None:
            fields.append("analysis_artifact_path=?")
            values.append(artifact_path)
        if knobs_json is not None:
            fields.append("analysis_knobs_json=?")
            values.append(knobs_json)
        values.append(video_id)
        with _connect() as conn:
            conn.execute(
                f"UPDATE videos SET {', '.join(fields)} WHERE id=?",
                values,
            )
            conn.commit()
    await asyncio.to_thread(_q)


# ---- Analyses ------------------------------------------------------------

async def create_analysis(
    video_id: str,
    *,
    algorithm: str,
    noninstant_knobs_json: str,
    instant_knobs_json: str,
    range_start_s: float,
    range_end_s: float,
) -> str:
    analysis_id = uuid4().hex

    def _q():
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    id, video_id, algorithm, status, noninstant_knobs_json,
                    instant_knobs_json, progress_percent, range_start_s, range_end_s,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, 'pending', ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    video_id,
                    algorithm,
                    noninstant_knobs_json,
                    instant_knobs_json,
                    float(range_start_s),
                    float(range_end_s),
                    _now(),
                    _now(),
                ),
            )
            conn.commit()
    await asyncio.to_thread(_q)
    return analysis_id


async def get_analysis_run(analysis_id: str):
    def _q():
        with _connect() as conn:
            return conn.execute(
                """
                SELECT a.*, v.filename, v.filepath, v.duration_s
                FROM analyses a
                JOIN videos v ON v.id = a.video_id
                WHERE a.id=?
                """,
                (analysis_id,),
            ).fetchone()
    return await asyncio.to_thread(_q)


async def list_analysis_runs() -> list[sqlite3.Row]:
    def _q():
        with _connect() as conn:
            return conn.execute(
                """
                SELECT
                    a.*, v.filename, v.filepath, v.duration_s,
                    (SELECT COUNT(*) FROM segments s WHERE s.analysis_id = a.id) AS segment_count,
                    (SELECT COUNT(*) FROM segments s WHERE s.analysis_id = a.id AND s.is_on = 1) AS on_segment_count
                FROM analyses a
                JOIN videos v ON v.id = a.video_id
                ORDER BY a.created_at DESC
                """
            ).fetchall()
    return await asyncio.to_thread(_q)


async def update_analysis_status(
    analysis_id: str,
    status: str,
    error_msg: str | None = None,
) -> None:
    def _q():
        with _connect() as conn:
            conn.execute(
                "UPDATE analyses SET status=?, error_msg=?, updated_at=? WHERE id=?",
                (status, error_msg, _now(), analysis_id),
            )
            conn.commit()
    await asyncio.to_thread(_q)


def update_analysis_progress_sync(
    analysis_id: str,
    percent: float,
    message: str,
    eta_s: float | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE analyses
            SET progress_percent=?, progress_message=?, progress_eta_s=?, updated_at=?
            WHERE id=?
            """,
            (percent, message, eta_s, _now(), analysis_id),
        )
        conn.commit()


async def complete_analysis(
    analysis_id: str,
    *,
    artifact_path: str,
    instant_knobs_json: str,
) -> None:
    def _q():
        with _connect() as conn:
            conn.execute(
                """
                UPDATE analyses
                SET status='done', artifact_path=?, instant_knobs_json=?,
                    progress_percent=100, progress_message='done', progress_eta_s=NULL, updated_at=?
                WHERE id=?
                """,
                (artifact_path, instant_knobs_json, _now(), analysis_id),
            )
            conn.commit()
    await asyncio.to_thread(_q)


async def update_analysis_knobs(analysis_id: str, instant_knobs_json: str) -> None:
    def _q():
        with _connect() as conn:
            conn.execute(
                "UPDATE analyses SET instant_knobs_json=?, updated_at=? WHERE id=?",
                (instant_knobs_json, _now(), analysis_id),
            )
            conn.commit()
    await asyncio.to_thread(_q)


async def delete_analysis_run(analysis_id: str) -> dict | None:
    def _q():
        with _connect() as conn:
            row = conn.execute(
                "SELECT artifact_path FROM analyses WHERE id=?",
                (analysis_id,),
            ).fetchone()
            if row is None:
                return None
            artifact_path = row["artifact_path"]
            conn.execute("DELETE FROM segments WHERE analysis_id=?", (analysis_id,))
            conn.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
            conn.commit()
            return {"artifact_path": artifact_path}
    return await asyncio.to_thread(_q)


def update_video_progress_sync(video_id: str, percent: float, message: str) -> None:
    """Synchronous progress update (callable from worker threads inside asyncio.to_thread)."""
    with _connect() as conn:
        conn.execute(
            "UPDATE videos SET progress_percent=?, progress_message=?, updated_at=? WHERE id=?",
            (percent, message, _now(), video_id),
        )
        conn.commit()


# ---- Segments ------------------------------------------------------------

async def replace_segments(
    video_id: str,
    segments: list[dict],
    analysis_id: str | None = None,
) -> None:
    """segments: list of {start_s, end_s, is_on}. Replaces all segments for video."""
    def _q():
        with _connect() as conn:
            if analysis_id is None:
                conn.execute("DELETE FROM segments WHERE video_id=? AND analysis_id IS NULL", (video_id,))
            else:
                conn.execute("DELETE FROM segments WHERE analysis_id=?", (analysis_id,))
            for i, seg in enumerate(segments):
                conn.execute(
                    """
                    INSERT INTO segments (
                        id, video_id, start_s, end_s, is_on, source,
                        analysis_id,
                        raw_start_s, raw_end_s, avg_score, max_score, min_score,
                        sample_count, decision_stage, samples_json, sort_order
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid4().hex,
                        video_id,
                        float(seg["start_s"]),
                        float(seg["end_s"]),
                        1 if seg.get("is_on", True) else 0,
                        seg.get("source", "manual"),
                        analysis_id,
                        seg.get("raw_start_s"),
                        seg.get("raw_end_s"),
                        seg.get("avg_score"),
                        seg.get("max_score"),
                        seg.get("min_score"),
                        seg.get("sample_count"),
                        seg.get("decision_stage"),
                        seg.get("samples_json") if "samples_json" in seg else json.dumps(seg.get("samples", [])),
                        i,
                    ),
                )
            conn.commit()
    await asyncio.to_thread(_q)


async def get_segments(video_id: str) -> list[sqlite3.Row]:
    def _q():
        with _connect() as conn:
            return conn.execute(
                "SELECT * FROM segments WHERE video_id=? ORDER BY sort_order ASC",
                (video_id,),
            ).fetchall()
    return await asyncio.to_thread(_q)


async def get_segments_for_analysis(analysis_id: str) -> list[sqlite3.Row]:
    def _q():
        with _connect() as conn:
            return conn.execute(
                "SELECT * FROM segments WHERE analysis_id=? ORDER BY sort_order ASC",
                (analysis_id,),
            ).fetchall()
    return await asyncio.to_thread(_q)


# ---- Exports -------------------------------------------------------------

async def create_export(video_id: str) -> str:
    export_id = uuid4().hex

    def _q():
        with _connect() as conn:
            conn.execute(
                "INSERT INTO exports (id, video_id, status, created_at) VALUES (?, ?, 'pending', ?)",
                (export_id, video_id, _now()),
            )
            conn.commit()
    await asyncio.to_thread(_q)
    return export_id


async def get_export(export_id: str):
    def _q():
        with _connect() as conn:
            return conn.execute(
                "SELECT * FROM exports WHERE id=?", (export_id,)
            ).fetchone()
    return await asyncio.to_thread(_q)


async def update_export(
    export_id: str,
    *,
    status: str | None = None,
    error_msg: str | None = None,
    output_filepath: str | None = None,
    duration_s: float | None = None,
) -> None:
    fields = []
    values: list = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if error_msg is not None:
        fields.append("error_msg=?")
        values.append(error_msg)
    if output_filepath is not None:
        fields.append("output_filepath=?")
        values.append(output_filepath)
    if duration_s is not None:
        fields.append("duration_s=?")
        values.append(duration_s)
    if not fields:
        return
    values.append(export_id)

    def _q():
        with _connect() as conn:
            conn.execute(
                f"UPDATE exports SET {', '.join(fields)} WHERE id=?",
                values,
            )
            conn.commit()
    await asyncio.to_thread(_q)
