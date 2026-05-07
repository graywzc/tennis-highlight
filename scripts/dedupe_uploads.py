import hashlib
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "app.db"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cols = {r[1] for r in conn.execute("PRAGMA table_info(videos)").fetchall()}
        if "content_hash" not in cols:
            conn.execute("ALTER TABLE videos ADD COLUMN content_hash TEXT")
        if "size_bytes" not in cols:
            conn.execute("ALTER TABLE videos ADD COLUMN size_bytes INTEGER")
        if "analysis_id" not in cols:
            conn.execute("ALTER TABLE videos ADD COLUMN analysis_id TEXT")
        conn.commit()

        rows = conn.execute(
            "SELECT id, filepath FROM videos WHERE filepath IS NOT NULL ORDER BY created_at ASC"
        ).fetchall()

        by_hash: dict[str, str] = {}
        delete_candidates: list[Path] = []

        for row in rows:
            path = Path(row["filepath"])
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                print(f"skip missing: {row['id']} {path}")
                continue

            content_hash = sha256_file(path)
            size_bytes = path.stat().st_size
            canonical = by_hash.get(content_hash)

            if canonical is None:
                by_hash[content_hash] = str(path.relative_to(ROOT))
                conn.execute(
                    "UPDATE videos SET filepath=?, content_hash=?, size_bytes=? WHERE id=?",
                    (by_hash[content_hash], content_hash, size_bytes, row["id"]),
                )
                print(f"canonical {content_hash[:12]} {row['id']} {path.name}")
            else:
                conn.execute(
                    """
                    UPDATE videos
                    SET filepath=?, content_hash=?, size_bytes=?
                    WHERE id=?
                    """,
                    (canonical, content_hash, size_bytes, row["id"]),
                )
                if str(path) != str(ROOT / canonical) and str(path) != canonical:
                    delete_candidates.append(path)
                print(f"dedupe {row['id']} -> {canonical} {path.name}")

        conn.commit()

    for path in sorted(set(delete_candidates)):
        try:
            path.unlink()
            print(f"deleted duplicate file: {path}")
        except OSError as e:
            print(f"could not delete {path}: {e}")


if __name__ == "__main__":
    main()
