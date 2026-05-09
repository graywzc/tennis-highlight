import asyncio
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app import database
from app.config import settings
from app.models import StrikeLabelRequest


_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


class _TempDb:
    """Repoint settings.db_path at a fresh temp file and init the schema."""

    def __enter__(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._original = settings.db_path
        settings.db_path = self._tmp.name
        _run(database.init_db())
        # Seed an analyses row so foreign-key references in strike_labels
        # don't dangle. The schema sets PRAGMA foreign_keys = ON inside
        # _connect(), so we need a real parent row.
        with database._connect() as conn:
            conn.execute(
                "INSERT INTO videos (id, filename, filepath, status, created_at, updated_at) "
                "VALUES ('vid1', 'fake.mov', '/tmp/fake.mov', 'done', 0, 0)"
            )
            conn.execute(
                "INSERT INTO analyses (id, video_id, algorithm, status, created_at, updated_at) "
                "VALUES ('an1', 'vid1', 'pose_skeleton_yolo', 'done', 0, 0)"
            )
            conn.commit()
        return self

    def __exit__(self, *exc):
        settings.db_path = self._original
        try:
            Path(self._tmp.name).unlink(missing_ok=True)
        except Exception:
            pass


class StrikeLabelModelTests(unittest.TestCase):
    def test_candidate_requires_algorithm_validated(self):
        with self.assertRaises(ValidationError):
            StrikeLabelRequest(time_s=1.0, source="candidate", is_strike=True)

    def test_manual_does_not_require_algorithm_validated(self):
        m = StrikeLabelRequest(time_s=1.0, source="manual", is_strike=True)
        self.assertEqual(m.source, "manual")
        self.assertIsNone(m.algorithm_validated)

    def test_candidate_with_algorithm_validated_ok(self):
        m = StrikeLabelRequest(
            time_s=1.5, source="candidate", is_strike=False, algorithm_validated=True
        )
        self.assertEqual(m.algorithm_validated, True)


class StrikeLabelDbTests(unittest.TestCase):
    def test_upsert_creates_then_updates(self):
        with _TempDb():
            row1 = _run(database.upsert_strike_label(
                "an1", time_s=12.345, source="candidate",
                is_strike=True, algorithm_validated=True, comment="first",
            ))
            row2 = _run(database.upsert_strike_label(
                "an1", time_s=12.345, source="candidate",
                is_strike=False, algorithm_validated=True, comment="second",
            ))
            self.assertEqual(row1["id"], row2["id"])
            labels = _run(database.list_strike_labels("an1"))
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["comment"], "second")
            self.assertEqual(labels[0]["is_strike"], 0)

    def test_list_returns_sorted_by_time(self):
        with _TempDb():
            _run(database.upsert_strike_label(
                "an1", time_s=30.0, source="candidate",
                is_strike=True, algorithm_validated=True, comment=None,
            ))
            _run(database.upsert_strike_label(
                "an1", time_s=5.0, source="candidate",
                is_strike=False, algorithm_validated=False, comment=None,
            ))
            _run(database.upsert_strike_label(
                "an1", time_s=15.0, source="manual",
                is_strike=True, algorithm_validated=None, comment="missed",
            ))
            labels = _run(database.list_strike_labels("an1"))
            times = [float(r["time_s"]) for r in labels]
            self.assertEqual(times, sorted(times))

    def test_candidate_and_manual_at_same_time_are_separate_rows(self):
        with _TempDb():
            _run(database.upsert_strike_label(
                "an1", time_s=10.0, source="candidate",
                is_strike=False, algorithm_validated=True, comment=None,
            ))
            _run(database.upsert_strike_label(
                "an1", time_s=10.0, source="manual",
                is_strike=True, algorithm_validated=None, comment=None,
            ))
            labels = _run(database.list_strike_labels("an1"))
            self.assertEqual(len(labels), 2)
            sources = {r["source"] for r in labels}
            self.assertEqual(sources, {"candidate", "manual"})

    def test_delete_label(self):
        with _TempDb():
            row = _run(database.upsert_strike_label(
                "an1", time_s=4.0, source="manual",
                is_strike=True, algorithm_validated=None, comment="x",
            ))
            ok = _run(database.delete_strike_label(row["id"]))
            self.assertTrue(ok)
            self.assertEqual(_run(database.list_strike_labels("an1")), [])
            self.assertFalse(_run(database.delete_strike_label(row["id"])))

    def test_delete_analysis_cascades_labels(self):
        with _TempDb():
            _run(database.upsert_strike_label(
                "an1", time_s=2.0, source="candidate",
                is_strike=True, algorithm_validated=True, comment=None,
            ))
            _run(database.delete_analysis_run("an1"))
            self.assertEqual(_run(database.list_strike_labels("an1")), [])

    def test_time_s_rounding_dedupes_close_floats(self):
        with _TempDb():
            r1 = _run(database.upsert_strike_label(
                "an1", time_s=12.3456789, source="candidate",
                is_strike=True, algorithm_validated=True, comment="a",
            ))
            r2 = _run(database.upsert_strike_label(
                "an1", time_s=12.3461,  # within rounding threshold of 1ms? -> 12.346 vs 12.346
                source="candidate", is_strike=False, algorithm_validated=True, comment="b",
            ))
            # 12.3456789 → 12.346, 12.3461 → 12.346, so they collide.
            self.assertEqual(r1["id"], r2["id"])
            self.assertEqual(len(_run(database.list_strike_labels("an1"))), 1)


if __name__ == "__main__":
    unittest.main()
