import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.routers.modular_scan import POSE_SCAN_JOBS, AUDIO_SCAN_JOBS
import os
import tempfile
from pathlib import Path
from app.config import settings
from app import database
import asyncio

_loop = asyncio.new_event_loop()

def _run(coro):
    return _loop.run_until_complete(coro)

class TestModularScanRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self._tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp_db.close()
        self._original_db = settings.db_path
        settings.db_path = self._tmp_db.name
        _run(database.init_db())

        # Create a fake video and analysis
        with database._connect() as conn:
            conn.execute(
                "INSERT INTO videos (id, filename, filepath, status, duration_s, created_at, updated_at) "
                "VALUES ('vid1', 'fake.mp4', 'tests/fixtures/fake.mp4', 'done', 10.0, 0, 0)"
            )
            conn.execute(
                "INSERT INTO analyses (id, video_id, algorithm, status, created_at, updated_at) "
                "VALUES ('an1', 'vid1', 'near_player_hit_study', 'done', 0, 0)"
            )
            conn.commit()
        
        # Ensure the fake video path exists (at least as a directory or something if we don't want to create a real file)
        # Actually, for start_pose_scan, it checks if video_path.exists()
        self.fake_video = Path("tests/fixtures/fake.mp4")
        self.fake_video.parent.mkdir(parents=True, exist_ok=True)
        self.fake_video.touch()

    def tearDown(self):
        settings.db_path = self._original_db
        os.unlink(self._tmp_db.name)
        if self.fake_video.exists():
            self.fake_video.unlink()
        POSE_SCAN_JOBS.clear()
        AUDIO_SCAN_JOBS.clear()

    def test_start_pose_scan_not_found(self):
        response = self.client.post("/hit-study/nonexistent/pose-scan", json={})
        self.assertEqual(response.status_code, 404)

    def test_start_pose_scan_any_algorithm(self):
        with database._connect() as conn:
            conn.execute(
                "INSERT INTO analyses (id, video_id, algorithm, status, created_at, updated_at) "
                "VALUES ('an2', 'vid1', 'median_frame', 'done', 0, 0)"
            )
            conn.commit()
        # Should now succeed (200) instead of 400
        response = self.client.post("/hit-study/an2/pose-scan", json={})
        self.assertEqual(response.status_code, 200)

    def test_start_pose_scan_success(self):
        # We don't want to actually run the scan because it needs ultralytics
        # But we can check if it starts and returns a job_id
        response = self.client.post("/hit-study/an1/pose-scan", json={
            "range_start_s": 0,
            "range_end_s": 5
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "pending")
        self.assertIn(data["job_id"], POSE_SCAN_JOBS)

    def test_pose_scan_status_not_found(self):
        response = self.client.get("/hit-study/pose-scan/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_pose_scan_status_success(self):
        job_id = "job1"
        POSE_SCAN_JOBS[job_id] = {"job_id": job_id, "status": "running"}
        response = self.client.get(f"/hit-study/pose-scan/{job_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")

    def test_start_audio_scan_success(self):
        response = self.client.post("/hit-study/an1/audio-scan", json={
            "range_start_s": 0,
            "range_end_s": 5
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "pending")
        self.assertIn(data["job_id"], AUDIO_SCAN_JOBS)

    def test_hit_study_data_fallback(self):
        # Even with no artifacts, it should return a basic structure
        response = self.client.get("/hit-study/an1/data")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["duration_s"], 10.0)
        self.assertEqual(len(data["frames"]), 0)

    def test_upload_hit_labels(self):
        payload = {
            "labels": [
                {"time_s": 1.2, "is_strike": True, "comment": "test1"},
                {"time_s": 3.4, "is_strike": True}
            ]
        }
        response = self.client.post("/hit-study/an1/labels/upload?filename=custom-labels.json", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["labels"]), 2)
        self.assertEqual(data["labels"][0]["time_s"], 1.2)
        self.assertEqual(data.get("source_file"), "custom-labels.json")

        # Verify DB updated active_labels_path
        with database._connect() as conn:
            row = conn.execute("SELECT active_labels_path FROM analyses WHERE id='an1'").fetchone()
            self.assertTrue(row[0].endswith("custom-labels.json"))

    def test_save_ball_scan_active_path(self):
        payload = {
            "job_id": "",
            "result": {
                "candidates": [],
                "candidate_count": 0,
                "ball_detector": "tracknet"
            }
        }
        response = self.client.post("/hit-study/an1/ball-scan/save", json=payload)
        self.assertEqual(response.status_code, 200)
        
        # Verify DB updated active_ball_scan_path
        with database._connect() as conn:
            row = conn.execute("SELECT active_ball_scan_path FROM analyses WHERE id='an1'").fetchone()
            self.assertIsNotNone(row[0])
            self.assertTrue(row[0].endswith("an1.ball-scan.json"))

if __name__ == "__main__":
    unittest.main()
