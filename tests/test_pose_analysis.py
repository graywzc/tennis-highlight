import unittest

from fastapi import HTTPException

from app.pipeline.pose_analysis import summarize_pose_frames
from app.routers.process import _analysis_range


class PoseAnalysisTests(unittest.TestCase):
    def test_analysis_range_defaults_full_video(self):
        self.assertEqual(_analysis_range(None, None, 100.0), (0.0, 100.0))

    def test_analysis_range_clamps_to_duration(self):
        self.assertEqual(_analysis_range(-10.0, 200.0, 100.0), (0.0, 100.0))

    def test_analysis_range_rejects_empty(self):
        with self.assertRaises(HTTPException):
            _analysis_range(50.0, 50.0, 100.0)

    def test_pose_summary_handles_fake_detections(self):
        frames = [
            {"time_s": 0.0, "detections": []},
            {
                "time_s": 0.5,
                "detections": [
                    {
                        "confidence": 0.8,
                        "keypoints": [
                            {"confidence": 0.5},
                            {"confidence": 0.7},
                        ],
                    }
                ],
            },
        ]
        summary = summarize_pose_frames(frames)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["frames_with_poses"], 1)
        self.assertAlmostEqual(summary["avg_detections_per_frame"], 0.5)
        self.assertAlmostEqual(summary["avg_keypoint_confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
