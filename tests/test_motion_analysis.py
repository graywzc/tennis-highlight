import unittest

import numpy as np

from app.pipeline.motion_analysis import (
    ratios_from_histograms,
    segments_from_scores,
)


class MotionAnalysisCacheTests(unittest.TestCase):
    def test_histogram_ratio_matches_direct_threshold(self):
        gray = np.array([[0, 10, 26], [25, 80, 255]], dtype=np.uint8)
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.uint32)

        cached = ratios_from_histograms(np.array([hist]), 25)[0]
        direct = float(np.count_nonzero(gray > 25) / gray.size)

        self.assertAlmostEqual(cached, direct)

    def test_merge_gap_toggle_changes_close_runs(self):
        times = np.arange(0, 8, dtype=np.float64)
        ratios = np.zeros_like(times)
        smoothed = np.array([0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0], dtype=np.float64)
        knobs = self._knobs(merge_gap_s=3.0, enable_merge_gap=True)

        merged = [s for s in segments_from_scores(times, ratios, smoothed, 7.0, knobs) if s["is_on"]]
        unmerged = [
            s for s in segments_from_scores(
                times, ratios, smoothed, 7.0, {**knobs, "enable_merge_gap": False}
            )
            if s["is_on"]
        ]

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(unmerged), 2)

    def test_min_segment_toggle_keeps_short_runs_when_disabled(self):
        times = np.arange(0, 5, dtype=np.float64)
        ratios = np.zeros_like(times)
        smoothed = np.array([0.2, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
        knobs = self._knobs(min_segment_s=3.0, enable_min_segment=True)

        filtered = [s for s in segments_from_scores(times, ratios, smoothed, 4.0, knobs) if s["is_on"]]
        kept = [
            s for s in segments_from_scores(
                times, ratios, smoothed, 4.0, {**knobs, "enable_min_segment": False}
            )
            if s["is_on"]
        ]

        self.assertEqual(len(filtered), 0)
        self.assertEqual(len(kept), 1)

    def test_padding_toggle_changes_segment_bounds(self):
        times = np.arange(0, 6, dtype=np.float64)
        ratios = np.zeros_like(times)
        smoothed = np.array([0.0, 0.2, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
        knobs = self._knobs(segment_padding_s=1.0, enable_padding=True)

        padded = [s for s in segments_from_scores(times, ratios, smoothed, 5.0, knobs) if s["is_on"]][0]
        unpadded = [
            s for s in segments_from_scores(
                times, ratios, smoothed, 5.0, {**knobs, "enable_padding": False}
            )
            if s["is_on"]
        ][0]

        self.assertEqual((padded["start_s"], padded["end_s"]), (0.0, 3.0))
        self.assertEqual((unpadded["start_s"], unpadded["end_s"]), (1.0, 2.0))

    def _knobs(self, **overrides):
        knobs = {
            "diff_threshold": 25,
            "motion_threshold": 0.1,
            "merge_gap_s": 3.0,
            "min_segment_s": 0.0,
            "segment_padding_s": 0.0,
            "sample_fps": 1.0,
            "median_bg_samples": 80,
            "enable_merge_gap": False,
            "enable_min_segment": False,
            "enable_padding": False,
        }
        knobs.update(overrides)
        return knobs


if __name__ == "__main__":
    unittest.main()
