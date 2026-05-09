import unittest

from app.pipeline.rally_detection import (
    LEFT_WRIST_IDX,
    RIGHT_WRIST_IDX,
    assemble_timeline,
    cluster_rallies,
    default_rally_knobs,
    validate_impacts_with_pose,
)


def _frame(time_s: float, person_at: list[dict]) -> dict:
    detections = []
    for i, p in enumerate(person_at):
        cx, cy = p["center"]
        half = 0.05
        det = {
            "person_index": i,
            "confidence": 0.9,
            "box": {"x1": cx - half, "y1": cy - half, "x2": cx + half, "y2": cy + half},
            "keypoints": [],
        }
        for j in range(17):
            wrist_xy = p.get("wrists", {}).get(j)
            if wrist_xy is not None:
                det["keypoints"].append({
                    "index": j,
                    "x": wrist_xy[0],
                    "y": wrist_xy[1],
                    "confidence": p.get("wrist_conf", 0.8),
                })
            else:
                det["keypoints"].append({"index": j, "x": 0.0, "y": 0.0, "confidence": 0.0})
        detections.append(det)
    return {"time_s": time_s, "detections": detections}


class ValidateImpactsTests(unittest.TestCase):
    def test_drops_low_velocity_wrists(self):
        knobs = default_rally_knobs()
        frames = [
            _frame(0.0, [{"center": (0.4, 0.5), "wrists": {RIGHT_WRIST_IDX: (0.45, 0.5)}}]),
            _frame(0.5, [{"center": (0.4, 0.5), "wrists": {RIGHT_WRIST_IDX: (0.46, 0.5)}}]),
            _frame(1.0, [{"center": (0.4, 0.5), "wrists": {RIGHT_WRIST_IDX: (0.45, 0.5)}}]),
        ]
        impacts = [{"time_s": 0.5, "amplitude": 0.1, "snr": 3.0}]
        out = validate_impacts_with_pose(impacts, frames, knobs)
        self.assertFalse(out[0]["validated"])

    def test_accepts_swing(self):
        knobs = default_rally_knobs()
        frames = [
            _frame(0.0, [{"center": (0.4, 0.5), "wrists": {RIGHT_WRIST_IDX: (0.4, 0.5)}}]),
            _frame(0.5, [{"center": (0.5, 0.4), "wrists": {RIGHT_WRIST_IDX: (0.7, 0.3)}}]),
            _frame(1.0, [{"center": (0.6, 0.45), "wrists": {RIGHT_WRIST_IDX: (0.65, 0.4)}}]),
        ]
        impacts = [{"time_s": 0.5, "amplitude": 0.2, "snr": 4.0}]
        out = validate_impacts_with_pose(impacts, frames, knobs)
        self.assertTrue(out[0]["validated"])
        self.assertGreaterEqual(out[0]["max_wrist_v"], knobs["min_wrist_velocity"])

    def test_no_detections_uses_snr_fallback(self):
        knobs = default_rally_knobs()
        impacts_high = [{"time_s": 1.0, "amplitude": 1.0, "snr": 12.0}]
        impacts_low = [{"time_s": 1.0, "amplitude": 0.1, "snr": 3.0}]
        # No pose frames → fall back to SNR threshold.
        self.assertTrue(validate_impacts_with_pose(impacts_high, [], knobs)[0]["validated"])
        self.assertFalse(validate_impacts_with_pose(impacts_low, [], knobs)[0]["validated"])

    def test_low_wrist_conf_uses_box_motion_fallback(self):
        knobs = default_rally_knobs()
        # Wrists are below conf threshold → fallback to body-center velocity bar.
        # Body moves 0.30 normalized units in 0.5s = 0.6/s, above the 0.5 fallback.
        frames = [
            _frame(0.0, [{
                "center": (0.3, 0.5),
                "wrists": {RIGHT_WRIST_IDX: (0.3, 0.5)},
                "wrist_conf": 0.05,
            }]),
            _frame(0.5, [{
                "center": (0.6, 0.5),
                "wrists": {RIGHT_WRIST_IDX: (0.6, 0.5)},
                "wrist_conf": 0.05,
            }]),
        ]
        impacts = [{"time_s": 0.25, "amplitude": 0.1, "snr": 4.0}]
        out = validate_impacts_with_pose(impacts, frames, knobs)
        self.assertTrue(out[0]["validated"])
        self.assertTrue(out[0]["fallback_used"])


class ClusterRalliesTests(unittest.TestCase):
    def test_two_close_hits_form_one_rally(self):
        knobs = default_rally_knobs()
        impacts = [
            {"time_s": 10.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
            {"time_s": 13.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
        ]
        rallies = cluster_rallies(impacts, knobs)
        self.assertEqual(len(rallies), 1)
        self.assertEqual(len(rallies[0]), 2)

    def test_far_apart_split_and_filtered_by_min_hits(self):
        knobs = default_rally_knobs()
        impacts = [
            {"time_s": 10.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
            {"time_s": 20.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
        ]
        rallies = cluster_rallies(impacts, knobs)
        self.assertEqual(rallies, [])

    def test_lone_impact_not_a_rally(self):
        knobs = default_rally_knobs()
        impacts = [{"time_s": 7.0, "validated": True, "snr": 5.0, "amplitude": 0.2}]
        self.assertEqual(cluster_rallies(impacts, knobs), [])

    def test_unvalidated_impacts_ignored(self):
        knobs = default_rally_knobs()
        impacts = [
            {"time_s": 1.0, "validated": False, "snr": 3.0, "amplitude": 0.1},
            {"time_s": 2.0, "validated": False, "snr": 3.0, "amplitude": 0.1},
        ]
        self.assertEqual(cluster_rallies(impacts, knobs), [])


class AssembleTimelineTests(unittest.TestCase):
    def test_fills_gaps_around_single_rally(self):
        knobs = {**default_rally_knobs(), "rally_padding_s": 1.0}
        rally = [
            {"time_s": 10.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
            {"time_s": 14.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
        ]
        timeline = assemble_timeline([rally], 60.0, rally, knobs)
        self.assertEqual(len(timeline), 3)
        self.assertFalse(timeline[0]["is_on"])
        self.assertEqual((timeline[0]["start_s"], timeline[0]["end_s"]), (0.0, 9.0))
        self.assertTrue(timeline[1]["is_on"])
        self.assertEqual((timeline[1]["start_s"], timeline[1]["end_s"]), (9.0, 15.0))
        self.assertEqual(timeline[1]["sample_count"], 2)
        self.assertFalse(timeline[2]["is_on"])
        self.assertEqual((timeline[2]["start_s"], timeline[2]["end_s"]), (15.0, 60.0))

    def test_padding_clamps_at_video_bounds(self):
        knobs = {**default_rally_knobs(), "rally_padding_s": 5.0}
        rally = [
            {"time_s": 0.2, "validated": True, "snr": 5.0, "amplitude": 0.2},
            {"time_s": 1.5, "validated": True, "snr": 5.0, "amplitude": 0.2},
        ]
        timeline = assemble_timeline([rally], 10.0, rally, knobs)
        on = [s for s in timeline if s["is_on"]][0]
        self.assertEqual(on["start_s"], 0.0)
        self.assertLessEqual(on["end_s"], 10.0)

    def test_overlapping_rallies_merge(self):
        knobs = {**default_rally_knobs(), "rally_padding_s": 2.0}
        rally_a = [
            {"time_s": 5.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
            {"time_s": 7.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
        ]
        rally_b = [
            {"time_s": 8.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
            {"time_s": 10.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
        ]
        timeline = assemble_timeline([rally_a, rally_b], 30.0, rally_a + rally_b, knobs)
        on = [s for s in timeline if s["is_on"]]
        self.assertEqual(len(on), 1)
        self.assertEqual(on[0]["sample_count"], 4)

    def test_unvalidated_impacts_appear_in_off_segment_samples(self):
        knobs = default_rally_knobs()
        rally = [
            {"time_s": 10.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
            {"time_s": 12.0, "validated": True, "snr": 5.0, "amplitude": 0.2},
        ]
        all_impacts = rally + [
            {"time_s": 30.0, "validated": False, "snr": 2.0, "amplitude": 0.05},
        ]
        timeline = assemble_timeline([rally], 60.0, all_impacts, knobs)
        trailing_off = [s for s in timeline if not s["is_on"] and s["start_s"] > 12.0][0]
        times = [s["time_s"] for s in trailing_off["samples"]]
        self.assertIn(30.0, times)


if __name__ == "__main__":
    unittest.main()
