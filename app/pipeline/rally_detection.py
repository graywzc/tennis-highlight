"""Fuse audio-impact candidates with pose data to produce rally segments.

A rally is a sequence of validated ball-strike sounds whose consecutive gaps
are below ``max_gap_s``. An impact is "validated" when at least one tracked
player's wrist had elevated velocity around the impact time. Audio gives ms-
precision timing; pose at 2 fps gives the velocity check.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from app.pipeline.audio_analysis import analyze_audio_impacts, analyze_audio_impacts_range

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str, float | None], None]

LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
RALLY_DECISION_STAGE = "rally_audio_pose_v1"
RALLY_SOURCE = "rally_audio_pose"
OFF_DECISION_STAGE = "off_gap"
SNR_FORCE_ACCEPT = 8.0  # accept impacts with very high SNR even if pose blinked


def detect_rallies(
    video_path: Path,
    video_duration_s: float,
    pose_frames: list[dict],
    knobs: dict,
    progress_cb: ProgressCallback | None = None,
) -> tuple[dict, list[dict]]:
    """Top-level rally detection. Returns (audio_result, full timeline)."""
    audio = analyze_audio_impacts(video_path, knobs, progress_cb)
    impacts = audio["impacts"]

    if progress_cb:
        progress_cb(12.0, f"validating {len(impacts)} impacts against pose", None)
    validated = validate_impacts_with_pose(impacts, pose_frames, knobs)

    if progress_cb:
        progress_cb(14.0, "clustering impacts into rallies", None)
    rallies = cluster_rallies(validated, knobs)
    timeline = assemble_timeline(rallies, video_duration_s, validated, knobs)

    audio_summary = {
        "sample_rate": audio["sample_rate"],
        "noise_floor": audio["noise_floor"],
        "duration_s": audio["duration_s"],
        "impact_count": len(impacts),
        "validated_impact_count": sum(1 for i in validated if i.get("validated")),
        "rally_count": sum(1 for s in timeline if s.get("is_on")),
    }
    if progress_cb:
        progress_cb(
            15.0,
            f"rallies={audio_summary['rally_count']} hits={audio_summary['validated_impact_count']}",
            None,
        )
    return {**audio, "summary": audio_summary, "validated_impacts": validated}, timeline


def preview_audio_range(
    video_path: Path,
    range_start_s: float,
    range_end_s: float,
    pose_frames: list[dict],
    knobs: dict,
) -> dict:
    """Re-run audio candidate detection for a small absolute-time range."""
    centroid_threshold = float(knobs.get("min_spectral_centroid_hz", 0.0))
    # For interactive debugging, keep every detected impact peak and annotate
    # whether it passes the active centroid threshold. Otherwise low-centroid
    # peaks vanish from the plot, which makes threshold tuning hard to reason about.
    detection_knobs = {**knobs, "min_spectral_centroid_hz": 0.0}
    audio = analyze_audio_impacts_range(video_path, range_start_s, range_end_s, detection_knobs)
    nearby_pose = [
        f for f in pose_frames
        if range_start_s - float(knobs["pose_window_s"]) <= float(f.get("time_s", 0.0)) <= range_end_s + float(knobs["pose_window_s"])
    ]
    validated = validate_impacts_with_pose(audio["impacts"], nearby_pose, knobs)
    for imp in validated:
        centroid = float(imp.get("spectral_centroid_hz", 0.0))
        centroid_pass = centroid_threshold <= 0.0 or centroid >= centroid_threshold
        imp["centroid_pass"] = bool(centroid_pass)
        imp["centroid_threshold_hz"] = float(centroid_threshold)
        if not centroid_pass:
            imp["validated"] = False
            imp["rejection_reason"] = "centroid"
    return {
        **audio,
        "validated_impacts": validated,
        "summary": {
            "sample_rate": audio["sample_rate"],
            "noise_floor": audio["noise_floor"],
            "range_start_s": float(range_start_s),
            "range_end_s": float(range_end_s),
            "impact_count": len(audio["impacts"]),
            "centroid_pass_count": sum(1 for i in validated if i.get("centroid_pass", True)),
            "validated_impact_count": sum(1 for i in validated if i.get("validated")),
            "rejected_impact_count": sum(1 for i in validated if not i.get("validated")),
        },
    }


def default_rally_knobs() -> dict:
    return {
        "audio_sample_rate": 22050,
        "bandpass_low_hz": 1000.0,
        "bandpass_high_hz": 8000.0,
        "peak_height_mad_k": 6.0,
        "peak_prominence_mult": 2.0,
        "min_impact_separation_s": 0.15,
        "min_spectral_centroid_hz": 2500.0,
        "pose_window_s": 0.75,
        "wrist_conf_min": 0.3,
        "min_wrist_velocity": 0.4,
        "max_gap_s": 5.0,
        "min_hits_per_rally": 2,
        "rally_padding_s": 1.0,
    }


# -- Pose validation ------------------------------------------------------

def validate_impacts_with_pose(
    impacts: list[dict],
    pose_frames: list[dict],
    knobs: dict,
) -> list[dict]:
    if not impacts:
        return []
    window = float(knobs["pose_window_s"])
    wrist_conf_min = float(knobs["wrist_conf_min"])
    min_v = float(knobs["min_wrist_velocity"])
    # Body-center is steadier than the wrist endpoint, so a slightly higher bar
    # than the wrist threshold filters incidental drift while staying achievable.
    fallback_v = max(min_v * 1.2, 0.5)
    sorted_frames = sorted(pose_frames or [], key=lambda f: float(f.get("time_s", 0.0)))

    out = []
    for imp in impacts:
        t = float(imp["time_s"])
        nearby = [
            f for f in sorted_frames
            if abs(float(f.get("time_s", 0.0)) - t) <= window
        ]
        max_wrist_v = 0.0
        max_box_v = 0.0
        player_id: int | None = None
        wrist_seen = False
        if len(nearby) >= 2:
            tracks = _build_tracks(nearby)
            for track_id, samples in tracks.items():
                for prev, cur in zip(samples, samples[1:]):
                    dt = max(1e-3, cur["time_s"] - prev["time_s"])
                    box_v = _norm_distance(cur["center"], prev["center"]) / dt
                    if box_v > max_box_v:
                        max_box_v = box_v
                    for kp_idx in (LEFT_WRIST_IDX, RIGHT_WRIST_IDX):
                        a = prev["wrists"].get(kp_idx)
                        b = cur["wrists"].get(kp_idx)
                        if a is None or b is None:
                            continue
                        if a["confidence"] < wrist_conf_min or b["confidence"] < wrist_conf_min:
                            continue
                        wrist_seen = True
                        v = _norm_distance(a["xy"], b["xy"]) / dt
                        if v > max_wrist_v:
                            max_wrist_v = v
                            player_id = track_id

        snr = float(imp.get("snr", 0.0))
        if wrist_seen:
            validated = max_wrist_v >= min_v
            fallback = False
        elif not nearby:
            validated = snr >= SNR_FORCE_ACCEPT
            fallback = validated
        else:
            # Wrists unreliable in this window; fall back to body motion + higher bar.
            validated = max_box_v >= fallback_v or snr >= SNR_FORCE_ACCEPT
            fallback = validated

        out.append({
            **imp,
            "validated": bool(validated),
            "max_wrist_v": float(max_wrist_v),
            "max_box_v": float(max_box_v),
            "player_id": int(player_id) if player_id is not None else None,
            "fallback_used": bool(fallback),
        })
    return out


def _build_tracks(frames: list[dict]) -> dict[int, list[dict]]:
    """Greedy box-center tracking across consecutive frames.

    Frames are ordered by time. For each detection in frame f, attach it to
    the nearest detection from frame f-1 within ``track_match_radius`` (in
    normalized coords). Otherwise start a new track.
    """
    # At 2 fps, a fast-running player can cover ~0.30 of a wide-framing frame
    # between samples; tighten this and tracks break mid-rally.
    track_match_radius = 0.35
    tracks: dict[int, list[dict]] = {}
    next_id = 0
    prev_signatures: list[tuple[int, tuple[float, float]]] = []
    for frame in frames:
        t = float(frame.get("time_s", 0.0))
        new_signatures: list[tuple[int, tuple[float, float]]] = []
        used_prev: set[int] = set()
        for det in frame.get("detections", []) or []:
            box = det.get("box") or {}
            try:
                cx = (float(box["x1"]) + float(box["x2"])) / 2.0
                cy = (float(box["y1"]) + float(box["y2"])) / 2.0
            except (KeyError, TypeError, ValueError):
                continue
            best_id: int | None = None
            best_d = float("inf")
            for tid, (px, py) in prev_signatures:
                if tid in used_prev:
                    continue
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < best_d and d <= track_match_radius:
                    best_d = d
                    best_id = tid
            if best_id is None:
                best_id = next_id
                next_id += 1
            else:
                used_prev.add(best_id)
            wrists: dict[int, dict] = {}
            for kp in det.get("keypoints", []) or []:
                idx = int(kp.get("index", -1))
                if idx in (LEFT_WRIST_IDX, RIGHT_WRIST_IDX):
                    wrists[idx] = {
                        "xy": (float(kp.get("x", 0.0)), float(kp.get("y", 0.0))),
                        "confidence": float(kp.get("confidence", 0.0)),
                    }
            tracks.setdefault(best_id, []).append({
                "time_s": t,
                "center": (cx, cy),
                "wrists": wrists,
            })
            new_signatures.append((best_id, (cx, cy)))
        prev_signatures = new_signatures
    return tracks


def _norm_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# -- Clustering -----------------------------------------------------------

def cluster_rallies(validated_impacts: list[dict], knobs: dict) -> list[list[dict]]:
    accepted = [i for i in validated_impacts if i.get("validated")]
    accepted.sort(key=lambda i: float(i["time_s"]))
    if not accepted:
        return []
    max_gap = float(knobs["max_gap_s"])
    min_hits = max(1, int(knobs["min_hits_per_rally"]))
    rallies: list[list[dict]] = []
    current: list[dict] = []
    for imp in accepted:
        if not current or (imp["time_s"] - current[-1]["time_s"]) <= max_gap:
            current.append(imp)
        else:
            rallies.append(current)
            current = [imp]
    if current:
        rallies.append(current)
    return [r for r in rallies if len(r) >= min_hits]


# -- Timeline assembly ----------------------------------------------------

def assemble_timeline(
    rallies: list[list[dict]],
    video_duration_s: float,
    all_impacts: list[dict],
    knobs: dict,
) -> list[dict]:
    pad = float(knobs["rally_padding_s"])
    on_segments: list[dict] = []
    for r in rallies:
        first_t = float(r[0]["time_s"])
        last_t = float(r[-1]["time_s"])
        snrs = [float(i.get("snr", 0.0)) for i in r]
        on_segments.append({
            "start_s": max(0.0, first_t - pad),
            "end_s": min(float(video_duration_s), last_t + pad),
            "is_on": True,
            "source": RALLY_SOURCE,
            "raw_start_s": first_t,
            "raw_end_s": last_t,
            "decision_stage": RALLY_DECISION_STAGE,
            "sample_count": len(r),
            "samples": [_impact_sample(i) for i in r],
            "avg_score": float(sum(snrs) / len(snrs)) if snrs else None,
            "max_score": float(max(snrs)) if snrs else None,
            "min_score": float(min(snrs)) if snrs else None,
            "_player_ids": sorted({i["player_id"] for i in r if i.get("player_id") is not None}),
        })
    on_segments = _merge_overlapping(on_segments)
    return _fill_off_segments(on_segments, float(video_duration_s), all_impacts)


def _merge_overlapping(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: float(s["start_s"]))
    out = [dict(segs[0])]
    for seg in segs[1:]:
        last = out[-1]
        if float(seg["start_s"]) <= float(last["end_s"]):
            last["end_s"] = max(float(last["end_s"]), float(seg["end_s"]))
            combined = (last.get("samples") or []) + (seg.get("samples") or [])
            last["samples"] = combined
            last["sample_count"] = int((last.get("sample_count") or 0) + (seg.get("sample_count") or 0))
            raws = [v for v in [last.get("raw_start_s"), seg.get("raw_start_s")] if v is not None]
            if raws:
                last["raw_start_s"] = min(raws)
            raw_ends = [v for v in [last.get("raw_end_s"), seg.get("raw_end_s")] if v is not None]
            if raw_ends:
                last["raw_end_s"] = max(raw_ends)
            snrs = [float(s.get("snr", 0.0)) for s in combined]
            last["avg_score"] = float(sum(snrs) / len(snrs)) if snrs else None
            last["max_score"] = float(max(snrs)) if snrs else None
            last["min_score"] = float(min(snrs)) if snrs else None
        else:
            out.append(dict(seg))
    return out


def _fill_off_segments(
    on_segments: list[dict],
    video_duration_s: float,
    all_impacts: list[dict],
) -> list[dict]:
    timeline: list[dict] = []
    cursor = 0.0
    sorted_impacts = sorted(all_impacts, key=lambda i: float(i["time_s"]))
    for seg in on_segments:
        start = float(seg["start_s"])
        end = float(seg["end_s"])
        if start > cursor:
            timeline.append(_off_segment(cursor, start, sorted_impacts))
        timeline.append(seg)
        cursor = end
    if cursor < video_duration_s:
        timeline.append(_off_segment(cursor, video_duration_s, sorted_impacts))
    return timeline


def _off_segment(start_s: float, end_s: float, all_impacts: list[dict]) -> dict:
    in_range = [
        _impact_sample(i)
        for i in all_impacts
        if start_s <= float(i["time_s"]) < end_s
    ]
    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "is_on": False,
        "source": RALLY_SOURCE,
        "raw_start_s": None,
        "raw_end_s": None,
        "decision_stage": OFF_DECISION_STAGE,
        "sample_count": len(in_range),
        "samples": in_range,
        "avg_score": None,
        "max_score": None,
        "min_score": None,
    }


def _impact_sample(imp: dict) -> dict:
    return {
        "time_s": float(imp["time_s"]),
        "amplitude": float(imp.get("amplitude", 0.0)),
        "snr": float(imp.get("snr", 0.0)),
        "spectral_centroid_hz": float(imp.get("spectral_centroid_hz", 0.0)),
        "validated": bool(imp.get("validated", False)),
        "max_wrist_v": float(imp.get("max_wrist_v", 0.0)),
        "max_box_v": float(imp.get("max_box_v", 0.0)),
        "player_id": imp.get("player_id"),
        "fallback_used": bool(imp.get("fallback_used", False)),
    }
