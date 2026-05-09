"""Experimental TrackNet inference for short tennis-ball diagnostic windows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import settings

TRACKNET_WIDTH = 640
TRACKNET_HEIGHT = 360

_MODEL_CACHE: dict[str, object] = {}


def default_tracknet_weights_path() -> Path:
    return settings.analysis_dir.parent / "models" / "tracknet" / "tracknet_weights.pth"


def run_tracknet_window(
    video_path: Path,
    time_s: float,
    *,
    window_s: float = 0.8,
    roi: dict | None = None,
    player_box: dict | None = None,
    exclude_player: bool = True,
    input_width: int = TRACKNET_WIDTH,
    input_height: int = TRACKNET_HEIGHT,
    weights_path: Path | None = None,
) -> dict:
    weights_path = weights_path or default_tracknet_weights_path()
    if not weights_path.exists():
        raise RuntimeError(f"TrackNet weights not found at {weights_path}")

    model, device = _load_model(weights_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for TrackNet diagnostic: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    start_s = max(0.0, float(time_s) - window_s)
    end_s = float(time_s) + window_s
    decode_start_s = max(0.0, start_s - 2.0 / max(1.0, fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, decode_start_s * 1000.0)

    frames: list[tuple[float, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = decode_start_s + idx / fps
        if t > end_s:
            break
        idx += 1
        frames.append((float(t), frame))
    cap.release()

    candidates: list[dict] = []
    for i in range(2, len(frames)):
        t = frames[i][0]
        if t < start_s or t > end_s:
            continue
        pred = _predict_ball(
            model,
            device,
            frames[i - 2][1],
            frames[i - 1][1],
            frames[i][1],
            input_width=input_width,
            input_height=input_height,
        )
        if pred is None:
            continue
        x, y, score = pred
        if roi and not (float(roi["x1"]) <= x <= float(roi["x2"]) and float(roi["y1"]) <= y <= float(roi["y2"])):
            continue
        if exclude_player and player_box and _point_in_box(x, y, player_box, pad_x=0.015, pad_y=0.02):
            continue
        candidates.append({
            "time_s": float(t),
            "dt_s": float(t - time_s),
            "x": float(x),
            "y": float(y),
            "area": 8.0,
            "score": float(score),
            "source": "tracknet",
        })

    before = [c for c in candidates if c["time_s"] < time_s]
    after = [c for c in candidates if c["time_s"] >= time_s]
    return {
        "fps": fps,
        "candidate_count": len(candidates),
        "before_count": len(before),
        "after_count": len(after),
        "candidates": candidates[:300],
    }


def _load_model(weights_path: Path):
    cache_key = str(weights_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.block(x)

    class BallTrackerNet(nn.Module):
        def __init__(self, out_channels: int = 256):
            super().__init__()
            self.out_channels = out_channels
            self.conv1 = ConvBlock(9, 64)
            self.conv2 = ConvBlock(64, 64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = ConvBlock(64, 128)
            self.conv4 = ConvBlock(128, 128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv5 = ConvBlock(128, 256)
            self.conv6 = ConvBlock(256, 256)
            self.conv7 = ConvBlock(256, 256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv8 = ConvBlock(256, 512)
            self.conv9 = ConvBlock(512, 512)
            self.conv10 = ConvBlock(512, 512)
            self.ups1 = nn.Upsample(scale_factor=2)
            self.conv11 = ConvBlock(512, 256)
            self.conv12 = ConvBlock(256, 256)
            self.conv13 = ConvBlock(256, 256)
            self.ups2 = nn.Upsample(scale_factor=2)
            self.conv14 = ConvBlock(256, 128)
            self.conv15 = ConvBlock(128, 128)
            self.ups3 = nn.Upsample(scale_factor=2)
            self.conv16 = ConvBlock(128, 64)
            self.conv17 = ConvBlock(64, 64)
            self.conv18 = ConvBlock(64, self.out_channels)

        def forward(self, x):
            batch_size = x.size(0)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.pool3(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.ups1(x)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.ups2(x)
            x = self.conv14(x)
            x = self.conv15(x)
            x = self.ups3(x)
            x = self.conv16(x)
            x = self.conv17(x)
            x = self.conv18(x)
            return x.reshape(batch_size, self.out_channels, -1)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    model = BallTrackerNet()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = (model, device)
    return model, device


def _predict_ball(
    model,
    device: str,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    frame_c: np.ndarray,
    *,
    input_width: int,
    input_height: int,
) -> tuple[float, float, float] | None:
    import torch

    input_width = _round_to_multiple(min(3840, max(128, int(input_width))), 8)
    input_height = _round_to_multiple(min(2160, max(72, int(input_height))), 8)
    resized = [
        cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
        for frame in (frame_c, frame_b, frame_a)
    ]
    stacked = np.concatenate(resized, axis=2).astype(np.float32) / 255.0
    inp = np.expand_dims(np.rollaxis(stacked, 2, 0), axis=0)
    with torch.no_grad():
        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()[0]
    return _postprocess_output(output, input_width, input_height)


def _postprocess_output(output: np.ndarray, input_width: int, input_height: int) -> tuple[float, float, float] | None:
    fmap = output.reshape((input_height, input_width)).astype(np.float32)
    if float(fmap.max()) <= 1.5:
        fmap = fmap * 255.0
    heatmap = np.clip(fmap, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is None:
        return None
    circle = circles[0][0]
    x = float(circle[0]) / input_width
    y = float(circle[1]) / input_height
    score = float(heatmap[int(max(0, min(input_height - 1, round(circle[1])))), int(max(0, min(input_width - 1, round(circle[0]))))])
    return x, y, score


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _point_in_box(x: float, y: float, box: dict, *, pad_x: float, pad_y: float) -> bool:
    return (
        max(0.0, float(box["x1"]) - pad_x) <= x <= min(1.0, float(box["x2"]) + pad_x)
        and max(0.0, float(box["y1"]) - pad_y) <= y <= min(1.0, float(box["y2"]) + pad_y)
    )
