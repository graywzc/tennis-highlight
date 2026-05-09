import unittest

import numpy as np

from app.pipeline.audio_analysis import bandpass, envelope, find_impacts, spectral_centroid_at


def _click(sr: int, freq: float, duration_s: float = 0.01, amp: float = 1.0) -> np.ndarray:
    """Hann-windowed sinusoid burst, used to fake a tennis impact."""
    n = int(round(sr * duration_s))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / sr
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / max(1, n - 1)))
    return (amp * np.sin(2 * np.pi * freq * t) * win).astype(np.float32)


def _knobs(**overrides) -> dict:
    base = {
        "peak_height_mad_k": 6.0,
        "peak_prominence_mult": 2.0,
        "min_impact_separation_s": 0.15,
    }
    base.update(overrides)
    return base


class AudioAnalysisTests(unittest.TestCase):
    def test_bandpass_attenuates_low_and_passes_high(self):
        sr = 22050
        t = np.arange(sr) / sr  # 1 s
        low = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        high = np.sin(2 * np.pi * 3000 * t).astype(np.float32)
        out_low = bandpass(low, sr, 1000.0, 8000.0)
        out_high = bandpass(high, sr, 1000.0, 8000.0)
        # Drop the first/last 10 ms — sosfiltfilt edge transient.
        edge = int(0.01 * sr)
        rms_low = float(np.sqrt(np.mean(out_low[edge:-edge] ** 2)))
        rms_high = float(np.sqrt(np.mean(out_high[edge:-edge] ** 2)))
        self.assertLess(rms_low, 0.05)
        self.assertGreater(rms_high, 0.5)

    def test_envelope_peaks_at_synthesized_clicks(self):
        sr = 22050
        rng = np.random.default_rng(0)
        signal = (rng.normal(scale=0.005, size=sr * 5)).astype(np.float32)  # 5 s of low noise
        click_times = [0.5, 1.7, 2.4, 3.1, 4.2]
        click_n = int(0.01 * sr)
        for ts in click_times:
            start = int(ts * sr)
            signal[start:start + click_n] += _click(sr, 3000.0, 0.01, amp=0.5)
        filtered = bandpass(signal, sr, 1000.0, 8000.0)
        env = envelope(filtered, sr)
        impacts, floor = find_impacts(env, sr, _knobs())
        detected = sorted(i["time_s"] for i in impacts)
        self.assertEqual(len(detected), len(click_times))
        # Window-RMS centers each click on its midpoint, ~5 ms after onset.
        for actual, expected in zip(detected, click_times):
            self.assertAlmostEqual(actual, expected, delta=0.02)
        self.assertGreater(floor, 0.0)

    def test_constant_noise_produces_no_peaks(self):
        sr = 22050
        rng = np.random.default_rng(1)
        signal = rng.normal(scale=0.05, size=sr * 3).astype(np.float32)
        filtered = bandpass(signal, sr, 1000.0, 8000.0)
        env = envelope(filtered, sr)
        impacts, _ = find_impacts(env, sr, _knobs(peak_height_mad_k=10.0))
        self.assertEqual(len(impacts), 0)

    def test_min_separation_dedupes_burst(self):
        sr = 22050
        rng = np.random.default_rng(2)
        signal = rng.normal(scale=0.005, size=sr * 2).astype(np.float32)
        click_n = int(0.01 * sr)
        # Two clicks 50 ms apart — should be deduped under default 150 ms separation.
        for ts in (0.5, 0.55):
            start = int(ts * sr)
            signal[start:start + click_n] += _click(sr, 3000.0, 0.01, amp=0.6)
        filtered = bandpass(signal, sr, 1000.0, 8000.0)
        env = envelope(filtered, sr)
        impacts, _ = find_impacts(env, sr, _knobs())
        self.assertEqual(len(impacts), 1)

    def test_spectral_centroid_separates_thud_from_click(self):
        sr = 22050
        low = _click(sr, 700.0, 0.03, amp=1.0)
        high = _click(sr, 4200.0, 0.01, amp=1.0)
        self.assertLess(spectral_centroid_at(low, sr, len(low) // 2), 1500.0)
        self.assertGreater(spectral_centroid_at(high, sr, len(high) // 2), 3000.0)

    def test_min_spectral_centroid_drops_basketball_like_thud(self):
        sr = 22050
        signal = np.zeros(sr, dtype=np.float32)
        signal[int(0.25 * sr):int(0.25 * sr) + len(_click(sr, 700.0, 0.03, 0.9))] += _click(sr, 700.0, 0.03, 0.9)
        signal[int(0.70 * sr):int(0.70 * sr) + len(_click(sr, 4200.0, 0.01, 0.9))] += _click(sr, 4200.0, 0.01, 0.9)
        filtered = bandpass(signal, sr, 400.0, 8000.0)
        env = envelope(filtered, sr)
        impacts, _ = find_impacts(
            env,
            sr,
            _knobs(min_spectral_centroid_hz=2500.0),
            raw_samples=signal,
        )
        self.assertEqual(len(impacts), 1)
        self.assertAlmostEqual(impacts[0]["time_s"], 0.70, delta=0.03)
        self.assertGreater(impacts[0]["spectral_centroid_hz"], 2500.0)


if __name__ == "__main__":
    unittest.main()
