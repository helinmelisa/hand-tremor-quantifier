"""
TremorAnalyzer — signal processing core.

Pipeline
--------
1. Collect centroid displacement samples into a rolling window.
2. Welch power-spectral density (PSD) on the scalar displacement signal.
3. Extract dominant tremor frequency + amplitude from the 2-15 Hz band.
4. Compute a tremor index (fraction of power in the tremor band).

Clinically relevant ranges
--------------------------
  3 –  6 Hz  →  Parkinson's resting tremor
  6 –  8 Hz  →  Mixed / overlap
  8 – 12 Hz  →  Essential tremor (action)
 >12 Hz      →  Physiological / high-frequency
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Tremor band classification
# ---------------------------------------------------------------------------

_BANDS = [
    (3.0,  6.0, "Parkinson's (resting)",   (0, 60, 255)),   # orange-red
    (6.0,  8.0, "Mixed",                   (0, 165, 255)),  # orange
    (8.0, 12.0, "Essential (action)",      (0, 255, 255)),  # yellow
    (12.0, 15.0, "High-frequency",         (200, 200, 200)),# grey
]

def classify_tremor(freq_hz: float) -> tuple[str, tuple]:
    """Return (label, BGR colour) for a dominant frequency."""
    for lo, hi, label, colour in _BANDS:
        if lo <= freq_hz < hi:
            return label, colour
    if freq_hz < 3.0:
        return "Stable / DC", (0, 255, 0)
    return "High-frequency", (200, 200, 200)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TremorMetrics:
    dom_frequency:  float           # Hz — dominant peak in 2-15 Hz band
    amplitude_px:   float           # pixels — 2·σ of displacement
    tremor_index:   float           # 0-1, fraction of PSD power in band
    classification: str
    class_colour:   tuple           # BGR
    freqs:          np.ndarray      # full PSD frequency axis
    psd:            np.ndarray      # full PSD values
    displacement:   np.ndarray      # scalar displacement time-series (window)
    n_samples:      int


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TremorAnalyzer:
    """Rolling-window tremor analyser."""

    TREMOR_LO = 2.0    # Hz
    TREMOR_HI = 15.0   # Hz

    def __init__(self, fps: float = 30.0, window_sec: float = 3.0):
        self.fps = fps
        win = max(int(fps * window_sec), 64)
        self._x: deque[float] = deque(maxlen=win)
        self._y: deque[float] = deque(maxlen=win)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._x.clear()
        self._y.clear()

    def update(self, points: np.ndarray) -> None:
        """Append the centroid of *points* (shape N×2) to the history."""
        if points is None or len(points) == 0:
            return
        cx, cy = points.mean(axis=0)
        self._x.append(float(cx))
        self._y.append(float(cy))

    def get_metrics(self) -> Optional[TremorMetrics]:
        """
        Compute and return TremorMetrics, or None if not enough data yet
        (need at least 1 second of samples).
        """
        min_samples = int(self.fps)
        if len(self._x) < min_samples:
            return None

        x = np.array(self._x)
        y = np.array(self._y)

        # Subtract mean so DC offset doesn't dominate
        disp = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2)

        # Welch PSD — nperseg covers ~1 s of data at minimum
        nperseg = min(len(disp), max(64, int(self.fps)))
        freqs, psd = welch(disp, fs=self.fps, nperseg=nperseg,
                           noverlap=nperseg // 2, window="hann")

        # Restrict analysis to tremor band
        band = (freqs >= self.TREMOR_LO) & (freqs <= self.TREMOR_HI)
        if not band.any():
            return None

        tremor_psd   = psd[band]
        tremor_freqs = freqs[band]

        dom_freq = float(tremor_freqs[np.argmax(tremor_psd)])
        amplitude = float(2.0 * disp.std())          # ~peak-to-peak

        total_power  = float(np.trapezoid(psd,          freqs))
        tremor_power = float(np.trapezoid(tremor_psd,   tremor_freqs))
        tremor_idx   = tremor_power / total_power if total_power > 0 else 0.0

        label, colour = classify_tremor(dom_freq)

        return TremorMetrics(
            dom_frequency  = dom_freq,
            amplitude_px   = amplitude,
            tremor_index   = tremor_idx,
            classification = label,
            class_colour   = colour,
            freqs          = freqs,
            psd            = psd,
            displacement   = disp,
            n_samples      = len(disp),
        )
