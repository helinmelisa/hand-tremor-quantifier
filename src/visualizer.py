"""
TremorVisualizer — renders a composite display frame.

Layout (side by side)
---------------------
  LEFT  : live video with KLT track trails + metric overlay
  RIGHT : two matplotlib sub-plots (displacement time-series + PSD spectrum)

The matplotlib figure is created once (Agg backend) and updated in-place
each call, then blitted to a numpy array — no new figure allocations per frame.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .analyzer import TremorMetrics


# ---------------------------------------------------------------------------
# Colour palette (BGR for OpenCV, RGB for matplotlib)
# ---------------------------------------------------------------------------
BG_DARK  = (26,  26,  46)       # BGR  #1a1a2e
PT_COLOR = (0, 230, 118)        # BGR  bright green
TR_COLOR = (0, 165, 255)        # BGR  orange


class TremorVisualizer:
    """
    Call render() every frame; returns a composite BGR numpy array ready
    for cv2.imshow().
    """

    TRAIL_LEN  = 20          # how many previous positions to draw per point
    PANEL_W    = 480         # width of the right-hand analysis panel (px)

    def __init__(self, fps: float = 30.0, video_h: int = 480):
        self.fps     = fps
        self.video_h = video_h

        # Per-point trails: dict[point_index → deque of (x,y)]
        self._trails: dict[int, list] = {}

        # Build matplotlib figure once
        fig_h_in = video_h  / 100
        fig_w_in = self.PANEL_W / 100
        self._fig, (self._ax_disp, self._ax_psd) = plt.subplots(
            2, 1,
            figsize=(fig_w_in, fig_h_in),
            dpi=100,
            facecolor="#0d0d1a",
        )
        self._fig.subplots_adjust(left=0.12, right=0.97,
                                   top=0.93, bottom=0.10, hspace=0.45)
        self._styled_axes()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(
        self,
        frame:    np.ndarray,
        good_new: Optional[np.ndarray],
        good_old: Optional[np.ndarray],
        metrics:  Optional[TremorMetrics],
    ) -> np.ndarray:
        """
        Returns composite BGR image: [video | analysis panel].
        """
        video = self._draw_video(frame, good_new, good_old, metrics)
        panel = self._draw_panel(metrics)

        # Resize panel to match video height if needed
        if panel.shape[0] != video.shape[0]:
            panel = cv2.resize(panel, (self.PANEL_W, video.shape[0]))

        return np.hstack([video, panel])

    # ------------------------------------------------------------------
    # Video side
    # ------------------------------------------------------------------

    def _draw_video(self, frame, good_new, good_old, metrics):
        out = frame.copy()
        h, w = out.shape[:2]

        # --- KLT trails ---
        if good_new is not None and good_old is not None:
            # Update per-point trails (match by proximity)
            new_trails: dict[int, list] = {}
            for i, (nx, ny) in enumerate(good_new):
                best_idx, best_d = i, float("inf")
                for prev_idx, trail in self._trails.items():
                    if prev_idx in new_trails:
                        continue
                    px, py = trail[-1]
                    d = (nx - px) ** 2 + (ny - py) ** 2
                    if d < best_d:
                        best_d, best_idx = d, prev_idx
                if best_d < 400:           # within 20 px — same point
                    new_trails[best_idx] = self._trails[best_idx]
                else:
                    new_trails[i] = []
                new_trails[best_idx if best_d < 400 else i].append((nx, ny))
                # Trim trail length
                new_trails[best_idx if best_d < 400 else i] = \
                    new_trails[best_idx if best_d < 400 else i][-self.TRAIL_LEN:]

            self._trails = new_trails

            # Draw trails
            for trail in self._trails.values():
                for j in range(1, len(trail)):
                    alpha = j / len(trail)
                    col   = tuple(int(c * alpha) for c in PT_COLOR)
                    cv2.line(out, _ipt(trail[j-1]), _ipt(trail[j]), col, 1, cv2.LINE_AA)

            # Draw current points
            for x, y in good_new:
                cv2.circle(out, (int(x), int(y)), 4, PT_COLOR, -1, cv2.LINE_AA)
        else:
            self._trails = {}

        # --- Metric overlay (top-left HUD) ---
        self._draw_hud(out, metrics)
        return out

    def _draw_hud(self, img, metrics: Optional[TremorMetrics]):
        h, w = img.shape[:2]
        # Semi-transparent dark banner
        banner_h = 110
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if metrics is None:
            _put(img, "Collecting data...", (12, 30), 0.6, (160, 160, 160))
            return

        freq_str  = f"Freq:  {metrics.dom_frequency:.2f} Hz"
        amp_str   = f"Amp:   {metrics.amplitude_px:.1f} px"
        idx_str   = f"Index: {metrics.tremor_index*100:.0f}%"
        cls_str   = metrics.classification
        col       = metrics.class_colour

        _put(img, freq_str, (12,  28), 0.65, (220, 220, 220))
        _put(img, amp_str,  (12,  52), 0.65, (220, 220, 220))
        _put(img, idx_str,  (12,  76), 0.65, (220, 220, 220))
        _put(img, cls_str,  (12, 100), 0.60, col, thickness=2)

    # ------------------------------------------------------------------
    # Analysis panel (matplotlib → numpy)
    # ------------------------------------------------------------------

    def _draw_panel(self, metrics: Optional[TremorMetrics]) -> np.ndarray:
        ax_d = self._ax_disp
        ax_p = self._ax_psd

        ax_d.cla(); ax_p.cla()
        self._styled_axes()

        if metrics is not None:
            t = np.arange(len(metrics.displacement)) / self.fps
            ax_d.plot(t, metrics.displacement, color="#00e676", lw=0.9)
            ax_d.set_xlim(0, max(t[-1], 1e-3))
            ax_d.set_ylim(0, max(metrics.displacement.max() * 1.3, 5))
            ax_d.set_title("Displacement (px)", color="white", fontsize=8, pad=3)
            ax_d.set_xlabel("time (s)", color="#888888", fontsize=7)

            # PSD — full spectrum
            ax_p.semilogy(metrics.freqs, metrics.psd + 1e-9,
                          color="#888888", lw=0.8)
            # Highlight tremor band
            band = (metrics.freqs >= 2) & (metrics.freqs <= 15)
            ax_p.semilogy(metrics.freqs[band], metrics.psd[band] + 1e-9,
                          color="#ff9800", lw=1.5)
            # Mark dominant frequency
            ax_p.axvline(metrics.dom_frequency, color="#ff4444",
                         lw=1.2, linestyle="--",
                         label=f"{metrics.dom_frequency:.2f} Hz")
            ax_p.legend(fontsize=7, facecolor="#1a1a2e",
                        edgecolor="none", labelcolor="white",
                        loc="upper right")
            ax_p.set_xlim(0, 20)
            ax_p.set_title("Power Spectrum", color="white", fontsize=8, pad=3)
            ax_p.set_xlabel("frequency (Hz)", color="#888888", fontsize=7)
        else:
            for ax in (ax_d, ax_p):
                ax.text(0.5, 0.5, "waiting for data…",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        color="#555577", fontsize=9)

        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    def _styled_axes(self):
        for ax in (self._ax_disp, self._ax_psd):
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#666688", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ipt(pt) -> tuple[int, int]:
    return (int(pt[0]), int(pt[1]))

def _put(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
