# Hand Tremor Quantifier

A real-time hand tremor analysis tool built entirely with classical computer vision — no neural networks, no pretrained models. Point a webcam at a hand, and it continuously reports tremor frequency, amplitude, and a clinical-band classification relevant to Parkinson's disease screening research.

---

## Motivation

Tremor is one of the earliest and most distinctive motor symptoms of Parkinson's disease, typically manifesting as a **3–6 Hz resting tremor** in the hands. Capturing and quantifying this signal non-invasively — using only a standard webcam — has real potential for low-cost, scalable screening tools.

This project demonstrates that a pure signal-processing pipeline (optical flow + FFT) can extract clinically meaningful tremor parameters without any machine learning.

---

## How It Works

### 1. Hand Detection — `src/detector.py`

The first step is finding the hand in the frame. We use **HSV colour-space skin segmentation**, which is robust to moderate lighting variation without requiring any trained model.

- The frame is converted from BGR to HSV.
- Two hue ranges are thresholded to form a binary skin mask (the two ranges handle the wraparound at 0°/360° in the hue channel, capturing reddish-pink tones):
  ```
  Range 1:  H ∈ [0°, 25°],   S ∈ [30, 255],   V ∈ [50, 255]
  Range 2:  H ∈ [160°, 180°], S ∈ [30, 255],   V ∈ [50, 255]
  ```
- Morphological **open** (removes small blobs) and **close** (fills holes) operations clean the mask.
- The largest connected contour is kept — this is assumed to be the hand. Contours below a minimum area threshold are rejected.
- Within the cleaned mask, **Shi-Tomasi corner detection** (`goodFeaturesToTrack`) finds up to 30 stable feature points. These become the seeds for the KLT tracker.

**Why Shi-Tomasi seeds instead of explicit fingertip detection?** Fingertip localisation from contour convexity is brittle (hands partially occluded, folded fingers). Distributing stable corner features across the whole hand and tracking their collective centroid gives a more stable tremor signal.

---

### 2. KLT Optical Flow Tracking — `src/tracker.py`

Once seeded, points are tracked frame-to-frame using the **Lucas-Kanade algorithm with pyramidal refinement** (`calcOpticalFlowPyrLK`).

#### Lucas-Kanade intuition

LK assumes that the intensity patch around a point moves rigidly between consecutive frames. It solves for the 2D flow vector (u, v) by minimising the sum of squared intensity differences within a local window — analytically, via the **optical flow constraint equation**:

```
I_x · u  +  I_y · v  +  I_t  =  0
```

where I_x, I_y are spatial gradients and I_t is the temporal gradient at each pixel. Over a window of pixels this becomes an overdetermined system solved in least-squares.

#### Pyramidal extension

The pyramid allows tracking of large inter-frame motions (useful when tremor amplitude is large, or frame rate is low). The image is downsampled L times; LK is applied coarse-to-fine, with each level's solution seeding the next.

**Parameters used:**
| Parameter | Value | Rationale |
|---|---|---|
| `winSize` | 21 × 21 | Large enough to span texture-sparse skin regions |
| `maxLevel` | 3 | 4-level pyramid handles motions up to ~40 px |
| termination | 30 iter / 0.01 px | Balance accuracy vs. frame budget |
| error threshold | 25 px | Reject points with high residual error |
| `MIN_POINTS` | 6 | If fewer survive, trigger re-initialisation |

---

### 3. Tremor Signal Extraction & Analysis — `src/analyzer.py`

After each frame, we compute the **centroid** of all currently tracked points. This gives a single (x, y) position per frame. The tremor signal is then the **scalar displacement** from the mean centroid position over the rolling window:

```
disp[t] = sqrt( (x[t] - x̄)² + (y[t] - ȳ)² )
```

Subtracting the mean removes camera ego-motion and slow voluntary movements, isolating the involuntary tremor oscillation.

#### Spectral analysis — Welch's method

Rather than a plain FFT (which is noisy on short windows), we use **Welch's power spectral density** estimate. Welch splits the signal into overlapping segments, applies a Hann window to each, computes each segment's periodogram, and averages them. This trades frequency resolution for reduced variance — ideal for a short, noisy physiological signal.

```
fps      = 30 Hz  (or source fps)
window   = 3 s    → 90 samples
nperseg  = 30     → ~1 Hz frequency resolution
```

#### Metrics extracted

| Metric | Definition | Unit |
|---|---|---|
| **Dominant frequency** | Frequency of peak PSD in 2–15 Hz band | Hz |
| **Amplitude** | 2 · σ(displacement) ≈ peak-to-peak | pixels |
| **Tremor index** | Band power (2–15 Hz) / total power | 0 – 1 |

The tremor index is particularly useful: a value near 1 means essentially all motion energy is oscillatory in the tremor band; a value near 0 means motion is smooth or mostly DC (slow drift).

#### Clinical band classification

| Frequency range | Classification | Clinical context |
|---|---|---|
| < 3 Hz | Stable / DC | Slow voluntary movement |
| 3 – 6 Hz | Parkinson's (resting) | Classic PD resting tremor |
| 6 – 8 Hz | Mixed | Overlap zone |
| 8 – 12 Hz | Essential (action) | Essential tremor, postural |
| > 12 Hz | High-frequency | Physiological / task tremor |

> **Disclaimer:** This tool is a research prototype. Classification labels are for educational and exploratory purposes only — not for clinical diagnosis.

---

### 4. Visualisation — `src/visualizer.py`

The display is a single composite window with two panels rendered side-by-side.

**Left panel — live video:**
- Each tracked point is drawn as a green circle.
- A motion trail (last 20 positions, alpha-faded) traces each point's recent path.
- A semi-transparent HUD in the top-left corner shows the current frequency, amplitude, tremor index, and classification (colour-coded by clinical band).

**Right panel — analysis charts** (rendered via Matplotlib Agg backend to a numpy array, then composited with cv2):
- **Top:** Displacement time-series for the current analysis window.
- **Bottom:** Log-scale power spectrum. Grey = full spectrum; orange = tremor band (2–15 Hz); red dashed line = dominant frequency peak.

The Matplotlib figure is created once and updated in-place each frame to avoid Python object allocation overhead.

---

## Installation

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` — capture, KLT, drawing
- `numpy` — array maths
- `scipy` — Welch PSD (`scipy.signal.welch`)
- `matplotlib` — chart rendering (Agg backend, headless)

---

## Usage

```bash
# Live webcam (default device 0)
python main.py

# Specific webcam index
python main.py --source 1

# Analyse a pre-recorded video file
python main.py --source path/to/video.mp4

# Save annotated output video
python main.py --save output.mp4

# Export per-frame metrics to CSV (for offline analysis)
python main.py --export metrics.csv

# Combine flags
python main.py --source video.mp4 --save annotated.mp4 --export results.csv
```

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or video file path |
| `--save` | — | Write annotated video to this path |
| `--export` | — | Write per-frame CSV metrics |
| `--window` | `3.0` | Analysis window length in seconds |
| `--reinit` | `45` | Re-detect hand every N frames |

### Keyboard controls (while running)

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset tracker and clear signal history |
| `s` | Save a PNG snapshot to the current directory |

---

## CSV Export Format

When `--export metrics.csv` is used, each row corresponds to one processed frame:

```
frame, timestamp_s, freq_hz, amplitude_px, tremor_index, classification
1,     0.033,       0.0000,  0.0000,       0.0000,       (collecting...)
90,    3.000,       4.8200,  12.3400,      0.8721,       Parkinson's (resting)
...
```

This CSV can be imported into Python, R, or Excel for offline statistical analysis, cohort studies, or longitudinal tracking.

---

## Limitations & Known Constraints

- **Lighting sensitivity:** HSV skin detection struggles under harsh directional lighting or mixed colour temperatures. Pressing `r` to recalibrate often helps.
- **Single hand:** The largest skin contour is assumed to be one hand. Two hands or a face in frame will confuse the detector.
- **Pixel units:** Amplitude is reported in pixels. Converting to physical units (millimetres) requires knowing the hand-to-camera distance and camera intrinsics.
- **Frame rate:** Accurate frequency estimates up to the Nyquist limit (fps / 2). At 30 fps, frequencies above 15 Hz are aliased. Use a higher-fps camera for high-frequency tremor analysis.
- **Not a medical device:** Output is for research and educational use only.

---

## Potential Extensions

- **Camera calibration** — convert pixel amplitude to millimetres using a known reference object.
- **Skin colour calibration** — sample ROI on first frame to adapt HSV thresholds to the specific subject and lighting.
- **Longitudinal tracking** — append CSV exports across sessions and plot tremor trends over time.
- **Multi-metric dashboard** — add RMS jerk (third derivative of position) as a complementary tremor metric.
- **REST API** — wrap the analyzer in a Flask/FastAPI endpoint to stream metrics to a clinical dashboard.
