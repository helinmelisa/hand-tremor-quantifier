"""
Hand Tremor Quantifier
======================
Classical CV pipeline for quantifying hand tremor frequency and amplitude.

Usage
-----
  python main.py                       # webcam 0
  python main.py --source 1            # webcam 1
  python main.py --source video.mp4    # video file
  python main.py --save out.mp4        # save annotated output
  python main.py --export metrics.csv  # write per-frame metrics to CSV

Controls
--------
  q  — quit
  r  — reset tracking + clear history
  d  — toggle pipeline debug view (6-step grid window)
  s  — save a snapshot PNG to disk
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.detector   import HandDetector
from src.tracker    import KLTTracker
from src.analyzer   import TremorAnalyzer
from src.visualizer import TremorVisualizer


# ---------------------------------------------------------------------------
# Debug grid helper
# ---------------------------------------------------------------------------

def _make_debug_grid(stages: dict, tile_h: int = 300) -> np.ndarray:
    """
    Tile all pipeline stage images into a 2-row grid with labels.
    Each image is resized to *tile_h* height, keeping aspect ratio.
    """
    tiles = []
    for label, img in stages.items():
        # Resize to fixed height
        h, w = img.shape[:2]
        tile_w = int(w * tile_h / h)
        tile = cv2.resize(img, (tile_w, tile_h))

        # Draw label banner at the top
        banner = np.zeros((28, tile_w, 3), dtype=np.uint8)
        cv2.putText(banner, label, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1,
                    cv2.LINE_AA)
        tiles.append(np.vstack([banner, tile]))

    # Split into two rows of 3
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:])

    # Pad narrower row to same width
    w1, w2 = row1.shape[1], row2.shape[1]
    if w1 > w2:
        row2 = np.hstack([row2, np.zeros((row2.shape[0], w1 - w2, 3), np.uint8)])
    elif w2 > w1:
        row1 = np.hstack([row1, np.zeros((row1.shape[0], w2 - w1, 3), np.uint8)])

    return np.vstack([row1, row2])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hand Tremor Quantifier")
    p.add_argument("--source",  default="1",
                   help="Webcam index (default 0) or path to a video file.")
    p.add_argument("--save",    default=None,
                   help="Save annotated video to this path (e.g. out.mp4).")
    p.add_argument("--export",  default=None,
                   help="Export per-frame metrics to a CSV file.")
    p.add_argument("--window",  default=3.0, type=float,
                   help="Analysis window in seconds (default 3).")
    p.add_argument("--reinit",  default=45,  type=int,
                   help="Re-detect hand every N frames (default 45).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {args.source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 30.0
        print(f"[INFO] Could not read FPS from source — assuming {fps:.0f} fps")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {args.source} | {frame_w}×{frame_h} @ {fps:.1f} fps")

    detector   = HandDetector()
    tracker    = KLTTracker()
    analyzer   = TremorAnalyzer(fps=fps, window_sec=args.window)
    visualizer = TremorVisualizer(fps=fps, video_h=frame_h)

    writer  = None
    csv_fh  = None
    csv_out = None

    if args.export:
        csv_fh  = open(args.export, "w", newline="")
        csv_out = csv.writer(csv_fh)
        csv_out.writerow(["frame", "timestamp_s",
                          "freq_hz", "amplitude_px",
                          "tremor_index", "classification"])

    snapshot_idx = 0
    frame_idx    = 0
    debug_mode   = False
    t0           = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ---- (Re-)initialise tracking --------------------------------
        needs_init = (not tracker.is_initialized or
                      frame_idx % args.reinit == 0)
        if needs_init:
            pts, _ = detector.get_seed_points(frame)
            if pts is not None:
                tracker.initialize(frame, pts)

        # ---- Debug pipeline view ------------------------------------
        if debug_mode:
            stages = detector.get_debug_stages(frame)
            debug_grid = _make_debug_grid(stages, tile_h=300)
            cv2.imshow("Pipeline stages  |  d=close", debug_grid)
        else:
            cv2.destroyWindow("Pipeline stages  |  d=close")

        # ---- Track ---------------------------------------------------
        good_new, good_old = tracker.update(frame)

        if good_new is not None:
            analyzer.update(good_new)

        metrics = analyzer.get_metrics()

        # ---- CSV export ----------------------------------------------
        if csv_out and metrics is not None:
            ts = frame_idx / fps
            csv_out.writerow([
                frame_idx, f"{ts:.3f}",
                f"{metrics.dom_frequency:.4f}",
                f"{metrics.amplitude_px:.4f}",
                f"{metrics.tremor_index:.4f}",
                metrics.classification,
            ])

        # ---- Render --------------------------------------------------
        display = visualizer.render(frame, good_new, good_old, metrics)

        # Initialise video writer after first frame (know display size)
        if writer is None and args.save:
            h, w = display.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        if writer:
            writer.write(display)

        cv2.imshow("Hand Tremor Quantifier  |  q=quit  r=reset  d=debug  s=snapshot",
                   display)

        # ---- Keyboard ------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker  = KLTTracker()
            analyzer.reset()
            print("[INFO] Reset.")
        elif key == ord("d"):
            debug_mode = not debug_mode
            print(f"[INFO] Debug view {'ON' if debug_mode else 'OFF'}.")
        elif key == ord("s"):
            path = f"snapshot_{snapshot_idx:04d}.png"
            cv2.imwrite(path, display)
            print(f"[INFO] Snapshot saved: {path}")
            snapshot_idx += 1

    # ---- Cleanup -----------------------------------------------------
    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Video saved: {args.save}")
    if csv_fh:
        csv_fh.close()
        print(f"[INFO] Metrics exported: {args.export}")
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"[INFO] Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps effective)")


if __name__ == "__main__":
    main()
