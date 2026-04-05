"""
KLTTracker — pyramidal Lucas-Kanade optical-flow point tracker.
"""
import cv2
import numpy as np


class KLTTracker:
    """
    Wraps cv2.calcOpticalFlowPyrLK.  Call initialize() once with seed
    points, then call update() every frame to get tracked positions.
    """

    MIN_POINTS = 6          # reinit trigger threshold
    MAX_ERR    = 25.0       # per-point pixel-error rejection

    _LK_PARAMS = dict(
        winSize  = (21, 21),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self):
        self._prev_gray: np.ndarray | None = None
        self._pts:       np.ndarray | None = None  # (N,1,2) float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._pts is not None and len(self._pts) >= self.MIN_POINTS

    def initialize(self, frame: np.ndarray, points: np.ndarray) -> None:
        """Seed the tracker with (N,1,2) float32 corner points."""
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._pts = points.astype(np.float32)

    def update(self, frame: np.ndarray):
        """
        Track points into *frame*.
        Returns (good_new, good_old) as (N,2) float32 arrays,
        or (None, None) if tracking failed / too few points remain.
        """
        if not self.is_initialized:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._pts, None, **self._LK_PARAMS
        )

        # Accept only forward-tracked, low-error points
        good_mask = (status.ravel() == 1) & (err.ravel() < self.MAX_ERR)
        good_new = new_pts[good_mask]
        good_old = self._pts[good_mask]

        self._prev_gray = gray

        if len(good_new) < self.MIN_POINTS:
            self._pts = None          # signal caller to reinitialise
            return None, None

        self._pts = good_new.reshape(-1, 1, 2)
        return good_new.reshape(-1, 2), good_old.reshape(-1, 2)
