"""
HandDetector — skin-color segmentation + Shi-Tomasi seed extraction.
No deep-learning; runs entirely with OpenCV.

Face exclusion: a Haar-cascade frontal-face detector is run first and its
bounding boxes are zeroed out of the skin mask so that face/neck skin does
not compete with the hand for tracking points.
"""
import cv2
import numpy as np


class HandDetector:
    """
    Detects the hand region via HSV skin-color segmentation and returns
    Shi-Tomasi corner seeds for KLT initialisation.
    """

    # Two hue ranges cover most skin tones under typical indoor lighting.
    _LOWER1 = np.array([0,  30, 50],  dtype=np.uint8)
    _UPPER1 = np.array([25, 255, 255], dtype=np.uint8)
    _LOWER2 = np.array([160, 30, 50],  dtype=np.uint8)
    _UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

    # Haar face detector shipped with OpenCV (classical, no ML weights needed)
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def __init__(self, max_corners: int = 30, min_area: int = 4000):
        self.max_corners = max_corners
        self.min_area    = min_area
        self._kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hand_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a binary mask isolating the largest skin-colored region
        that is NOT the face."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, self._LOWER1, self._UPPER1) |
                cv2.inRange(hsv, self._LOWER2, self._UPPER2))

        # Clean small blobs and fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # --- Erase detected faces from the skin mask ---
        mask = self._erase_faces(mask, frame)

        # Keep only the largest remaining contour (the hand)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return np.zeros_like(mask)

        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)
        return clean

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _erase_faces(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Zero out any face bounding boxes in *mask* (with a 20 % padding)."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        h_img, w_img = mask.shape[:2]
        for (x, y, w, h) in faces:
            pad_x = int(w * 0.20)
            pad_y = int(h * 0.20)
            x1 = max(0,     x - pad_x)
            y1 = max(0,     y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            mask[y1:y2, x1:x2] = 0
        return mask

    def get_debug_stages(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return an ordered dict of labelled BGR images showing every step of
        the detection pipeline.  Use this to build a debug tile grid.

        Stages
        ------
        1. original       — raw camera frame
        2. skin_raw       — HSV threshold before any morphology
        3. skin_morph     — after open+close morphology
        4. face_erased    — after face bounding boxes are zeroed out
        5. hand_mask      — final single-contour hand mask
        6. seeds_overlay  — original frame with seed points drawn
        """
        # 1. Original
        original = frame.copy()

        # 2. Raw HSV skin mask
        hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = (cv2.inRange(hsv, self._LOWER1, self._UPPER1) |
                    cv2.inRange(hsv, self._LOWER2, self._UPPER2))

        # 3. After morphology
        morph = cv2.morphologyEx(raw_mask.copy(), cv2.MORPH_OPEN,  self._kernel)
        morph = cv2.morphologyEx(morph,           cv2.MORPH_CLOSE, self._kernel)

        # 4. After face erasure
        face_erased = self._erase_faces(morph.copy(), frame)

        # Draw face boxes on a colour copy for visibility
        face_vis = frame.copy()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        h_img, w_img = frame.shape[:2]
        for (x, y, w, h) in faces:
            pad_x = int(w * 0.20); pad_y = int(h * 0.20)
            x1 = max(0, x - pad_x);     y1 = max(0, y - pad_y)
            x2 = min(w_img, x+w+pad_x); y2 = min(h_img, y+h+pad_y)
            cv2.rectangle(face_vis, (x1, y1), (x2, y2), (0, 0, 220), 2)
        # Blend erased mask (green) over face_vis
        face_stage = _mask_overlay(face_vis, face_erased, colour=(0, 200, 80))

        # 5. Final hand contour mask
        contours, _ = cv2.findContours(face_erased, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        hand_mask = np.zeros_like(raw_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= self.min_area:
                cv2.drawContours(hand_mask, [largest], -1, 255, cv2.FILLED)

        # 6. Seeds overlay
        seeds_vis = frame.copy()
        if hand_mask.any():
            gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pts = cv2.goodFeaturesToTrack(gray_f, maxCorners=self.max_corners,
                                          qualityLevel=0.02, minDistance=12,
                                          mask=hand_mask, blockSize=7)
            if pts is not None:
                for p in pts.reshape(-1, 2):
                    cv2.circle(seeds_vis, (int(p[0]), int(p[1])), 5,
                               (0, 230, 118), -1, cv2.LINE_AA)
            seeds_vis = _mask_overlay(seeds_vis, hand_mask, colour=(0, 100, 255),
                                      alpha=0.25)

        return {
            "1 Original":      original,
            "2 Skin (raw HSV)": _gray_bgr(raw_mask),
            "3 Skin (morph)":  _gray_bgr(morph),
            "4 Face erased":   face_stage,
            "5 Hand mask":     _gray_bgr(hand_mask),
            "6 KLT seeds":     seeds_vis,
        }

    def get_seed_points(self, frame: np.ndarray):
        """
        Returns (points, mask) where points is an (N,1,2) float32 array of
        Shi-Tomasi corners inside the hand mask, or (None, mask) if none found.
        """
        mask = self.get_hand_mask(frame)
        if not mask.any():
            return None, mask

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=0.02,
            minDistance=12,
            mask=mask,
            blockSize=7,
        )
        return pts, mask


# ---------------------------------------------------------------------------
# Module-level helpers (used by get_debug_stages)
# ---------------------------------------------------------------------------

def _gray_bgr(mask: np.ndarray) -> np.ndarray:
    """Convert a single-channel binary mask to a 3-channel BGR image."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _mask_overlay(frame: np.ndarray, mask: np.ndarray,
                  colour=(0, 200, 80), alpha=0.45) -> np.ndarray:
    """Blend a coloured mask over *frame* with transparency *alpha*."""
    out     = frame.copy()
    colored = np.zeros_like(frame)
    colored[mask > 0] = colour
    cv2.addWeighted(colored, alpha, out, 1 - alpha, 0, out)
    return out
