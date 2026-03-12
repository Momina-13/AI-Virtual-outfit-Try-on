"""
clothing_overlay.py
-------------------
Functions for overlaying clothing images onto a body frame using
pose-derived measurements.

Usage
-----
    from utils.clothing_overlay import ClothingOverlay

    overlay = ClothingOverlay("clothing_samples/shirt1.png")
    result   = overlay.apply(frame, measurements, x_offset=0, y_offset=0)
"""

import os
import cv2
import numpy as np
from PIL import Image


class ClothingOverlay:
    """Loads a clothing image (RGBA) and blends it onto a BGR frame.

    Parameters
    ----------
    image_path : str
        Path to the clothing PNG (ideally with an alpha channel).
    """

    def __init__(self, image_path: str) -> None:
        self._path = image_path
        self._clothing_rgba = self._load(image_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        frame: np.ndarray,
        measurements: dict,
        x_offset: int = 0,
        y_offset: int = 0,
        scale_factor: float = 1.2,
    ) -> np.ndarray:
        """Blend the clothing onto *frame* using *measurements*.

        Parameters
        ----------
        frame : np.ndarray
            BGR camera frame to draw onto.
        measurements : dict
            Output of :meth:`PoseDetector.get_body_measurements`.
        x_offset : int
            Horizontal fine-tuning offset in pixels.
        y_offset : int
            Vertical fine-tuning offset in pixels.
        scale_factor : float
            Multiplier applied to the shoulder-width estimate.

        Returns
        -------
        np.ndarray
            Frame with the clothing blended in (BGR).
        """
        shoulder_width = measurements.get("shoulder_width", 0)
        torso_height = measurements.get("torso_height", 0)
        center_x = measurements.get("center_x", frame.shape[1] // 2)
        shoulder_y = measurements.get("shoulder_y", frame.shape[0] // 3)

        if shoulder_width < 10 or torso_height < 10:
            return frame

        target_w = int(shoulder_width * scale_factor)
        target_h = int(torso_height * 1.5)

        if target_w < 1 or target_h < 1:
            return frame

        resized = self._resize(self._clothing_rgba, target_w, target_h)

        x_start = center_x - target_w // 2 + x_offset
        y_start = shoulder_y - int(target_h * 0.1) + y_offset

        return self._blend(frame, resized, x_start, y_start)

    @staticmethod
    def list_catalog(samples_dir: str = "clothing_samples") -> list:
        """Return a list of PNG filenames found in *samples_dir*."""
        if not os.path.isdir(samples_dir):
            return []
        return sorted(
            f for f in os.listdir(samples_dir) if f.lower().endswith(".png")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str) -> np.ndarray:
        """Load a PNG (with or without alpha) as an RGBA NumPy array."""
        img = Image.open(path).convert("RGBA")
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _resize(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize an RGBA array to (*width*, *height*)."""
        pil = Image.fromarray(rgba, mode="RGBA")
        pil = pil.resize((width, height), Image.LANCZOS)
        return np.array(pil, dtype=np.uint8)

    @staticmethod
    def _blend(
        background: np.ndarray,
        overlay_rgba: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        """Alpha-blend *overlay_rgba* onto *background* at position (x, y).

        The function clips the overlay to the frame boundaries and handles
        transparent regions via the alpha channel.
        """
        result = background.copy()
        h_bg, w_bg = result.shape[:2]
        h_ov, w_ov = overlay_rgba.shape[:2]

        # Compute valid source and destination rectangles
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(w_ov, w_bg - x)
        src_y2 = min(h_ov, h_bg - y)

        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return result  # Completely out of frame

        ov_crop = overlay_rgba[src_y1:src_y2, src_x1:src_x2]
        alpha = ov_crop[:, :, 3:4].astype(np.float32) / 255.0

        fg_bgr = cv2.cvtColor(ov_crop[:, :, :3], cv2.COLOR_RGB2BGR)
        bg_crop = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)

        blended = fg_bgr.astype(np.float32) * alpha + bg_crop * (1.0 - alpha)
        result[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
        return result
