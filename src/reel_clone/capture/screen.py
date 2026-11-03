from __future__ import annotations

import mss
import numpy as np
from numpy.typing import NDArray

from reel_clone.capture.viewport import BoundingBox


class ScreenGrabber:
    """Fast screen region capture using mss, returning contiguous RGB uint8 arrays."""

    def __init__(self, window_offset_x: int = 0, window_offset_y: int = 0) -> None:
        self.window_offset_x = window_offset_x
        self.window_offset_y = window_offset_y
        self._sct = mss.mss()

    def grab_rgb(self, bbox: BoundingBox) -> NDArray[np.uint8]:
        """Captures the specified bounding box and returns an (H, W, 3) RGB uint8 numpy array."""
        monitor = bbox.to_mss_monitor(self.window_offset_x, self.window_offset_y)
        sct_img = self._sct.grab(monitor)
        # mss returns BGRA (H, W, 4); slice to BGR and reverse to RGB
        bgra = np.asarray(sct_img, dtype=np.uint8)
        rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
        return rgb

    def close(self) -> None:
        self._sct.close()