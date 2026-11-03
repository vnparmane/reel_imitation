from __future__ import annotations

import mss
import numpy as np
from numpy.typing import NDArray

from reel_clone.capture.viewport import BoundingBox


class ScreenGrabber:
    def __init__(self, window_offset_x: int = 0, window_offset_y: int = 0) -> None:
        self.window_offset_x = window_offset_x
        self.window_offset_y = window_offset_y

    def grab_rgb(self, bbox: BoundingBox) -> NDArray[np.uint8]:
        monitor = bbox.to_mss_monitor(self.window_offset_x, self.window_offset_y)
        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            bgra = np.asarray(sct_img, dtype=np.uint8)
            return np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])

    def close(self) -> None:
        pass