from __future__ import annotations

import warnings
import cv2
import numpy as np
from numpy.typing import NDArray


class ReelOCR:
    def __init__(self, max_height: int = 480) -> None:
        self.max_height = max_height
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]
            self._engine = RapidOCR()
        except Exception as exc:
            warnings.warn(f"RapidOCR initialization failed ({exc}); OCR will return empty strings.", RuntimeWarning)
            self._engine = None

    def extract_text(self, frame_rgb: NDArray[np.uint8]) -> str:
        if self._engine is None or frame_rgb.size == 0:
            return ""

        h, w, _ = frame_rgb.shape
        if h > self.max_height:
            scale = self.max_height / float(h)
            new_w = max(1, int(w * scale))
            resized = cv2.resize(frame_rgb, (new_w, self.max_height), interpolation=cv2.INTER_AREA)
        else:
            resized = frame_rgb

        frame_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        result, _ = self._engine(frame_bgr)
        if not result:
            return ""

        lines = [str(item[1]).strip() for item in result if float(item[2]) >= 0.5]
        return " | ".join(lines)