from __future__ import annotations

import time
import numpy as np
import pytest

from reel_clone.capture.audio import WasapiLoopbackRecorder
from reel_clone.capture.ocr import ReelOCR
from reel_clone.capture.screen import ScreenGrabber
from reel_clone.capture.sync_stream import SynchronizedCaptureStream
from reel_clone.capture.viewport import BoundingBox, ReelViewportController, ReelViewportState
from reel_clone.config import CaptureConfig


class DummyViewportController(ReelViewportController):
    async def inspect_active_reel(self) -> ReelViewportState | None:
        return ReelViewportState(
            reel_id="test_reel_123",
            bbox=BoundingBox(left=10, top=10, width=120, height=200),
            current_time_sec=1.5,
            duration_sec=15.0,
            is_paused=False,
            loop_count=0,
            is_liked=False,
        )


class DummyScreenGrabber(ScreenGrabber):
    def __init__(self) -> None:
        pass

    def grab_rgb(self, bbox: BoundingBox) -> np.ndarray:
        return np.zeros((bbox.height, bbox.width, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class DummyOCR(ReelOCR):
    def __init__(self) -> None:
        self.call_count = 0

    def extract_text(self, frame_rgb: np.ndarray) -> str:
        self.call_count += 1
        return f"caption_{self.call_count}"


@pytest.mark.asyncio
async def test_synchronized_capture_stream_2hz_and_ocr_caching() -> None:
    cfg = CaptureConfig(tick_interval_sec=0.1, ocr_every_n_ticks=2)
    vp = DummyViewportController(config=cfg)
    screen = DummyScreenGrabber()
    audio = WasapiLoopbackRecorder(window_sec=0.1)
    ocr = DummyOCR()

    stream = SynchronizedCaptureStream(
        viewport_ctrl=vp,
        screen_grabber=screen,
        audio_recorder=audio,
        ocr_engine=ocr,
        config=cfg,
    )

    ticks = []
    t0 = time.perf_counter()
    async for tick in stream.stream_ticks(max_ticks=4):
        ticks.append(tick)
    elapsed = time.perf_counter() - t0

    assert len(ticks) == 4
    # 4 ticks at 0.1s interval (first immediate + 3 intervals = ~0.3s)
    assert 0.25 <= elapsed <= 0.45

    # Check shapes
    assert ticks[0].frame_rgb.shape == (200, 120, 3)
    assert ticks[0].audio_pcm.shape == (1600,)  # 0.1s * 16000 Hz

    # OCR runs on tick 0 and tick 2 (cached on tick 1 and tick 3)
    assert ocr.call_count == 2
    assert ticks[0].ocr_text == "caption_1"
    assert ticks[1].ocr_text == "caption_1"
    assert ticks[2].ocr_text == "caption_2"
    assert ticks[3].ocr_text == "caption_2"