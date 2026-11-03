from __future__ import annotations

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
async def test_synchronized_capture_stream_and_ocr_caching() -> None:
    cfg = CaptureConfig(tick_interval_sec=0.05, ocr_every_n_ticks=2)
    vp = DummyViewportController(config=cfg)
    screen = DummyScreenGrabber()
    audio = WasapiLoopbackRecorder(window_sec=0.05)
    ocr = DummyOCR()

    stream = SynchronizedCaptureStream(
        viewport_ctrl=vp,
        screen_grabber=screen,
        audio_recorder=audio,
        ocr_engine=ocr,
        config=cfg,
    )

    ticks = []
    async for tick in stream.stream_ticks(max_ticks=4):
        ticks.append(tick)

    assert len(ticks) == 4
    assert ticks[0].frame_rgb.shape == (200, 120, 3)
    assert ticks[0].audio_pcm.shape == (800,)
    assert ocr.call_count == 2
    assert [t.ocr_text for t in ticks] == ["caption_1", "caption_1", "caption_2", "caption_2"]