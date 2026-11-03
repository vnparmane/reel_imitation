from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from reel_clone.capture.audio import WasapiLoopbackRecorder
from reel_clone.capture.ocr import ReelOCR
from reel_clone.capture.screen import ScreenGrabber
from reel_clone.capture.viewport import ReelViewportController, ReelViewportState
from reel_clone.config import CaptureConfig


@dataclass(frozen=True)
class RawObservationTick:
    tick_index: int
    wall_time: float
    unix_ts_ms: float
    viewport: ReelViewportState
    frame_rgb: NDArray[np.uint8]
    audio_pcm: NDArray[np.float32]
    ocr_text: str


class SynchronizedCaptureStream:
    def __init__(
        self,
        viewport_ctrl: ReelViewportController,
        screen_grabber: ScreenGrabber | None = None,
        audio_recorder: WasapiLoopbackRecorder | None = None,
        ocr_engine: ReelOCR | None = None,
        config: CaptureConfig | None = None,
    ) -> None:
        self.config = config or CaptureConfig()
        self.viewport_ctrl = viewport_ctrl
        self.screen = screen_grabber
        self._owns_audio = audio_recorder is None
        self.audio = audio_recorder or WasapiLoopbackRecorder(window_sec=self.config.tick_interval_sec)
        self.ocr = ocr_engine or ReelOCR()

        self._cached_ocr_text: str = ""
        self._last_ocr_reel_id: str | None = None

    async def _grab_frame(self, vp_state: ReelViewportState) -> NDArray[np.uint8]:
        if self.screen is not None:
            return await asyncio.to_thread(self.screen.grab_rgb, vp_state.bbox)
        return await self.viewport_ctrl.capture_reel_rgb(vp_state.bbox)

    async def stream_ticks(self, max_ticks: int | None = None) -> AsyncIterator[RawObservationTick]:
        self.audio.start()
        tick_idx = 0
        interval = self.config.tick_interval_sec
        next_deadline = time.perf_counter()

        try:
            while max_ticks is None or tick_idx < max_ticks:
                now = time.perf_counter()
                if now < next_deadline:
                    await asyncio.sleep(next_deadline - now)

                tick_start = time.perf_counter()
                unix_ms = time.time() * 1000.0
                next_deadline = max(next_deadline + interval, tick_start + interval * 0.5)

                vp_state = await self.viewport_ctrl.inspect_active_reel()
                if vp_state is None:
                    continue

                frame_rgb = await self._grab_frame(vp_state)
                audio_pcm = self.audio.get_latest_window()

                reel_changed = vp_state.reel_id != self._last_ocr_reel_id
                should_run_ocr = reel_changed or (tick_idx % self.config.ocr_every_n_ticks == 0)

                if should_run_ocr:
                    self._cached_ocr_text = await asyncio.to_thread(
                        self.ocr.extract_text, frame_rgb
                    )
                    self._last_ocr_reel_id = vp_state.reel_id

                yield RawObservationTick(
                    tick_index=tick_idx,
                    wall_time=tick_start,
                    unix_ts_ms=unix_ms,
                    viewport=vp_state,
                    frame_rgb=frame_rgb,
                    audio_pcm=audio_pcm,
                    ocr_text=self._cached_ocr_text,
                )
                tick_idx += 1
        finally:
            if self._owns_audio:
                self.audio.stop()