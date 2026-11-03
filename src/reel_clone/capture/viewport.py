from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from playwright.async_api import async_playwright, BrowserContext, Page, Playwright

from reel_clone.config import CaptureConfig


@dataclass(frozen=True)
class BoundingBox:
    """Screen-relative or viewport-relative pixel coordinates for mss capture."""
    left: int
    top: int
    width: int
    height: int

    def to_mss_monitor(self, window_offset_x: int = 0, window_offset_y: int = 0) -> dict[str, int]:
        return {
            "left": self.left + window_offset_x,
            "top": self.top + window_offset_y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class ReelViewportState:
    """Snapshot of the currently active Reel at tick t."""
    reel_id: str
    bbox: BoundingBox
    current_time_sec: float
    duration_sec: float
    is_paused: bool
    loop_count: int
    is_liked: bool


class ReelViewportController:
    """Manages the Instagram Reels browser session and detects reel boundaries & state."""

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self.config = config or CaptureConfig()
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._active_reel_id: str | None = None
        self._last_playback_time: float = 0.0
        self._loop_count: int = 0

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser page not initialized. Call start() first.")
        return self._page

    async def start(self, headless: bool = False) -> None:
        self.config.user_data_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.config.user_data_dir),
            headless=headless,
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            args=[
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        self._page = (
            self._context.pages[0]
            if self._context.pages
            else await self._context.new_page()
        )
        await self._page.goto(self.config.reels_url, wait_until="domcontentloaded")

    async def stop(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def inspect_active_reel(self) -> ReelViewportState | None:
        raw: dict[str, Any] | None = await self.page.evaluate(
            """() => {
                const videos = Array.from(document.querySelectorAll("video"));
                if (videos.length === 0) return null;

                const centerY = window.innerHeight / 2;
                let bestVideo = null;
                let bestDist = Infinity;

                for (const v of videos) {
                    const rect = v.getBoundingClientRect();
                    if (rect.height < 150 || rect.width < 100) continue;
                    const vCenterY = rect.top + rect.height / 2;
                    const dist = Math.abs(vCenterY - centerY);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestVideo = { video: v, rect };
                    }
                }

                if (!bestVideo) return null;
                const { video, rect } = bestVideo;

                const pathParts = window.location.pathname.split("/").filter(Boolean);
                const urlId = (pathParts[0] === "reels" && pathParts[1]) ? pathParts[1] : null;
                const reelId = urlId || video.currentSrc || "unknown_reel";

                const unlikeSvg = document.querySelector("svg[aria-label=\\"Unlike\\"]");
                const isLiked = Boolean(unlikeSvg);

                return {
                    reelId: reelId,
                    left: Math.max(0, Math.round(rect.left)),
                    top: Math.max(0, Math.round(rect.top)),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    currentTime: Number(video.currentTime || 0),
                    duration: Number(video.duration || 0),
                    paused: Boolean(video.paused),
                    isLiked: isLiked
                };
            }"""
        )

        if raw is None or raw["width"] <= 0 or raw["height"] <= 0:
            return None

        reel_id = str(raw["reelId"])
        current_time = float(raw["currentTime"])
        duration = float(raw["duration"])

        if reel_id != self._active_reel_id:
            self._active_reel_id = reel_id
            self._loop_count = 0
            self._last_playback_time = current_time
        else:
            if duration > 1.0 and (self._last_playback_time - current_time) > (duration * 0.5):
                self._loop_count += 1
            self._last_playback_time = current_time

        return ReelViewportState(
            reel_id=reel_id,
            bbox=BoundingBox(
                left=int(raw["left"]),
                top=int(raw["top"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
            ),
            current_time_sec=current_time,
            duration_sec=duration,
            is_paused=bool(raw["paused"]),
            loop_count=self._loop_count,
            is_liked=bool(raw["isLiked"]),
        )

