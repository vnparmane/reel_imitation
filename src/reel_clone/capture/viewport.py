from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
from numpy.typing import NDArray
from playwright.async_api import Error as PlaywrightError, async_playwright, BrowserContext, Page, Playwright

from reel_clone.config import CaptureConfig


def sanitize_reel_id(raw_id: str) -> str:
    """Converts shortcodes or blob URLs into collision-free, Windows-safe identifiers."""
    clean = raw_id.strip()
    if not clean or clean == "unknown_reel":
        return "unknown_reel"
    if re.fullmatch(r"[A-Za-z0-9_-]{5,32}", clean):
        return clean
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:12]
    tail = re.sub(r"[^A-Za-z0-9]", "", clean[-12:])[:8]
    return f"reel_{tail}_{digest}"


@dataclass(frozen=True)
class BoundingBox:
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

    def to_playwright_clip(self) -> dict[str, float]:
        return {
            "x": float(max(0, self.left)),
            "y": float(max(0, self.top)),
            "width": float(max(1, self.width)),
            "height": float(max(1, self.height)),
        }


@dataclass(frozen=True)
class ReelViewportState:
    reel_id: str
    bbox: BoundingBox
    current_time_sec: float
    duration_sec: float
    is_paused: bool
    loop_count: int
    is_liked: bool


class ReelViewportController:
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
            args=["--autoplay-policy=no-user-gesture-required"],
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

    async def capture_reel_rgb(self, bbox: BoundingBox) -> NDArray[np.uint8]:
        try:
            jpeg_bytes = await self.page.screenshot(
                clip=bbox.to_playwright_clip(),
                type="jpeg",
                quality=85,
            )
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                return np.ascontiguousarray(bgr[:, :, ::-1])
        except PlaywrightError:
            pass
        return np.zeros((max(1, bbox.height), max(1, bbox.width), 3), dtype=np.uint8)

    async def inspect_active_reel(self) -> ReelViewportState | None:
        try:
            raw: dict[str, Any] | None = await self.page.evaluate(
                """() => {
                    const videos = Array.from(document.querySelectorAll('video'));
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
                            bestVideo = { video: v, rect, dist };
                        }
                    }

                    if (!bestVideo || bestVideo.dist > window.innerHeight * 0.35) return null;
                    const { video, rect } = bestVideo;

                    let container = video.parentElement;
                    for (let i = 0; i < 6 && container && container.parentElement; i++) {
                        if (container.querySelector('svg[aria-label="Like"], svg[aria-label="Unlike"], svg[fill="#ff3040"], svg[fill="rgb(255, 48, 64)"]')) {
                            break;
                        }
                        container = container.parentElement;
                    }
                    const scope = container || video.parentElement || document;

                    if (!video.dataset.reelCloneId) {
                        let candidateId = video.getAttribute('data-reel-id');
                        if (!candidateId) {
                            const link = scope.querySelector('a[href*="/reel/"], a[href*="/reels/"]');
                            if (link && link.getAttribute('href')) {
                                const parts = link.getAttribute('href').split('/').filter(Boolean);
                                if (parts.length >= 2) candidateId = parts[1];
                            }
                        }
                        if (!candidateId) {
                            const pathParts = window.location.pathname.split('/').filter(Boolean);
                            if ((pathParts[0] === 'reels' || pathParts[0] === 'reel') && pathParts[1]) {
                                candidateId = pathParts[1];
                            }
                        }
                        if (!candidateId) {
                            window.__reel_seq = (window.__reel_seq || 0) + 1;
                            candidateId = (video.currentSrc || 'reel') + '_' + window.__reel_seq;
                        }
                        video.dataset.reelCloneId = String(candidateId);
                    }

                    const reelId = video.dataset.reelCloneId;
                    const unlikeSvg = scope.querySelector('svg[aria-label="Unlike"], svg[fill="#ff3040"], svg[fill="rgb(255, 48, 64)"]');
                    const isLiked = Boolean(unlikeSvg);

                    return {
                        reelId: String(reelId),
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
        except PlaywrightError:
            return None

        if raw is None or raw["width"] <= 0 or raw["height"] <= 0:
            return None

        reel_id = sanitize_reel_id(str(raw["reelId"]))
        current_time = float(raw["currentTime"])
        duration = float(raw["duration"])

        if reel_id != self._active_reel_id:
            self._active_reel_id = reel_id
            self._loop_count = 0
            self._last_playback_time = current_time
        else:
            if (
                duration > 1.0
                and self._last_playback_time >= duration * 0.80
                and current_time <= duration * 0.25
            ):
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