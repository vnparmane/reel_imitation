from __future__ import annotations

import asyncio
import random
import time
from playwright.async_api import Error as PlaywrightError, Page


class ActionExecutor:
    def __init__(self, page: Page, skip_cooldown_sec: float = 1.0) -> None:
        self.page = page
        self.skip_cooldown_sec = skip_cooldown_sec
        self._last_skip_wall: float = 0.0

    async def execute(self, action: int) -> bool:
        try:
            if action == 1:
                now = time.perf_counter()
                if (now - self._last_skip_wall) < self.skip_cooldown_sec:
                    return False
                self._last_skip_wall = now
                await self.page.keyboard.press("ArrowDown")
                jitter_ms = random.gauss(120, 25)
                await asyncio.sleep(max(0.05, jitter_ms / 1000.0))
                return True

            if action == 2:
                liked = await self.page.evaluate(
                    """() => {
                        const videos = Array.from(document.querySelectorAll('video'));
                        const centerY = window.innerHeight / 2;
                        let bestVideo = null;
                        let bestDist = Infinity;
                        for (const v of videos) {
                            const rect = v.getBoundingClientRect();
                            if (rect.height < 150) continue;
                            const dist = Math.abs((rect.top + rect.height / 2) - centerY);
                            if (dist < bestDist) {
                                bestDist = dist;
                                bestVideo = v;
                            }
                        }
                        if (!bestVideo) return false;
                        let container = bestVideo.parentElement;
                        for (let i = 0; i < 6 && container && container.parentElement; i++) {
                            if (container.querySelector('svg[aria-label="Like"], svg[aria-label="Unlike"], svg[fill="#ff3040"]')) {
                                break;
                            }
                            container = container.parentElement;
                        }
                        const scope = container || bestVideo.parentElement;
                        if (!scope || scope.querySelector('svg[aria-label="Unlike"], svg[fill="#ff3040"]')) {
                            return false;
                        }
                        const likeSvg = scope.querySelector('svg[aria-label="Like"]');
                        if (likeSvg) {
                            const btn = likeSvg.closest('button, [role="button"]') || likeSvg.parentElement;
                            if (btn) {
                                btn.click();
                                return true;
                            }
                        }
                        return false;
                    }"""
                )
                if liked:
                    await asyncio.sleep(0.08)
                return bool(liked)
        except PlaywrightError:
            return False

        return False