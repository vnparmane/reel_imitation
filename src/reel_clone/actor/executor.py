from __future__ import annotations

import asyncio
import random
from playwright.async_api import Page


class ActionExecutor:
    """Executes model decisions on the live Instagram Reels DOM with human-like jitter."""

    def __init__(self, page: Page) -> None:
        self.page = page

    async def execute(self, action: int) -> None:
        """
        Args:
            action: 0 for CONTINUE (no-op), 1 for SKIP (ArrowDown), 2 for LIKE (click heart)
        """
        if action == 1:
            # SKIP: Press ArrowDown to scroll to the next reel
            await self.page.keyboard.press("ArrowDown")
            # Human-like delay jitter between actions (80ms to 180ms)
            jitter_ms = random.gauss(120, 25)
            await asyncio.sleep(max(0.05, jitter_ms / 1000.0))

        elif action == 2:
            # LIKE: Click the like button if not already liked
            liked = await self.page.evaluate(
                """() => {
                    const likeSvg = document.querySelector('svg[aria-label="Like"]');
                    if (likeSvg) {
                        const btn = likeSvg.closest('button') || likeSvg.parentElement;
                        if (btn) { btn.click(); return true; }
                    }
                    return false;
                }"""
            )
            if liked:
                await asyncio.sleep(0.08)
        else:
            # CONTINUE (0): Do nothing, let the reel play
            pass