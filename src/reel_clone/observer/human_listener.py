from __future__ import annotations

import asyncio
from playwright.async_api import Page


_LISTENER_JS = """() => {
    if (window.__reel_listener_active) return;
    window.__reel_listener_active = true;
    let lastSkipTime = 0;

    function isTypingInInput(el) {
        if (!el) return false;
        const tag = (el.tagName || '').toUpperCase();
        return tag === 'INPUT' || tag === 'TEXTAREA' || Boolean(el.isContentEditable);
    }

    window.addEventListener('keydown', (e) => {
        if (window.__is_autonomous) return;
        if (isTypingInInput(e.target) || e.repeat) return;
        if (e.key === 'ArrowDown' || e.key === 'PageDown') {
            const now = Date.now();
            if (now - lastSkipTime > 250) {
                lastSkipTime = now;
                window.__py_record_action(1);
            }
        }
    });

    window.addEventListener('wheel', (e) => {
        if (window.__is_autonomous) return;
        if (e.deltaY > 45) {
            const now = Date.now();
            if (now - lastSkipTime > 450) {
                lastSkipTime = now;
                window.__py_record_action(1);
            }
        }
    }, { passive: true });

    window.addEventListener('click', (e) => {
        if (window.__is_autonomous) return;
        if (!e.target || !e.target.closest) return;
        const likeIcon = e.target.closest('svg[aria-label="Like"]');
        if (likeIcon) {
            window.__py_record_action(2);
        }
    }, true);

    window.addEventListener('dblclick', (e) => {
        if (window.__is_autonomous) return;
        if (!e.target || !e.target.closest) return;
        if (e.target.tagName === 'VIDEO' || e.target.closest('video')) {
            window.__py_record_action(2);
        }
    }, true);
}"""


class HumanActionListener:
    """Listens to real user input events on the browser page to label demonstration actions."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self._action_queue: asyncio.Queue[int] = asyncio.Queue()
        self._listener_installed = False

    async def _on_action(self, source: object, action_type: int) -> None:
        await self._action_queue.put(int(action_type))

    async def install_listeners(self) -> None:
        if self._listener_installed:
            return
        await self.page.expose_binding("__py_record_action", self._on_action)
        await self.page.add_init_script(f"({_LISTENER_JS})();")
        await self.page.evaluate(_LISTENER_JS)
        self._listener_installed = True

    def drain_actions(self) -> list[int]:
        """Drains all queued events accumulated during the current tick."""
        events: list[int] = []
        while not self._action_queue.empty():
            try:
                events.append(self._action_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    @staticmethod
    def resolve_tick_action(
        events: list[int],
        already_liked_reel: bool = False,
        dom_liked_now: bool = False,
    ) -> int:
        """
        Resolves a list of drained events into a bitmask label:
          0 = CONTINUE
          1 = SKIP
          2 = LIKE
          3 = LIKE + SKIP (both occurred in the same tick)
        """
        did_skip = 1 in events
        did_like = ((2 in events) or dom_liked_now) and (not already_liked_reel)
        mask = (1 if did_skip else 0) | (2 if did_like else 0)
        return mask