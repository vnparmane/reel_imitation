from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from playwright.async_api import Page


_LISTENER_JS = """() => {
    if (window.__reel_listener_active) return;
    window.__reel_listener_active = true;
    let lastSkipTime = 0;

    function nowMs() {
        return performance.timeOrigin + performance.now();
    }

    function isTypingInInput(el) {
        if (!el) return false;
        const tag = (el.tagName || '').toUpperCase();
        return tag === 'INPUT' || tag === 'TEXTAREA' || Boolean(el.isContentEditable);
    }

    window.addEventListener('keydown', (e) => {
        if (window.__is_autonomous) return;
        if (isTypingInInput(e.target) || e.repeat) return;
        const ts = nowMs();
        if (e.key === 'ArrowDown' || e.key === 'PageDown') {
            if (ts - lastSkipTime > 180) {
                lastSkipTime = ts;
                window.__py_record_action({ type: 1, ts: ts });
            }
        } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
            window.__py_record_action({ type: 8, ts: ts });
        }
    });

    window.addEventListener('wheel', (e) => {
        if (window.__is_autonomous) return;
        const ts = nowMs();
        if (e.deltaY > 45) {
            if (ts - lastSkipTime > 350) {
                lastSkipTime = ts;
                window.__py_record_action({ type: 1, ts: ts });
            }
        } else if (e.deltaY < -45) {
            window.__py_record_action({ type: 8, ts: ts });
        }
    }, { passive: true });

    window.addEventListener('click', (e) => {
        if (window.__is_autonomous) return;
        if (!e.target || !e.target.closest) return;
        const ts = nowMs();
        const btn = e.target.closest('button, [role="button"]') || e.target;
        if (btn.querySelector && btn.querySelector('svg[aria-label="Unlike"], svg[fill="#ff3040"], svg[fill="rgb(255, 48, 64)"]')) {
            window.__py_record_action({ type: 4, ts: ts });
            return;
        }
        if (
            e.target.closest('svg[aria-label="Like"]') ||
            (btn.querySelector && btn.querySelector('svg[aria-label="Like"]'))
        ) {
            window.__py_record_action({ type: 2, ts: ts });
        }
    }, true);

    window.addEventListener('dblclick', (e) => {
        if (window.__is_autonomous) return;
        if (!e.target || !e.target.closest) return;
        if (e.target.tagName === 'VIDEO' || e.target.closest('video')) {
            window.__py_record_action({ type: 2, ts: nowMs() });
        }
    }, true);
}"""


@dataclass(frozen=True)
class ActionEvent:
    action_type: int   # 1=SKIP_DOWN, 2=LIKE, 4=UNLIKE, 8=SCROLL_UP
    timestamp_ms: float


class HumanActionListener:
    def __init__(self, page: Page) -> None:
        self.page = page
        self._action_queue: asyncio.Queue[ActionEvent] = asyncio.Queue()
        self._listener_installed = False

    async def _on_action(self, source: object, payload: object) -> None:
        if isinstance(payload, dict):
            act = int(payload.get("type", 0))
            ts = float(payload.get("ts", time.time() * 1000.0))
        else:
            act = int(payload)  # type: ignore[arg-type]
            ts = time.time() * 1000.0
        await self._action_queue.put(ActionEvent(action_type=act, timestamp_ms=ts))

    async def install_listeners(self) -> None:
        if self._listener_installed:
            return
        await self.page.expose_binding("__py_record_action", self._on_action)
        await self.page.add_init_script(f"({_LISTENER_JS})();")
        await self.page.evaluate(_LISTENER_JS)
        self._listener_installed = True

    def drain_events(self) -> list[ActionEvent]:
        events: list[ActionEvent] = []
        while not self._action_queue.empty():
            try:
                events.append(self._action_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    def drain_actions(self) -> list[int]:
        return [e.action_type for e in self.drain_events()]

    @staticmethod
    def resolve_tick_action(
        events: list[int],
        already_liked_reel: bool = False,
    ) -> tuple[int, bool]:
        liked_state = already_liked_reel
        did_like = False
        did_skip = 1 in events

        for code in events:
            if code == 4:
                liked_state = False
            elif code == 2 and not liked_state:
                did_like = True
                liked_state = True

        mask = (1 if did_skip else 0) | (2 if did_like else 0)
        return mask, liked_state