from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
from playwright.async_api import async_playwright

from reel_clone.actor.executor import ActionExecutor
from reel_clone.cli import make_initial_loop_state, process_single_tick
from reel_clone.config import HardwareConfig, StorageConfig
from reel_clone.encoders.fusion import EncodedTick
from reel_clone.memory.buffer import TrajectoryBuffer
from reel_clone.observer.human_listener import HumanActionListener
from reel_clone.policy.network import TemporalBehaviorPolicy


FAKE_INSTAGRAM_HTML = """
<!DOCTYPE html>
<html>
<body style="margin:0; padding:0;">
  <div id="feed" style="height: 800px; display: flex; align-items: center; justify-content: center;">
    <div id="reel-card">
      <a id="reel-link" href="/reels/REEL_ONE/">Link</a>
      <video id="v1" src="blob:https://www.instagram.com/aaa-111" style="width: 360px; height: 640px;"></video>
      <button id="like-btn" style="padding: 16px;" onclick="
        const svg = document.getElementById('heart-icon');
        svg.setAttribute('aria-label', svg.getAttribute('aria-label') === 'Like' ? 'Unlike' : 'Like');
      ">
        <svg id="heart-icon" aria-label="Like" width="24" height="24"></svg>
      </button>
    </div>
  </div>
  <script>
    window.__is_autonomous = false;
    const reels = ['REEL_ONE', 'REEL_TWO', 'REEL_THREE'];
    let idx = 0;
    window.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' && idx < reels.length - 1) {
        idx += 1;
        document.getElementById('reel-link').setAttribute('href', '/reels/' + reels[idx] + '/');
        const v = document.getElementById('v1');
        delete v.dataset.reelCloneId;
        document.getElementById('heart-icon').setAttribute('aria-label', 'Like');
      } else if (e.key === 'ArrowUp' && idx > 0) {
        idx -= 1;
        document.getElementById('reel-link').setAttribute('href', '/reels/' + reels[idx] + '/');
        const v = document.getElementById('v1');
        delete v.dataset.reelCloneId;
      }
    });
  </script>
</body>
</html>
"""


def _tick(reel_id: str, idx: int, wall_sec: float, unix_ms: float, liked: bool = False) -> EncodedTick:
    return EncodedTick(
        tick_index=idx,
        reel_id=reel_id,
        elapsed_sec=idx * 0.5,
        duration_sec=10.0,
        loop_count=0,
        is_liked=liked,
        is_paused=False,
        wall_time=wall_sec,
        unix_ts_ms=unix_ms,
        fused_embedding=torch.zeros(1284, dtype=torch.float32),
        latency_ms=5.0,
    )


@pytest.mark.asyncio
async def test_boundary_like_and_skip_attribution_censoring_and_f8_isolation(tmp_path: Path) -> None:
    storage_cfg = StorageConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        sessions_dir=tmp_path / "data" / "sessions",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    buffer = TrajectoryBuffer(storage_cfg)
    buffer.start_session("live_attr_session")
    hw = HardwareConfig()
    policy = TemporalBehaviorPolicy(hw_config=hw)
    state = make_initial_loop_state()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 860})
        await page.set_content(FAKE_INSTAGRAM_HTML)

        listener = HumanActionListener(page)
        await listener.install_listeners()
        actor = ActionExecutor(page, skip_cooldown_sec=0.0)

        t0_ms = await page.evaluate("performance.timeOrigin + performance.now()")

        await process_single_tick(_tick("REEL_ONE", 0, 1.0, t0_ms), False, listener, buffer, actor, policy, state, "live_attr_session")
        await process_single_tick(_tick("REEL_ONE", 1, 1.5, t0_ms + 500), False, listener, buffer, actor, policy, state, "live_attr_session")

        await page.click("#like-btn")
        await page.keyboard.press("ArrowDown")
        t_after_skip_ms = await page.evaluate("performance.timeOrigin + performance.now()")
        await page.mouse.wheel(0, 100)

        act_reel_two = await process_single_tick(
            _tick("REEL_TWO", 0, 2.0, t_after_skip_ms + 200),
            False,
            listener,
            buffer,
            actor,
            policy,
            state,
            "live_attr_session",
            inertia_grace_sec=0.45,
        )
        assert act_reel_two == 0

        saved_files = sorted(storage_cfg.sessions_dir.glob("*.npz"))
        assert len(saved_files) == 1
        with np.load(saved_files[0]) as r1:
            assert list(r1["actions"]) == [0, 3]
            assert bool(r1["censored"]) is False

        await page.keyboard.press("ArrowUp")
        t_up_ms = await page.evaluate("performance.timeOrigin + performance.now()")
        await process_single_tick(
            _tick("REEL_ONE", 0, 3.0, t_up_ms + 200),
            False,
            listener,
            buffer,
            actor,
            policy,
            state,
            "live_attr_session",
        )
        saved_files = sorted(storage_cfg.sessions_dir.glob("*.npz"))
        assert len(saved_files) == 2
        with np.load(saved_files[1]) as r2:
            assert list(r2["actions"]) == [0]
            assert bool(r2["censored"]) is True

        await page.evaluate("window.__is_autonomous = true;")
        await actor.execute(1)
        t_auto_ms = await page.evaluate("performance.timeOrigin + performance.now()")
        await process_single_tick(
            _tick("REEL_TWO", 0, 4.0, t_auto_ms + 100),
            True,
            listener,
            buffer,
            actor,
            policy,
            state,
            "live_attr_session",
        )

        await page.evaluate("window.__is_autonomous = false;")
        act_after_auto = await process_single_tick(
            _tick("REEL_TWO", 1, 5.0, t_auto_ms + 600),
            False,
            listener,
            buffer,
            actor,
            policy,
            state,
            "live_attr_session",
        )
        assert act_after_auto == 0

        await browser.close()