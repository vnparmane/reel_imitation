from __future__ import annotations

import pytest
from playwright.async_api import async_playwright
from reel_clone.capture.viewport import BoundingBox, ReelViewportController, sanitize_reel_id


MOCK_REELS_HTML = """
<!DOCTYPE html>
<html>
<body style="margin:0; padding:0;">
  <div id="reel-1" style="height: 860px; display: flex; align-items: center; justify-content: center;">
    <a href="/reels/C8abcXYZ123/">Link</a>
    <video id="v1" src="blob:https://www.instagram.com/11111111-2222-3333"
           style="width: 360px; height: 640px;"></video>
    <button><svg aria-label="Unlike"></svg></button>
  </div>
</body>
</html>
"""


def test_sanitize_reel_id_handles_blob_collisions() -> None:
    id1 = sanitize_reel_id("blob:https://www.instagram.com/aaaa-1111")
    id2 = sanitize_reel_id("blob:https://www.instagram.com/bbbb-2222")
    assert id1 != id2
    assert ":" not in id1 and "/" not in id1
    assert sanitize_reel_id("C8abcXYZ123") == "C8abcXYZ123"


def test_bounding_box_to_mss_and_playwright_clip() -> None:
    bbox = BoundingBox(left=100, top=50, width=360, height=640)
    assert bbox.to_mss_monitor(10, 20) == {"left": 110, "top": 70, "width": 360, "height": 640}
    assert bbox.to_playwright_clip() == {"x": 100.0, "y": 50.0, "width": 360.0, "height": 640.0}


@pytest.mark.asyncio
async def test_inspect_active_reel_dom_and_natural_loop_detection() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 860})
        await page.set_content(MOCK_REELS_HTML)

        await page.evaluate(
            """() => {
                const v = document.getElementById('v1');
                Object.defineProperty(v, 'duration', { value: 10.0, writable: true });
                Object.defineProperty(v, 'currentTime', { value: 2.5, writable: true });
            }"""
        )

        controller = ReelViewportController()
        controller._page = page

        state1 = await controller.inspect_active_reel()
        assert state1 is not None
        assert state1.reel_id == "C8abcXYZ123"
        assert state1.is_liked is True
        assert state1.loop_count == 0

        # Mid-reel blob URL rotation must NOT split the reel ID
        await page.evaluate("document.getElementById('v1').src = 'blob:https://www.instagram.com/rotated-999';")
        state_rotated = await controller.inspect_active_reel()
        assert state_rotated is not None
        assert state_rotated.reel_id == "C8abcXYZ123"

        # Advance near end (>= 80% of 10s = 9.0s), then wrap to start (<= 25% of 10s = 0.4s)
        await page.evaluate("document.getElementById('v1').currentTime = 9.0;")
        await controller.inspect_active_reel()

        await page.evaluate("document.getElementById('v1').currentTime = 0.4;")
        state3 = await controller.inspect_active_reel()
        assert state3 is not None
        assert state3.loop_count == 1

        frame = await controller.capture_reel_rgb(state3.bbox)
        assert frame.shape == (640, 360, 3)

        await browser.close()