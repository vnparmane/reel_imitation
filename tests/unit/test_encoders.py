from __future__ import annotations

import numpy as np
import torch

from reel_clone.capture.sync_stream import RawObservationTick
from reel_clone.capture.viewport import BoundingBox, ReelViewportState
from reel_clone.config import HardwareConfig
from reel_clone.encoders.fusion import MultimodalFeatureExtractor


def _make_dummy_tick(ocr_text: str = "Wait for the end 😂", silent: bool = False) -> RawObservationTick:
    vp = ReelViewportState(
        reel_id="reel_xyz",
        bbox=BoundingBox(left=0, top=0, width=224, height=224),
        current_time_sec=3.0,
        duration_sec=12.0,
        is_paused=False,
        loop_count=1,
        is_liked=True,
    )
    frame = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    if silent:
        audio = np.zeros((8000,), dtype=np.float32)
    else:
        t = np.linspace(0, 0.5, 8000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    return RawObservationTick(
        tick_index=6,
        wall_time=100.0,
        viewport=vp,
        frame_rgb=frame,
        audio_pcm=audio,
        ocr_text=ocr_text,
    )


def test_multimodal_feature_extractor_shapes_and_metadata() -> None:
    hw = HardwareConfig()
    extractor = MultimodalFeatureExtractor(hw_config=hw, mock_mode=True)

    tick = _make_dummy_tick(ocr_text="POV: coding at 3am", silent=False)
    encoded = extractor.extract(tick)

    assert encoded.fused_embedding.shape == (1284,)
    assert not torch.isnan(encoded.fused_embedding).any()

    # Verify the last 4 elements match temporal_meta: [3.0/12.0=0.25, log(4.0), loop=1.0, liked=1.0]
    meta = encoded.fused_embedding[-4:].float().cpu()
    assert torch.isclose(meta[0], torch.tensor(0.25), atol=1e-3)
    assert torch.isclose(meta[2], torch.tensor(1.0), atol=1e-3)
    assert torch.isclose(meta[3], torch.tensor(1.0), atol=1e-3)


def test_silent_audio_and_empty_ocr_produce_zero_subvectors() -> None:
    extractor = MultimodalFeatureExtractor(mock_mode=True)
    tick = _make_dummy_tick(ocr_text="   ", silent=True)
    encoded = extractor.extract(tick)

    # Slices: [0:512] vision, [512:1024] ocr_text, [1024:1280] audio, [1280:1284] meta
    txt_slice = encoded.fused_embedding[512:1024]
    aud_slice = encoded.fused_embedding[1024:1280]

    assert torch.all(txt_slice == 0)
    assert torch.all(aud_slice == 0)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        assert vram_gb < 1.5