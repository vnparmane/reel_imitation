from __future__ import annotations

from dataclasses import dataclass
import math
import time
import torch

from reel_clone.capture.sync_stream import RawObservationTick
from reel_clone.config import HardwareConfig
from reel_clone.encoders.audio import AudioEncoder
from reel_clone.encoders.vision import VisionTextEncoder


@dataclass(frozen=True)
class EncodedTick:
    tick_index: int
    reel_id: str
    elapsed_sec: float
    duration_sec: float
    loop_count: int
    is_liked: bool
    is_paused: bool
    wall_time: float
    unix_ts_ms: float
    fused_embedding: torch.Tensor
    latency_ms: float


class MultimodalFeatureExtractor:
    FUSED_DIM: int = VisionTextEncoder.EMBED_DIM * 2 + AudioEncoder.EMBED_DIM + 4  # 1284

    def __init__(
        self,
        hw_config: HardwareConfig | None = None,
        vision_encoder: VisionTextEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
        mock_mode: bool = False,
    ) -> None:
        self.hw = hw_config or HardwareConfig()
        self.vision = vision_encoder or VisionTextEncoder(hw_config=self.hw, mock_mode=mock_mode)
        self.audio = audio_encoder or AudioEncoder(hw_config=self.hw, use_wav2vec2=not mock_mode)
        self.device = self.vision.device
        self.dtype = self.vision.dtype

    @torch.inference_mode()
    def extract(self, raw_tick: RawObservationTick) -> EncodedTick:
        t0 = time.perf_counter()

        vis_emb = self.vision.encode_frame(raw_tick.frame_rgb)
        txt_emb = self.vision.encode_ocr_text(raw_tick.ocr_text)
        aud_emb = self.audio.encode_audio(raw_tick.audio_pcm)

        vp = raw_tick.viewport
        safe_dur = max(vp.duration_sec, 1.0)
        progress_ratio = min(max(vp.current_time_sec / safe_dur, 0.0), 1.0)
        scaled_log_elapsed = min(math.log1p(max(0.0, vp.current_time_sec)) / math.log1p(60.0), 1.0)
        scaled_loops = min(float(vp.loop_count) / 5.0, 1.0)
        is_replaying = 1.0 if vp.loop_count > 0 else 0.0

        temporal_meta = torch.tensor(
            [
                progress_ratio,
                scaled_log_elapsed,
                scaled_loops,
                is_replaying,
            ],
            device=self.device,
            dtype=self.dtype,
        )

        fused = torch.cat([vis_emb, txt_emb, aud_emb, temporal_meta], dim=-1)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return EncodedTick(
            tick_index=raw_tick.tick_index,
            reel_id=vp.reel_id,
            elapsed_sec=vp.current_time_sec,
            duration_sec=vp.duration_sec,
            loop_count=vp.loop_count,
            is_liked=vp.is_liked,
            is_paused=vp.is_paused,
            wall_time=raw_tick.wall_time,
            unix_ts_ms=raw_tick.unix_ts_ms,
            fused_embedding=fused,
            latency_ms=latency_ms,
        )