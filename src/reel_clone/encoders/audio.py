from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from reel_clone.config import HardwareConfig


class AudioEncoder(nn.Module):
    """
    Frozen FP16 Audio Encoder mapping a 0.5s (8000-sample @ 16kHz) waveform
    into an L2-normalized (256,) acoustic/speech feature vector.
    """

    EMBED_DIM: int = 256

    def __init__(
        self,
        hw_config: HardwareConfig | None = None,
        use_wav2vec2: bool = True,
    ) -> None:
        super().__init__()
        self.hw = hw_config or HardwareConfig()
        self.device = torch.device(self.hw.device)
        self.dtype = (
            torch.float16
            if (self.hw.use_fp16 and self.device.type == "cuda")
            else torch.float32
        )
        self.use_wav2vec2 = use_wav2vec2

        # Fixed seeded projection from 768 (Wav2Vec2) or 64 (Mel) -> 256
        gen = torch.Generator().manual_seed(42)
        if self.use_wav2vec2:
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.backbone = bundle.get_model().eval().to(device=self.device, dtype=self.dtype)
            self.backbone.requires_grad_(False)
            proj_weight = torch.randn((768, self.EMBED_DIM), generator=gen) / (768 ** 0.5)
            self.register_buffer("proj", proj_weight.to(device=self.device, dtype=self.dtype))
            self.mel_spec = None
        else:
            self.backbone = None
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, hop_length=160, n_mels=64
            ).to(device=self.device)
            proj_weight = torch.randn((128, self.EMBED_DIM), generator=gen) / (128 ** 0.5)
            self.register_buffer("proj", proj_weight.to(device=self.device, dtype=self.dtype))

    @torch.inference_mode()
    def encode_audio(self, audio_pcm: NDArray[np.float32]) -> torch.Tensor:
        """Maps a 1D float32 numpy waveform (N,) to an L2-normalized (256,) tensor."""
        if audio_pcm.size == 0 or float(np.abs(audio_pcm).max()) < 1e-6:
            return torch.zeros((self.EMBED_DIM,), device=self.device, dtype=self.dtype)

        wav = torch.from_numpy(audio_pcm).unsqueeze(0).to(device=self.device)

        if self.use_wav2vec2 and self.backbone is not None:
            wav_cast = wav.to(dtype=self.dtype)
            features, _ = self.backbone(wav_cast)  # Shape: (1, T_frames, 768)
            pooled = features.mean(dim=1).squeeze(0)  # Shape: (768,)
            projected = pooled @ self.proj  # Shape: (256,)
        else:
            assert self.mel_spec is not None
            mel = torch.log(self.mel_spec(wav.float()) + 1e-6).squeeze(0)  # (64, T)
            stats = torch.cat([mel.mean(dim=-1), mel.std(dim=-1)], dim=0).to(dtype=self.dtype)  # (128,)
            projected = stats @ self.proj  # (256,)

        return F.normalize(projected.float(), dim=-1).to(dtype=self.dtype)