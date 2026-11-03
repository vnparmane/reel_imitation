from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from reel_clone.config import HardwareConfig


class VisionTextEncoder(nn.Module):
    """
    Frozen FP16 CLIP encoder (ViT-B-32) that extracts both visual scene embeddings (512-d)
    and OCR text embeddings (512-d) in a shared semantic space on the RTX 3050.
    """

    EMBED_DIM: int = 512

    def __init__(
        self,
        hw_config: HardwareConfig | None = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        mock_mode: bool = False,
    ) -> None:
        super().__init__()
        self.hw = hw_config or HardwareConfig()
        self.device = torch.device(self.hw.device)
        self.dtype = (
            torch.float16
            if (self.hw.use_fp16 and self.device.type == "cuda")
            else torch.float32
        )
        self.mock_mode = mock_mode

        if not self.mock_mode:
            import open_clip  # type: ignore[import-untyped]

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.model = model.eval().to(device=self.device, dtype=self.dtype)
            self.model.requires_grad_(False)
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            self.model = None
            self.preprocess = None
            self.tokenizer = None

        # Cache the last OCR text embedding so unchanged captions cost 0 GPU cycles
        self._last_text: str | None = None
        self._last_text_emb: torch.Tensor | None = None

    @torch.inference_mode()
    def encode_frame(self, frame_rgb: NDArray[np.uint8]) -> torch.Tensor:
        """Maps an (H, W, 3) RGB uint8 frame to an L2-normalized (512,) tensor."""
        if self.mock_mode or self.model is None:
            # Deterministic lightweight projection from image mean/std for fast unit tests
            mean_val = float(frame_rgb.mean()) / 255.0 if frame_rgb.size > 0 else 0.0
            vec = torch.full((self.EMBED_DIM,), mean_val, device=self.device, dtype=self.dtype)
            vec[0] = 1.0
            return F.normalize(vec, dim=-1)

        pil_img = Image.fromarray(frame_rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        vis_emb = self.model.encode_image(img_tensor).squeeze(0)
        return F.normalize(vis_emb.float(), dim=-1).to(dtype=self.dtype)

    @torch.inference_mode()
    def encode_ocr_text(self, ocr_text: str) -> torch.Tensor:
        """Maps OCR overlay string to an L2-normalized (512,) tensor with caching."""
        clean_text = ocr_text.strip()
        if clean_text == self._last_text and self._last_text_emb is not None:
            return self._last_text_emb

        if not clean_text:
            zero_emb = torch.zeros((self.EMBED_DIM,), device=self.device, dtype=self.dtype)
            self._last_text = clean_text
            self._last_text_emb = zero_emb
            return zero_emb

        if self.mock_mode or self.model is None or self.tokenizer is None:
            seed_val = float(len(clean_text) % 17) / 17.0
            vec = torch.full((self.EMBED_DIM,), seed_val, device=self.device, dtype=self.dtype)
            vec[1] = 1.0
            normed = F.normalize(vec, dim=-1)
            self._last_text = clean_text
            self._last_text_emb = normed
            return normed

        tokens = self.tokenizer([clean_text[:250]]).to(self.device)
        txt_emb = self.model.encode_text(tokens).squeeze(0)
        normed = F.normalize(txt_emb.float(), dim=-1).to(dtype=self.dtype)

        self._last_text = clean_text
        self._last_text_emb = normed
        return normed