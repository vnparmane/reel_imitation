from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehavioralCloningLoss(nn.Module):
    IGNORE_INDEX: int = -100

    def __init__(
        self,
        like_pos_weight: float = 6.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "pos_weights",
            torch.tensor([1.0, like_pos_weight], dtype=torch.float32, device=device),
        )

    def forward(self, logits: torch.Tensor, bitmask_targets: torch.Tensor) -> torch.Tensor:
        valid_mask = bitmask_targets != self.IGNORE_INDEX
        if not valid_mask.any():
            return logits.sum() * 0.0

        safe_bits = bitmask_targets.clamp(min=0)
        skip_target = (safe_bits & 1).float()
        like_target = ((safe_bits & 2) >> 1).float()
        binary_targets = torch.stack([skip_target, like_target], dim=-1)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            binary_targets,
            pos_weight=self.pos_weights.to(logits.device),
            reduction="none",
        )
        return bce[valid_mask].mean()