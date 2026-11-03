# src/reel_clone/policy/loss.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehavioralCloningLoss(nn.Module):
    """
    Two-Head Binary Cross-Entropy Loss over packed sequences:
      - Channel 0: Skip Hazard loss (pos_weight balances single terminal skip vs continue ticks)
      - Channel 1: Like event loss (pos_weight balances rare mid-reel likes)
    """

    IGNORE_INDEX: int = -100

    def __init__(
        self,
        skip_pos_weight: float = 8.0,
        like_pos_weight: float = 10.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "pos_weights",
            torch.tensor([skip_pos_weight, like_pos_weight], dtype=torch.float32, device=device),
        )

    def forward(self, logits: torch.Tensor, bitmask_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, T, 2) or (B, 2) tensor of [skip_hazard_logit, like_logit]
            bitmask_targets: (B, T) or (B,) integer bitmask (0=CONT, 1=SKIP, 2=LIKE, 3=BOTH, -100=PAD)
        """
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
        masked_bce = bce[valid_mask]
        return masked_bce.mean()