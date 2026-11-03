# src/reel_clone/policy/network.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from reel_clone.config import HardwareConfig
from reel_clone.encoders.fusion import MultimodalFeatureExtractor


class TemporalBehaviorPolicy(nn.Module):
    """
    Two-Head Causal GRU Policy:
      - Head 0 (skip_hazard_logit): instantaneous hazard h_t = P(skip at t | survived to t)
      - Head 1 (like_logit): instantaneous probability l_t = P(like at t)
    """

    NUM_HEADS: int = 2  # [skip_hazard_logit, like_logit]

    def __init__(
        self,
        input_dim: int = MultimodalFeatureExtractor.FUSED_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
        hw_config: HardwareConfig | None = None,
    ) -> None:
        super().__init__()
        self.hw = hw_config or HardwareConfig()
        self.device = torch.device(self.hw.device)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, self.NUM_HEADS)
        self.to(self.device)

    def forward(
        self,
        seq_features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = seq_features.to(device=self.device, dtype=self.head.weight.dtype)

        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, next_hidden = self.gru(packed, hidden)
            out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
        else:
            out, next_hidden = self.gru(x, hidden)

        if return_sequence:
            return self.head(out), next_hidden

        if lengths is not None:
            batch_indices = torch.arange(out.size(0), device=out.device)
            last_indices = (lengths.to(out.device) - 1).clamp(min=0)
            last_hidden = out[batch_indices, last_indices, :]
        else:
            last_hidden = out[:, -1, :]

        return self.head(last_hidden), next_hidden