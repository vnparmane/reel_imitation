from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from reel_clone.config import StorageConfig, HardwareConfig
from reel_clone.encoders.fusion import EncodedTick
from reel_clone.memory.buffer import TrajectoryBuffer
from reel_clone.policy.network import TemporalBehaviorPolicy
from reel_clone.policy.trainer import OnlinePolicyTrainer


@pytest.mark.asyncio
async def test_end_to_end_pipeline_flow(tmp_path: Path) -> None:
    storage_cfg = StorageConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        sessions_dir=tmp_path / "data" / "sessions",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    storage_cfg.ensure_dirs()
    hw = HardwareConfig()

    # 1. Simulate Trajectory Logging (Mode A)
    buffer = TrajectoryBuffer(storage_cfg)
    buffer.start_session("integration_session")

    ticks = [
        EncodedTick(
            tick_index=i,
            reel_id="reel_integration_test",
            elapsed_sec=i * 0.5,
            duration_sec=5.0,
            loop_count=0,
            is_liked=False,
            fused_embedding=torch.randn(1284, dtype=torch.float32),
            latency_ms=10.0,
        )
        for i in range(4)
    ]
    buffer.save_reel_trajectory("integration_session", "reel_integration_test", ticks, final_action=1)

    # 2. Simulate Online Training (Train Mode)
    policy = TemporalBehaviorPolicy(hw_config=hw)
    trainer = OnlinePolicyTrainer(policy, storage_config=storage_cfg, hw_config=hw)
    loss = trainer.train_epoch(batch_size=1)
    assert loss > 0.0

    ckpt = trainer.save_checkpoint("integration_policy.pt")
    assert ckpt.exists()

    # 3. Simulate Autonomous Policy Inference (Mode B)
    policy.eval()
    with torch.inference_mode():
        sample_input = ticks[0].fused_embedding.unsqueeze(0).unsqueeze(0).to(hw.device)
        logits, _ = policy(sample_input)
        assert logits.shape == (1, 3)
        action = int(torch.argmax(logits, dim=-1).item())
        assert action in [0, 1, 2]