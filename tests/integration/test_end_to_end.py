from __future__ import annotations

from pathlib import Path
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

    buffer = TrajectoryBuffer(storage_cfg)
    buffer.start_session("integration_session")

    ticks = [
        EncodedTick(
            tick_index=i,
            reel_id="blob:https://www.instagram.com/integration-reel-1",
            elapsed_sec=i * 0.5,
            duration_sec=5.0,
            loop_count=0,
            is_liked=(i >= 1),
            is_paused=False,
            wall_time=100.0 + i * 0.5,
            unix_ts_ms=1700000000000.0 + i * 500.0,
            fused_embedding=torch.randn(1284, dtype=torch.float16),
            latency_ms=10.0,
        )
        for i in range(4)
    ]
    buffer.save_reel_trajectory(
        "integration_session",
        "blob:https://www.instagram.com/integration-reel-1",
        ticks,
        tick_actions=[0, 2, 0, 1],
    )

    policy = TemporalBehaviorPolicy(hw_config=hw)
    trainer = OnlinePolicyTrainer(policy, storage_config=storage_cfg, hw_config=hw)
    loss = trainer.train_epoch(batch_size=1)
    assert loss > 0.0

    ckpt = trainer.save_checkpoint("integration_policy.pt")
    assert ckpt.exists()

    trainer.load_checkpoint(ckpt)
    assert not policy.training
    with torch.inference_mode():
        sample_input = ticks[0].fused_embedding.unsqueeze(0).unsqueeze(0).to(hw.device)
        logits, _ = policy(sample_input)
        assert logits.shape == (1, 2)