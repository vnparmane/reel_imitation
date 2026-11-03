from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from reel_clone.config import StorageConfig, HardwareConfig
from reel_clone.policy.network import TemporalBehaviorPolicy
from reel_clone.policy.trainer import OnlinePolicyTrainer


def test_temporal_policy_forward_and_trainer(tmp_path: Path) -> None:
    storage_cfg = StorageConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        sessions_dir=tmp_path / "data" / "sessions",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    storage_cfg.ensure_dirs()

    # Create synthetic NPZ demonstration file
    dummy_embeddings = np.random.randn(8, 1284).astype(np.float32)
    dummy_actions = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)  # SKIP at end
    np.savez_compressed(
        storage_cfg.sessions_dir / "session_dummy_reel_1.npz",
        embeddings=dummy_embeddings,
        actions=dummy_actions,
        elapsed_sec=np.linspace(0, 3.5, 8),
    )

    hw = HardwareConfig()
    policy = TemporalBehaviorPolicy(hw_config=hw)
    trainer = OnlinePolicyTrainer(policy, storage_config=storage_cfg, lr=1e-3)

    # Test forward pass shape
    seq = torch.randn(2, 10, 1284, device=hw.device)
    logits, next_hidden = policy(seq)
    assert logits.shape == (2, 3)

    # Test training epoch update
    loss_val = trainer.train_epoch(batch_size=2)
    assert loss_val > 0.0

    # Test checkpoint saving & loading
    ckpt_path = trainer.save_checkpoint("test_policy.pt")
    assert ckpt_path.exists()
    trainer.load_checkpoint(ckpt_path)