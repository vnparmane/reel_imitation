from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from reel_clone.config import StorageConfig, HardwareConfig
from reel_clone.policy.network import TemporalBehaviorPolicy
from reel_clone.policy.trainer import OnlinePolicyTrainer, compute_expected_watch_ticks


def test_watch_ticks_calibration_and_never_skip_penalty() -> None:
    immediate_skip_logits = torch.full((4,), 10.0)
    assert abs(compute_expected_watch_ticks(immediate_skip_logits) - 1.0) < 0.05

    never_skip_logits = torch.full((4,), -10.0)
    pred_ticks = compute_expected_watch_ticks(never_skip_logits, max_horizon_ticks=120)
    assert 103.0 <= pred_ticks <= 120.0


def test_temporal_policy_two_head_survival_and_eval_restoration(tmp_path: Path) -> None:
    storage_cfg = StorageConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        sessions_dir=tmp_path / "data" / "sessions",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    storage_cfg.ensure_dirs()

    for idx, length in enumerate([4, 9, 6, 12, 5, 8]):
        embs = np.random.randn(length, 1284).astype(np.float32)
        acts = np.zeros((length,), dtype=np.int64)
        if length > 3:
            acts[2] = 2
        acts[-1] = 3 if idx % 2 == 0 else 1
        np.savez_compressed(
            storage_cfg.sessions_dir / f"session_01_{idx:05d}_reel_{idx}.npz",
            embeddings=embs,
            actions=acts,
            elapsed_sec=np.arange(length, dtype=np.float32) * 0.5,
            censored=np.bool_(False),
        )

    hw = HardwareConfig()
    policy = TemporalBehaviorPolicy(hw_config=hw)
    trainer = OnlinePolicyTrainer(policy, storage_config=storage_cfg, hw_config=hw, lr=1e-3)

    fp16_seq = torch.randn(2, 7, 1284, device=hw.device, dtype=torch.float16)
    lengths = torch.tensor([3, 7], dtype=torch.long)
    step_logits, _ = policy(fp16_seq, lengths=lengths, return_sequence=False)
    seq_logits, _ = policy(fp16_seq, lengths=lengths, return_sequence=True)
    assert step_logits.shape == (2, 2)
    assert seq_logits.shape == (2, 7, 2)

    metrics = trainer.train_and_evaluate(batch_size=3)
    assert metrics["train_loss"] > 0.0
    assert metrics["val_loss"] > 0.0
    assert metrics["val_watch_mae_sec"] > 0.0
    assert not policy.training

    adapt_loss = trainer.online_adapt_step(recent_k=2, replay_k=2)
    assert adapt_loss > 0.0
    assert not policy.training