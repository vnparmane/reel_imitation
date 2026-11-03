from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import torch

from reel_clone.config import StorageConfig
from reel_clone.encoders.fusion import EncodedTick
from reel_clone.memory.buffer import TrajectoryBuffer


def test_trajectory_buffer_preserves_mid_reel_likes_and_censoring(tmp_path: Path) -> None:
    storage_cfg = StorageConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        sessions_dir=tmp_path / "data" / "sessions",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    buffer = TrajectoryBuffer(storage_cfg)
    buffer.start_session("session_test_01")

    ticks = [
        EncodedTick(
            tick_index=i,
            reel_id="blob:https://www.instagram.com/9999-aaaa",
            elapsed_sec=i * 0.5,
            duration_sec=10.0,
            loop_count=0,
            is_liked=(i >= 2),
            is_paused=False,
            wall_time=10.0 + i * 0.5,
            unix_ts_ms=1700000000000.0 + i * 500.0,
            fused_embedding=torch.randn(1284),
            latency_ms=15.0,
        )
        for i in range(5)
    ]

    npz_1 = buffer.save_reel_trajectory(
        session_id="session_test_01",
        reel_id="blob:https://www.instagram.com/9999-aaaa",
        ticks=ticks,
        tick_actions=[0, 0, 2, 0, 1],
        censored=False,
    )

    npz_2 = buffer.save_reel_trajectory(
        session_id="session_test_01",
        reel_id="blob:https://www.instagram.com/9999-aaaa",
        ticks=ticks,
        tick_actions=[0, 0, 0, 0, 1],
        censored=True,
    )

    assert npz_1.exists() and npz_2.exists()
    assert npz_1 != npz_2

    with np.load(npz_1) as d1:
        assert list(d1["actions"]) == [0, 0, 2, 0, 1]
        assert bool(d1["censored"]) is False

    with np.load(npz_2) as d2:
        assert list(d2["actions"]) == [0, 0, 0, 0, 0]
        assert bool(d2["censored"]) is True

    with sqlite3.connect(storage_cfg.db_path) as conn:
        rows = conn.execute("SELECT liked, censored FROM reels ORDER BY trajectory_id").fetchall()
        assert rows == [(1, 0), (0, 1)]