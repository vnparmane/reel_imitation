from __future__ import annotations

from pathlib import Path
import torch

from reel_clone.config import StorageConfig
from reel_clone.encoders.fusion import EncodedTick
from reel_clone.memory.buffer import TrajectoryBuffer


def test_trajectory_buffer_sqlite_and_npz(tmp_path: Path) -> None:
    storage_cfg = StorageConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "test.db",
        sessions_dir=tmp_path / "data" / "sessions",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    buffer = TrajectoryBuffer(storage_cfg)
    buffer.start_session("session_test_01")

    # Create dummy encoded ticks
    ticks = [
        EncodedTick(
            tick_index=i,
            reel_id="reel_abc123",
            elapsed_sec=i * 0.5,
            duration_sec=10.0,
            loop_count=0,
            is_liked=False,
            fused_embedding=torch.randn(1284),
            latency_ms=15.0,
        )
        for i in range(5)
    ]

    npz_path = buffer.save_reel_trajectory(
        session_id="session_test_01",
        reel_id="reel_abc123",
        ticks=ticks,
        final_action=1,  # SKIP
    )

    assert npz_path.exists()

    # Verify SQLite record
    import sqlite3
    with sqlite3.connect(storage_cfg.db_path) as conn:
        cursor = conn.execute("SELECT total_reels FROM sessions WHERE session_id = 'session_test_01'")
        assert cursor.fetchone()[0] == 1

        cursor = conn.execute("SELECT final_action, watch_time_sec FROM reels WHERE reel_id = 'reel_abc123'")
        row = cursor.fetchone()
        assert row[0] == 1
        assert row[1] == 2.0  # 4 * 0.5s

    # Verify Colab bundle export
    tar_path = storage_cfg.data_dir / "colab_bundle.tar.gz"
    exported = buffer.export_colab_bundle("session_test_01", tar_path)
    assert exported.exists()