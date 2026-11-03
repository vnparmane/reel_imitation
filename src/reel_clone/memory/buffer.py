from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import torch

from reel_clone.config import StorageConfig
from reel_clone.encoders.fusion import EncodedTick


class TrajectoryBuffer:
    """Manages SQLite session history and compressed tensor replay buffers (.npz)."""

    def __init__(self, storage_config: StorageConfig | None = None) -> None:
        self.config = storage_config or StorageConfig()
        self.config.ensure_dirs()
        self._init_db()

    def _init_db(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with sqlite3.connect(self.config.db_path) as conn, open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

    def start_session(self, session_id: str) -> None:
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, start_time, total_reels) VALUES (?, datetime('now'), 0)",
                (session_id,),
            )

    def save_reel_trajectory(
        self,
        session_id: str,
        reel_id: str,
        ticks: list[EncodedTick],
        final_action: int,  # 0: CONTINUE, 1: SKIP, 2: LIKE
    ) -> Path:
        """Saves sequence ticks to an NPZ bundle and metadata to SQLite."""
        if not ticks:
            raise ValueError("Cannot save empty trajectory ticks.")

        duration = ticks[-1].duration_sec
        watch_time = ticks[-1].elapsed_sec
        loops = ticks[-1].loop_count

        # 1. Save metadata to SQLite
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO reels 
                   (reel_id, session_id, duration_sec, watch_time_sec, final_action, loop_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (reel_id, session_id, duration, watch_time, final_action, loops),
            )
            conn.execute(
                "UPDATE sessions SET total_reels = total_reels + 1 WHERE session_id = ?",
                (session_id,),
            )

        # 2. Pack sequence tensors into NumPy arrays
        embeddings = torch.stack([t.fused_embedding.float().cpu() for t in ticks]).numpy()
        actions = np.zeros((len(ticks),), dtype=np.int64)
        # All ticks except the last are CONTINUE (0); the last tick has the final action (SKIP=1 or LIKE=2)
        actions[:-1] = 0
        actions[-1] = final_action

        npz_path = self.config.sessions_dir / f"{session_id}_{reel_id[:16]}.npz"
        np.savez_compressed(
            npz_path,
            embeddings=embeddings,
            actions=actions,
            elapsed_sec=np.array([t.elapsed_sec for t in ticks]),
        )
        return npz_path

    def export_colab_bundle(self, session_id: str, output_path: Path) -> Path:
        """Bundles all reel NPZ files for a session into a single tar.gz for Google Colab T4."""
        import tarfile
        with tarfile.open(output_path, "w:gz") as tar:
            for npz_file in self.config.sessions_dir.glob(f"{session_id}_*.npz"):
                tar.add(npz_file, arcname=npz_file.name)
        return output_path