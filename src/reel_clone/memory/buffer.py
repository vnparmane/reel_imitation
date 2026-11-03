from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import torch

from reel_clone.capture.viewport import sanitize_reel_id
from reel_clone.config import StorageConfig
from reel_clone.encoders.fusion import EncodedTick


class TrajectoryBuffer:
    def __init__(self, storage_config: StorageConfig | None = None) -> None:
        self.config = storage_config or StorageConfig()
        self.config.ensure_dirs()
        self._init_db()

    def _init_db(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Missing SQLite schema file at {schema_path}")
        with sqlite3.connect(self.config.db_path) as conn, open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

    def start_session(self, session_id: str) -> None:
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, start_time, total_reels) VALUES (?, datetime('now'), 0)",
                (session_id,),
            )

    @staticmethod
    def compute_active_watch_sec(ticks: list[EncodedTick], default_dt: float = 0.5) -> float:
        if not ticks:
            return 0.0
        total = 0.0
        for i, t in enumerate(ticks):
            if t.is_paused:
                continue
            if i == 0:
                total += default_dt
            else:
                dt = max(0.0, min(t.wall_time - ticks[i - 1].wall_time, default_dt * 3.0))
                total += dt if dt > 0.0 else default_dt
        return total

    def save_reel_trajectory(
        self,
        session_id: str,
        reel_id: str,
        ticks: list[EncodedTick],
        final_action: int | None = None,
        tick_actions: list[int] | None = None,
        censored: bool = False,
    ) -> Path:
        if not ticks:
            raise ValueError("Cannot save empty trajectory ticks.")

        safe_id = sanitize_reel_id(reel_id)
        duration = ticks[-1].duration_sec
        watch_time = self.compute_active_watch_sec(ticks)
        loops = ticks[-1].loop_count

        if tick_actions is not None and len(tick_actions) == len(ticks):
            actions = np.array(tick_actions, dtype=np.int64)
            resolved_final_action = int(actions[-1])
        else:
            resolved_final_action = (0 if censored else 1) if final_action is None else int(final_action)
            actions = np.zeros((len(ticks),), dtype=np.int64)
            actions[-1] = resolved_final_action

        if censored:
            actions[-1] = int(actions[-1]) & (~1)
            resolved_final_action = int(actions[-1])

        liked_flag = int(any((int(a) & 2) != 0 for a in actions))
        censored_flag = 1 if censored else 0

        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO reels
                   (reel_id, session_id, duration_sec, watch_time_sec, final_action, liked, censored, loop_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (safe_id, session_id, duration, watch_time, resolved_final_action, liked_flag, censored_flag, loops),
            )
            traj_id = int(cursor.lastrowid or 0)
            npz_name = f"{session_id}_{traj_id:05d}_{safe_id}.npz"
            conn.execute(
                "UPDATE reels SET npz_filename = ? WHERE trajectory_id = ?",
                (npz_name, traj_id),
            )
            conn.execute(
                "UPDATE sessions SET total_reels = total_reels + 1, end_time = datetime('now') WHERE session_id = ?",
                (session_id,),
            )

        embeddings = torch.stack([t.fused_embedding.float().cpu() for t in ticks]).numpy()
        npz_path = self.config.sessions_dir / npz_name
        np.savez_compressed(
            npz_path,
            embeddings=embeddings,
            actions=actions,
            elapsed_sec=np.array([t.elapsed_sec for t in ticks], dtype=np.float32),
            active_watch_sec=np.float32(watch_time),
            duration_sec=np.float32(duration),
            loop_count=np.int64(loops),
            censored=np.bool_(censored),
        )
        return npz_path

    def export_colab_bundle(self, session_id: str, output_path: Path) -> Path:
        import tarfile
        with tarfile.open(output_path, "w:gz") as tar:
            for npz_file in sorted(self.config.sessions_dir.glob(f"{session_id}_*.npz")):
                tar.add(npz_file, arcname=npz_file.name)
        return output_path