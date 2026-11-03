from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware budget tailored for i5-12HX + RTX 3050 (6GB VRAM) + Colab T4 sync."""
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True
    max_vram_gb: float = 3.2
    ocr_on_cpu: bool = True
    cpu_threads: int = 6


@dataclass(frozen=True)
class CaptureConfig:
    """Timing and viewport parameters for Stage 1 & 2 (Observation & Event Collection)."""
    tick_interval_sec: float = 0.5          # 2 Hz decision & observation clock
    ocr_every_n_ticks: int = 2              # Run OCR at 1 Hz (every 2nd tick)
    viewport_width: int = 1280
    viewport_height: int = 860
    user_data_dir: Path = Path("data/browser_profile")
    reels_url: str = "https://www.instagram.com/reels/"


@dataclass(frozen=True)
class StorageConfig:
    """Paths for SQLite interaction logger, NPZ replay buffer, and Colab T4 checkpoints."""
    data_dir: Path = Path("data")
    db_path: Path = Path("data/trajectories.db")
    sessions_dir: Path = Path("data/sessions")
    checkpoints_dir: Path = Path("checkpoints")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)