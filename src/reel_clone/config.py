from dataclasses import dataclass, field
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HardwareConfig:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True
    max_vram_gb: float = 3.2
    ocr_on_cpu: bool = True
    cpu_threads: int = 6


@dataclass(frozen=True)
class CaptureConfig:
    tick_interval_sec: float = 0.5
    ocr_every_n_ticks: int = 2
    viewport_width: int = 1280
    viewport_height: int = 860
    skip_cooldown_sec: float = 1.0
    online_update_every_n_reels: int = 5
    user_data_dir: Path = PROJECT_ROOT / "data" / "browser_profile"
    reels_url: str = "https://www.instagram.com/reels/"


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    db_path: Path = PROJECT_ROOT / "data" / "trajectories.db"
    sessions_dir: Path = PROJECT_ROOT / "data" / "sessions"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)