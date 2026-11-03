from __future__ import annotations

import random
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from reel_clone.config import StorageConfig, HardwareConfig
from reel_clone.policy.loss import BehavioralCloningLoss
from reel_clone.policy.network import TemporalBehaviorPolicy


def _trajectory_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"_(\d{5})_", path.name)
    traj_id = int(m.group(1)) if m else 0
    return (traj_id, path.name)


class ReelTrajectoryDataset(Dataset):
    def __init__(self, files: list[Path] | Path) -> None:
        if isinstance(files, Path):
            self.npz_files = sorted(files.glob("*.npz"), key=_trajectory_sort_key)
        else:
            self.npz_files = list(files)

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
        with np.load(self.npz_files[idx]) as data:
            embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
            actions = torch.tensor(data["actions"], dtype=torch.long)
            censored = bool(data["censored"]) if "censored" in data else False
        length = int(embeddings.size(0))
        return embeddings, actions, length, censored


def collate_trajectories(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, bool]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings_list, actions_list, lengths_list, censored_list = zip(*batch)
    padded_embs = torch.nn.utils.rnn.pad_sequence(
        embeddings_list, batch_first=True, padding_value=0.0
    )
    padded_actions = torch.nn.utils.rnn.pad_sequence(
        actions_list, batch_first=True, padding_value=BehavioralCloningLoss.IGNORE_INDEX
    )
    lengths = torch.tensor(lengths_list, dtype=torch.long)
    censored = torch.tensor(censored_list, dtype=torch.bool)
    return padded_embs, padded_actions, lengths, censored


def compute_expected_watch_ticks(
    skip_logits: torch.Tensor,
    max_horizon_ticks: int = 120,
) -> float:
    hazards = torch.sigmoid(skip_logits.detach()).clamp(min=1e-4, max=1.0 - 1e-4)
    survivals = torch.cumprod(1.0 - hazards, dim=0)
    observed_after_first = float(survivals[:-1].sum().item()) if survivals.numel() > 1 else 0.0
    final_survival = float(survivals[-1].item())
    terminal_hazard = max(float(hazards[-1].item()), 0.01)
    tail_expected = final_survival / terminal_hazard
    return min(float(max_horizon_ticks), 1.0 + observed_after_first + tail_expected)


class OnlinePolicyTrainer:
    def __init__(
        self,
        policy: TemporalBehaviorPolicy,
        storage_config: StorageConfig | None = None,
        hw_config: HardwareConfig | None = None,
        lr: float = 1e-4,
        tick_interval_sec: float = 0.5,
    ) -> None:
        self.policy = policy
        self.storage = storage_config or StorageConfig()
        self.hw = hw_config or HardwareConfig()
        self.tick_interval_sec = tick_interval_sec
        self.criterion = BehavioralCloningLoss(device=str(self.policy.device))
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-3)

    def _list_ordered_files(self) -> list[Path]:
        return sorted(self.storage.sessions_dir.glob("*.npz"), key=_trajectory_sort_key)

    def _run_batches(self, loader: DataLoader, train: bool, compute_mae: bool = False) -> tuple[float, float]:
        self.policy.train(mode=train)
        total_loss = 0.0
        total_watch_mae_sec = 0.0
        total_uncensored = 0

        context = torch.enable_grad() if train else torch.inference_mode()
        try:
            with context:
                for seqs, targets, lengths, censored in loader:
                    seqs = seqs.to(self.policy.device)
                    targets = targets.to(self.policy.device)
                    lengths = lengths.to(self.policy.device)

                    if train:
                        self.optimizer.zero_grad()

                    seq_logits, _ = self.policy(seqs, lengths=lengths, return_sequence=True)
                    loss = self.criterion(seq_logits, targets)

                    if train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                        self.optimizer.step()

                    total_loss += float(loss.item())

                    if compute_mae:
                        for b in range(seqs.size(0)):
                            if bool(censored[b].item()):
                                continue
                            true_len = int(lengths[b].item())
                            prefix_len = min(4, true_len)
                            pred_ticks = compute_expected_watch_ticks(seq_logits[b, :prefix_len, 0])
                            total_watch_mae_sec += abs(pred_ticks - true_len) * self.tick_interval_sec
                            total_uncensored += 1
        finally:
            self.policy.eval()

        avg_loss = total_loss / max(1, len(loader))
        avg_mae = total_watch_mae_sec / max(1, total_uncensored) if compute_mae else 0.0
        return avg_loss, avg_mae

    def train_epoch(self, batch_size: int = 16) -> float:
        return self.train_and_evaluate(batch_size=batch_size)["train_loss"]

    def train_and_evaluate(self, batch_size: int = 16) -> dict[str, float]:
        all_files = self._list_ordered_files()
        if not all_files:
            return {"train_loss": 0.0, "val_loss": 0.0, "val_watch_mae_sec": 0.0}

        if len(all_files) >= 2:
            val_count = max(1, int(round(len(all_files) * 0.2)))
            train_files = all_files[:-val_count]
            val_files = all_files[-val_count:]
        else:
            train_files = all_files
            val_files = []

        train_loader = DataLoader(
            ReelTrajectoryDataset(train_files),
            batch_size=min(batch_size, len(train_files)),
            shuffle=True,
            collate_fn=collate_trajectories,
        )
        train_loss, train_mae = self._run_batches(train_loader, train=True, compute_mae=not val_files)

        if val_files:
            val_loader = DataLoader(
                ReelTrajectoryDataset(val_files),
                batch_size=min(batch_size, len(val_files)),
                shuffle=False,
                collate_fn=collate_trajectories,
            )
            val_loss, val_mae = self._run_batches(val_loader, train=False, compute_mae=True)
        else:
            val_loss, val_mae = train_loss, train_mae

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_watch_mae_sec": val_mae,
        }

    def online_adapt_step(self, recent_k: int = 6, replay_k: int = 6, steps: int = 3) -> float:
        all_files = self._list_ordered_files()
        if not all_files:
            return 0.0

        recent = all_files[-recent_k:]
        older = all_files[:-recent_k]
        replay = random.sample(older, min(len(older), replay_k)) if older else []
        selected = recent + replay

        loader = DataLoader(
            ReelTrajectoryDataset(selected),
            batch_size=len(selected),
            shuffle=True,
            collate_fn=collate_trajectories,
        )
        loss = 0.0
        for _ in range(max(1, steps)):
            loss, _ = self._run_batches(loader, train=True, compute_mae=False)
        self.policy.eval()
        self.save_checkpoint("latest_policy.pt")
        return loss

    def save_checkpoint(self, filename: str = "latest_policy.pt") -> Path:
        self.storage.ensure_dirs()
        path = self.storage.checkpoints_dir / filename
        torch.save(self.policy.state_dict(), path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=self.policy.device, weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()