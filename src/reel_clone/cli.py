from __future__ import annotations

import argparse
import asyncio
import copy
import time
import torch

from reel_clone.actor.executor import ActionExecutor
from reel_clone.capture.sync_stream import SynchronizedCaptureStream
from reel_clone.capture.viewport import ReelViewportController
from reel_clone.config import CaptureConfig, HardwareConfig, StorageConfig
from reel_clone.encoders.fusion import EncodedTick, MultimodalFeatureExtractor
from reel_clone.memory.buffer import TrajectoryBuffer
from reel_clone.observer.human_listener import HumanActionListener
from reel_clone.policy.network import TemporalBehaviorPolicy
from reel_clone.policy.trainer import OnlinePolicyTrainer


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    return 0.0


def _finalize_and_save_reel(
    buffer: TrajectoryBuffer,
    session_id: str,
    reel_id: str,
    ticks: list[EncodedTick],
    tick_actions: list[int],
    transition_occurred: bool,
) -> None:
    if not ticks:
        return
    if transition_occurred and tick_actions:
        # Set the SKIP bit (bit 0) on the outgoing reel's final tick while preserving any LIKE bit (bit 1)
        tick_actions[-1] = tick_actions[-1] | 1
    buffer.save_reel_trajectory(
        session_id=session_id,
        reel_id=reel_id,
        ticks=ticks,
        tick_actions=tick_actions,
    )


async def process_single_tick(
    encoded: EncodedTick,
    is_autonomous: bool,
    listener: HumanActionListener,
    buffer: TrajectoryBuffer,
    actor: ActionExecutor,
    policy: TemporalBehaviorPolicy,
    state: dict,
    session_id: str,
    inertia_grace_sec: float = 0.55,
) -> None:
    """
    Processes a single synchronized tick. Extracted so integration tests can verify
    exact label attribution across reel boundaries and F8 mode switches.
    """
    # 1. Drain queue FIRST before checking reel transition
    events = listener.drain_actions()
    if is_autonomous:
        events = []  # Never let synthetic autonomous keys leak into demonstration labels

    current_reel_id: str | None = state["current_reel_id"]
    reel_changed = (current_reel_id is not None) and (encoded.reel_id != current_reel_id)

    # 2. On reel boundary, attribute any queued SKIP to the outgoing reel and strip it from the new reel
    if reel_changed:
        if 1 in events and state["current_actions"]:
            state["current_actions"][-1] = state["current_actions"][-1] | 1
        events = [e for e in events if e != 1]

        if state["current_ticks"] and current_reel_id is not None:
            _finalize_and_save_reel(
                buffer,
                session_id,
                current_reel_id,
                state["current_ticks"],
                state["current_actions"],
                transition_occurred=True,
            )
            state["completed_demo_reels"] += 1
            state["current_ticks"] = []
            state["current_actions"] = []

        state["hidden_state"] = None
        state["survival_prob"] = 1.0
        state["already_liked_current"] = encoded.is_liked
        state["current_reel_id"] = encoded.reel_id
        state["last_transition_wall"] = time.perf_counter()
    elif current_reel_id is None:
        state["current_reel_id"] = encoded.reel_id
        state["already_liked_current"] = encoded.is_liked
        state["last_transition_wall"] = time.perf_counter()

    # 3. Suppress trackpad wheel-inertia SKIPs immediately after entering a new reel
    if (time.perf_counter() - state["last_transition_wall"]) < inertia_grace_sec:
        events = [e for e in events if e != 1]

    if not is_autonomous:
        action_mask = listener.resolve_tick_action(
            events=events,
            already_liked_reel=state["already_liked_current"],
            dom_liked_now=encoded.is_liked,
        )
        if (action_mask & 2) != 0:
            state["already_liked_current"] = True

        state["current_ticks"].append(encoded)
        state["current_actions"].append(action_mask)
    else:
        # Flush partial Mode A trajectory when toggling F8 into Mode B
        if state["current_ticks"] and state["current_reel_id"] is not None:
            _finalize_and_save_reel(
                buffer,
                session_id,
                state["current_reel_id"],
                state["current_ticks"],
                state["current_actions"],
                transition_occurred=False,
            )
            state["current_ticks"] = []
            state["current_actions"] = []

        feat_tensor = encoded.fused_embedding.float().unsqueeze(0).unsqueeze(0).to(policy.device)
        with torch.inference_mode():
            logits, state["hidden_state"] = policy(feat_tensor, hidden=state["hidden_state"])
            skip_hazard = float(torch.sigmoid(logits[0, 0]).item())
            like_prob = float(torch.sigmoid(logits[0, 1]).item())

        state["survival_prob"] *= (1.0 - skip_hazard)
        should_like = (like_prob >= 0.5) and (not state["already_liked_current"])
        should_skip = (state["survival_prob"] < 0.5) or (skip_hazard >= 0.5)

        if should_like:
            liked_ok = await actor.execute(2)
            if liked_ok:
                state["already_liked_current"] = True

        if should_skip:
            skipped_ok = await actor.execute(1)
            if skipped_ok:
                state["survival_prob"] = 1.0