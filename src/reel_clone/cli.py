from __future__ import annotations

import argparse
import asyncio
import copy
import random
import torch

from reel_clone.actor.executor import ActionExecutor
from reel_clone.capture.sync_stream import SynchronizedCaptureStream
from reel_clone.capture.viewport import ReelViewportController
from reel_clone.config import CaptureConfig, HardwareConfig, StorageConfig
from reel_clone.encoders.fusion import EncodedTick, MultimodalFeatureExtractor
from reel_clone.memory.buffer import TrajectoryBuffer
from reel_clone.observer.human_listener import ActionEvent, HumanActionListener
from reel_clone.policy.network import TemporalBehaviorPolicy
from reel_clone.policy.trainer import OnlinePolicyTrainer


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    return 0.0


def _action_name(mask: int) -> str:
    if mask == 3:
        return "LIKE+SKIP"
    if mask == 2:
        return "LIKE"
    if mask == 1:
        return "SKIP"
    return "CONTINUE"


def _finalize_and_save_reel(
    buffer: TrajectoryBuffer,
    session_id: str,
    reel_id: str,
    ticks: list[EncodedTick],
    tick_actions: list[int],
    explicit_skip: bool,
    censored: bool,
) -> None:
    if not ticks:
        return
    if explicit_skip and not censored and tick_actions:
        tick_actions[-1] = int(tick_actions[-1]) | 1
    buffer.save_reel_trajectory(
        session_id=session_id,
        reel_id=reel_id,
        ticks=ticks,
        tick_actions=tick_actions,
        censored=censored,
    )


def make_initial_loop_state() -> dict:
    return {
        "current_ticks": [],
        "current_actions": [],
        "current_reel_id": None,
        "already_liked_current": False,
        "hidden_state": None,
        "survival_prob": 1.0,
        "target_survival_threshold": random.uniform(0.20, 0.80),
        "completed_demo_reels": 0,
        "last_transition_wall": -999.0,
    }


async def process_single_tick(
    encoded: EncodedTick,
    is_autonomous: bool,
    listener: HumanActionListener,
    buffer: TrajectoryBuffer,
    actor: ActionExecutor,
    policy: TemporalBehaviorPolicy,
    state: dict,
    session_id: str,
    inertia_grace_sec: float = 0.45,
) -> int:
    """
    Processes a single synchronized tick with event-timestamp attribution,
    right-censoring for non-skip transitions, and stochastic survival sampling in Mode B.
    """
    # 1. Drain timestamped events FIRST
    raw_events: list[ActionEvent] = listener.drain_events()
    if is_autonomous:
        raw_events = []  # Never allow synthetic Mode B actions into Mode A queue

    current_reel_id: str | None = state["current_reel_id"]
    reel_changed = (current_reel_id is not None) and (encoded.reel_id != current_reel_id)

    # 2. Handle reel boundary transitions using event timestamps and explicit skip/censoring rules
    if reel_changed:
        prev_ticks: list[EncodedTick] = state["current_ticks"]
        prev_actions: list[int] = state["current_actions"]

        # Midpoint timestamp between outgoing reel's last tick and incoming reel's first tick
        if prev_ticks:
            boundary_ts_ms = 0.5 * (prev_ticks[-1].unix_ts_ms + encoded.unix_ts_ms)
        else:
            boundary_ts_ms = encoded.unix_ts_ms

        outgoing_codes: list[int] = []
        incoming_events: list[ActionEvent] = []

        for ev in raw_events:
            # Skip or scroll-up events that caused the transition (or any event before boundary_ts_ms)
            # belong to the outgoing reel
            if ev.action_type in (1, 8) or ev.timestamp_ms <= boundary_ts_ms:
                outgoing_codes.append(ev.action_type)
            else:
                incoming_events.append(ev)

        if prev_ticks and current_reel_id is not None:
            # Apply any late LIKE/UNLIKE/SKIP that occurred on the outgoing reel prior to transition
            out_mask, _ = listener.resolve_tick_action(
                outgoing_codes,
                already_liked_reel=state["already_liked_current"],
            )
            prev_actions[-1] = int(prev_actions[-1]) | out_mask

            # Did the user explicitly skip down (1), or did they scroll up (8) / reload / navigate?
            explicit_skip = (1 in outgoing_codes) or ((prev_actions[-1] & 1) != 0)
            scrolled_up = (8 in outgoing_codes) and not explicit_skip
            is_censored = scrolled_up or (not explicit_skip)

            _finalize_and_save_reel(
                buffer=buffer,
                session_id=session_id,
                reel_id=current_reel_id,
                ticks=prev_ticks,
                tick_actions=prev_actions,
                explicit_skip=explicit_skip,
                censored=is_censored,
            )
            state["completed_demo_reels"] += 1
            state["current_ticks"] = []
            state["current_actions"] = []

        # Reset recurrent state and draw a fresh stochastic survival threshold for the new reel
        state["hidden_state"] = None
        state["survival_prob"] = 1.0
        state["target_survival_threshold"] = random.uniform(0.20, 0.80)
        state["already_liked_current"] = encoded.is_liked
        state["current_reel_id"] = encoded.reel_id
        state["last_transition_wall"] = encoded.wall_time
        raw_events = incoming_events

    elif current_reel_id is None:
        state["current_reel_id"] = encoded.reel_id
        state["already_liked_current"] = encoded.is_liked
        state["last_transition_wall"] = encoded.wall_time

    # 3. Filter residual trackpad wheel-inertia SKIPs using the tick's wall_time
    elapsed_since_transition = encoded.wall_time - float(state["last_transition_wall"])
    if 0.0 <= elapsed_since_transition < inertia_grace_sec:
        event_codes = [ev.action_type for ev in raw_events if ev.action_type != 1]
    else:
        event_codes = [ev.action_type for ev in raw_events]

    # 4. Mode A: Record human demonstration tick
    if not is_autonomous:
        action_mask, updated_liked = listener.resolve_tick_action(
            events=event_codes,
            already_liked_reel=state["already_liked_current"],
        )
        state["already_liked_current"] = updated_liked
        state["current_ticks"].append(encoded)
        state["current_actions"].append(action_mask)
        return action_mask

    # 5. Mode B: Autonomous Imitation
    # If switching from Mode A -> Mode B mid-reel via F8, save the partial trajectory as right-censored
    if state["current_ticks"] and state["current_reel_id"] is not None:
        _finalize_and_save_reel(
            buffer=buffer,
            session_id=session_id,
            reel_id=state["current_reel_id"],
            ticks=state["current_ticks"],
            tick_actions=state["current_actions"],
            explicit_skip=False,
            censored=True,
        )
        state["current_ticks"] = []
        state["current_actions"] = []

    feat_tensor = encoded.fused_embedding.float().unsqueeze(0).unsqueeze(0).to(policy.device)
    policy.eval()
    with torch.inference_mode():
        logits, state["hidden_state"] = policy(feat_tensor, hidden=state["hidden_state"])
        skip_hazard = float(torch.sigmoid(logits[0, 0]).item())
        like_prob = float(torch.sigmoid(logits[0, 1]).item())

    state["survival_prob"] *= (1.0 - skip_hazard)
    threshold = float(state.get("target_survival_threshold", 0.5))

    should_like = (like_prob >= 0.5) and (not state["already_liked_current"])
    should_skip = (state["survival_prob"] < threshold) or (skip_hazard >= 0.5)

    executed_mask = 0
    if should_like:
        if await actor.execute(2):
            state["already_liked_current"] = True
            executed_mask |= 2

    if should_skip:
        if await actor.execute(1):
            state["survival_prob"] = 1.0
            state["target_survival_threshold"] = random.uniform(0.20, 0.80)
            executed_mask |= 1

    return executed_mask


async def run_hybrid_f8_loop(session_id: str, start_autonomous: bool = False) -> None:
    print(f"\n[Reel-Clone] Launching Session ({session_id})...")
    hw = HardwareConfig()
    storage = StorageConfig()
    capture_cfg = CaptureConfig()

    viewport = ReelViewportController(capture_cfg)
    await viewport.start(headless=False)

    stream = SynchronizedCaptureStream(viewport_ctrl=viewport, config=capture_cfg)
    extractor = MultimodalFeatureExtractor(hw_config=hw)
    buffer = TrajectoryBuffer(storage)
    buffer.start_session(session_id)

    listener = HumanActionListener(viewport.page)
    await listener.install_listeners()
    actor = ActionExecutor(viewport.page, skip_cooldown_sec=capture_cfg.skip_cooldown_sec)

    policy = TemporalBehaviorPolicy(hw_config=hw)
    trainer = OnlinePolicyTrainer(policy, storage_config=storage, hw_config=hw)
    ckpt_path = storage.checkpoints_dir / "latest_policy.pt"
    if ckpt_path.exists():
        trainer.load_checkpoint(ckpt_path)
        print(f"[Policy] Loaded weights from {ckpt_path}")

    policy.eval()

    init_flag = "true" if start_autonomous else "false"
    toggle_js = f"""() => {{
        if (typeof window.__is_autonomous === 'undefined') {{
            window.__is_autonomous = {init_flag};
            window.addEventListener('keydown', (e) => {{
                if (e.key === 'F8') {{
                    window.__is_autonomous = !window.__is_autonomous;
                    console.log('F8 Toggled! Autonomous Mode:', window.__is_autonomous);
                }}
            }});
        }}
    }}"""
    await viewport.page.add_init_script(f"({toggle_js})();")
    await viewport.page.evaluate(toggle_js)

    state = make_initial_loop_state()
    adapt_task: asyncio.Task | None = None
    last_adapted_count = 0

    async def _background_online_update() -> None:
        shadow_policy = copy.deepcopy(policy)
        shadow_trainer = OnlinePolicyTrainer(shadow_policy, storage_config=storage, hw_config=hw)
        loss = await asyncio.to_thread(shadow_trainer.online_adapt_step)
        policy.load_state_dict(shadow_policy.state_dict())
        policy.eval()
        print(f"[Background Online Adaptation] Updated policy weights | Loss: {loss:.4f}")

    try:
        async for raw_tick in stream.stream_ticks():
            encoded = await asyncio.to_thread(extractor.extract, raw_tick)
            is_autonomous = await viewport.page.evaluate("Boolean(window.__is_autonomous)")

            action_mask = await process_single_tick(
                encoded=encoded,
                is_autonomous=is_autonomous,
                listener=listener,
                buffer=buffer,
                actor=actor,
                policy=policy,
                state=state,
                session_id=session_id,
            )

            completed = state["completed_demo_reels"]
            if (
                completed > last_adapted_count
                and completed % capture_cfg.online_update_every_n_reels == 0
                and (adapt_task is None or adapt_task.done())
            ):
                last_adapted_count = completed
                adapt_task = asyncio.create_task(_background_online_update())

            mode_tag = "Mode B | AUTO" if is_autonomous else "Mode A | DEMO"
            print(
                f"[{mode_tag}] Reel: {encoded.reel_id[:12]} | "
                f"t={encoded.elapsed_sec:4.1f}s | S_t={state['survival_prob']:.2f} | "
                f"Action: {_action_name(action_mask)} | VRAM: {_vram_mb():.0f}MB"
            )
    except KeyboardInterrupt:
        print("\n[Reel-Clone] Session interrupted by user.")
    finally:
        if adapt_task is not None and not adapt_task.done():
            await adapt_task
        if len(state["current_ticks"]) >= 2 and state["current_reel_id"] is not None:
            _finalize_and_save_reel(
                buffer=buffer,
                session_id=session_id,
                reel_id=state["current_reel_id"],
                ticks=state["current_ticks"],
                tick_actions=state["current_actions"],
                explicit_skip=False,
                censored=True,
            )
        await viewport.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reel-Clone Agent")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["demonstrate", "train", "autonomous", "hybrid"])
    parser.add_argument("--session-id", type=str, default="user_session_01")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    if args.mode in ("hybrid", "demonstrate"):
        asyncio.run(run_hybrid_f8_loop(args.session_id, start_autonomous=False))
    elif args.mode == "autonomous":
        asyncio.run(run_hybrid_f8_loop(args.session_id, start_autonomous=True))
    elif args.mode == "train":
        hw = HardwareConfig()
        storage = StorageConfig()
        policy = TemporalBehaviorPolicy(hw_config=hw)
        trainer = OnlinePolicyTrainer(policy, storage_config=storage, hw_config=hw)
        for e in range(args.epochs):
            metrics = trainer.train_and_evaluate(batch_size=16)
            print(
                f"Epoch {e+1}/{args.epochs} | "
                f"Train Loss: {metrics['train_loss']:.4f} | "
                f"Val Loss: {metrics['val_loss']:.4f} | "
                f"Val Watch MAE: {metrics['val_watch_mae_sec']:.2f}s"
            )
        trainer.save_checkpoint("latest_policy.pt")
        print("[Training] Checkpoint saved successfully.")


if __name__ == "__main__":
    main()