"""Training loop for LSTM+PPO elevator scheduling model."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
import swanlab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, PROJ_ROOT, get_device, merge_reward_config, set_seed
from src.data.dataset import load_raw_data, split_indices, SCENARIO_NAMES
from src.env.elevator_env import ElevatorEnv
from src.models.lstm_ppo import PPOTrainer
from src.runner import MultiEnvRunner

CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"

SCENARIO_LABELS = {k: v.replace("_", " ").title() for k, v in SCENARIO_NAMES.items()}


def setup_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR / "plots", exist_ok=True)


def main(cfg: dict | None = None):
    cfg = cfg or load_config()
    setup_dirs()

    seed = cfg.get("data", {}).get("random_seed", 42)
    set_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    device = get_device()
    print(f"Using device: {device}")

    # Init SwanLab
    swanlab.init(
        project="elevator-scheduling",
        config={
            "env": cfg.get("env", {}),
            "model": cfg.get("model", {}),
            "ppo": cfg.get("ppo", {}),
            "training": cfg.get("training", {}),
            "reward": cfg.get("reward", {}),
            "device": device,
        },
    )

    # Load data
    datasets_dir = PROJ_ROOT / cfg["data"]["datasets_dir"]
    raw = load_raw_data(str(datasets_dir))
    labels_raw = raw["labels"]["arr_0"]
    labels = np.squeeze(labels_raw)
    event_seqs = raw["event_sequences"]["arr_0"]
    event_lens = raw["event_lengths"]["arr_0"]

    train_idx, val_idx, test_idx = split_indices(
        labels, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["random_seed"]
    )
    print(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Build per-scenario val subsets for detailed logging
    val_by_scenario = {}
    for idx in val_idx:
        lbl = int(labels[idx])
        val_by_scenario.setdefault(lbl, []).append(idx)

    # Create environment template
    env_cfg = cfg.get("env", {})
    env_template = ElevatorEnv(merge_reward_config(env_cfg, cfg))

    # Create trainer
    training_cfg = cfg.get("training", {})
    resume_path = training_cfg.get("resume_checkpoint")
    start_epoch = 0
    trainer = PPOTrainer.from_config(env_template.STATE_DIM, env_template.action_space.n, cfg, device)

    total_epochs = training_cfg.get("total_epochs", 200)
    eval_every = training_cfg.get("eval_every", 10)
    early_stop_patience = training_cfg.get("early_stop_patience", 30)
    num_envs = training_cfg.get("num_envs", 1)

    # LR scheduler with optional warmup
    ppo_cfg = cfg.get("ppo", {})
    lr = ppo_cfg.get("learning_rate", 5e-4)
    warmup_epochs = training_cfg.get("warmup_epochs", 0)

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=lr * 0.1,
    )
    trainer.scheduler = main_scheduler  # attach for save/load

    # Resume from checkpoint (after scheduler is created so state can be restored)
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.load(resume_path)
        if "checkpoint_epoch" in resume_path:
            try:
                start_epoch = int(resume_path.split("epoch")[-1].replace(".pt", ""))
            except ValueError:
                start_epoch = 0

    # Entropy annealing params
    entropy_start = ppo_cfg.get("entropy_coef_start", 0.10)
    entropy_end = ppo_cfg.get("entropy_coef_end", 0.01)

    # Training loop
    best_val_reward = -float("inf")
    patience_counter = 0
    epoch_rewards: list[float] = []
    val_rewards: list[float] = []
    print(f"\n{'='*60}")
    print(f"Starting training: {total_epochs} epochs, {num_envs} parallel envs")
    print(f"LR schedule: {'warmup → ' if warmup_epochs else ''}cosine {lr}→{lr * 0.1:.2e}")
    print(f"Entropy: {entropy_start}→{entropy_end}")
    print(f"{'='*60}\n")

    bar_fmt = "{l_bar}{bar:30}{r_bar}"
    epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Epoch", unit="ep", ncols=100,
                     bar_format=bar_fmt, mininterval=0.5)

    # Create multi-env runner once, reuse across epochs
    runner = MultiEnvRunner(env_template, num_envs, trainer.device)

    for epoch in epoch_bar:
        epoch_start = time.time()
        epoch_total_reward = 0.0
        epoch_steps = 0
        n_episodes = 0

        # LR warmup: linear ramp from lr*warmup_ratio to lr
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_ratio = training_cfg.get("warmup_ratio", 0.1)
            progress = (epoch + 1) / max(warmup_epochs, 1)
            warmup_lr = lr * (warmup_ratio + (1.0 - warmup_ratio) * progress)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # Update entropy coefficient (linear anneal)
        progress = epoch / max(total_epochs - 1, 1)
        trainer.set_entropy_coef(entropy_start + (entropy_end - entropy_start) * progress)

        # Shuffle train indices for this epoch
        epoch_indices = np.random.permutation(train_idx)
        feed_ptr = 0

        # Feed initial episodes to all envs
        for i in range(num_envs):
            while feed_ptr < len(epoch_indices):
                idx = epoch_indices[feed_ptr]
                feed_ptr += 1
                events_trimmed = event_seqs[idx][:int(event_lens[idx])]
                if len(events_trimmed) > 0:
                    runner.reset_env(i, events_trimmed, trainer.policy)
                    n_episodes += 1
                    break

        # Collect rollouts in eval mode (no dropout on LSTM, deterministic value estimates)
        trainer.policy.eval()

        # Main collection loop
        loop_iter = 0
        while feed_ptr < len(epoch_indices) or not runner.all_done:
            total_r, steps, n_done = runner.step_all(trainer.policy, trainer.buffer)
            epoch_total_reward += total_r
            epoch_steps += steps
            loop_iter += 1

            # Feed new episodes to finished envs
            for i in range(num_envs):
                if runner.done[i] and feed_ptr < len(epoch_indices):
                    idx = epoch_indices[feed_ptr]
                    feed_ptr += 1
                    events_trimmed = event_seqs[idx][:int(event_lens[idx])]
                    if len(events_trimmed) > 0:
                        runner.reset_env(i, events_trimmed, trainer.policy)
                        n_episodes += 1

            # PPO update when buffer has enough data
            if trainer.buffer.is_ready(trainer.rollout_steps):
                trainer.policy.train()
                trainer.update(last_obs_per_env=runner.get_last_obs_per_env())
                trainer.policy.eval()

            # Live metrics (throttled to avoid excessive terminal I/O)
            if loop_iter % 10 == 0 or loop_iter == 1:
                avg_r = epoch_total_reward / max(epoch_steps, 1)
                epoch_bar.set_postfix_str(
                    f"Ep{n_episodes}/{len(train_idx)} R{avg_r:+.2f} "
                    f"Buf{trainer.buffer.size()}/{trainer.rollout_steps}"
                )

        # Flush remaining buffer data at end of epoch
        if trainer.buffer.size() >= trainer.rollout_steps // 2:
            trainer.policy.train()
            trainer.update(last_obs_per_env=runner.get_last_obs_per_env())

        # Step LR scheduler after optimizer updates (avoids PyTorch warning)
        if epoch >= warmup_epochs:
            main_scheduler.step()

        # Epoch summary
        avg_epoch_reward = epoch_total_reward / max(epoch_steps, 1)
        avg_episode_reward = epoch_total_reward / max(n_episodes, 1)
        epoch_rewards.append(avg_epoch_reward)
        epoch_time = time.time() - epoch_start

        ppo_info = ""
        if trainer.stats:
            ppo_info = (f" | PPO: policy_loss={trainer.stats['policy_loss']:.4e} "
                        f"value_loss={trainer.stats['value_loss']:.4f} "
                        f"entropy={trainer.stats['entropy']:.4f}")
        epoch_bar.write(
            f"Epoch {epoch+1:>3d} done | Time {epoch_time:.1f}s | "
            f"Episodes {n_episodes} | AvgR/step {avg_epoch_reward:.4f} | "
            f"AvgR/ep {avg_episode_reward:.1f}{ppo_info}"
        )

        # SwanLab logging
        current_lr = trainer.optimizer.param_groups[0]['lr']
        swanlab_log = {
            "train/avg_reward_per_episode": avg_episode_reward,
            "train/avg_reward_per_step": avg_epoch_reward,
            "train/episodes": n_episodes,
            "train/steps": epoch_steps,
            "train/epoch_time_s": epoch_time,
            "train/entropy_coef": trainer.entropy_coef,
            "train/learning_rate": current_lr,
        }
        if trainer.stats:
            swanlab_log.update({
                "ppo/policy_loss": trainer.stats["policy_loss"],
                "ppo/value_loss": trainer.stats["value_loss"],
                "ppo/entropy": trainer.stats["entropy"],
                "ppo/n_updates": trainer.stats["n_updates"],
                "ppo/grad_norm": trainer.stats.get("grad_norm", 0.0),
                "ppo/approx_kl": trainer.stats.get("approx_kl", 0.0),
                "ppo/clip_frac": trainer.stats.get("clip_frac", 0.0),
                "ppo/explained_var": trainer.stats.get("explained_var", 0.0),
                "ppo/value_pred_error": trainer.stats.get("value_pred_error", 0.0),
            })
        swanlab.log(swanlab_log, step=epoch)

        # Validation
        if (epoch + 1) % eval_every == 0:
            val_reward = validate(trainer, env_template, event_seqs, event_lens, val_idx)
            val_rewards.append(val_reward)
            epoch_bar.write(f"  Validation avg reward: {val_reward:.4f}")
            swanlab.log({"val/avg_reward": val_reward}, step=epoch)

            # Per-scenario validation logging
            if len(val_by_scenario) > 1:
                scenario_rewards = {}
                for lbl, idxs in val_by_scenario.items():
                    sr = validate(trainer, env_template, event_seqs, event_lens,
                                  np.array(idxs, dtype=np.int64), quiet=True)
                    scenario_rewards[f"val/scenario_{SCENARIO_LABELS.get(lbl, lbl)}_reward"] = sr
                swanlab.log(scenario_rewards, step=epoch)

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                patience_counter = 0
                trainer.save(str(CHECKPOINT_DIR / "best_model.pt"))
                epoch_bar.write(f"  -> New best model saved!")
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                epoch_bar.write(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            trainer.save(str(CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pt"))

    # Final save and plot
    runner.close()
    trainer.save(str(CHECKPOINT_DIR / "final_model.pt"))
    plot_training_curve(epoch_rewards, val_rewards, eval_every)
    swanlab.finish()
    print(f"\nTraining complete. Best val reward: {best_val_reward:.4f}")
    print(f"Model saved to {CHECKPOINT_DIR}/best_model.pt")


def validate(trainer: PPOTrainer, env: ElevatorEnv,
             event_seqs: np.ndarray, event_lens: np.ndarray,
             val_idx: np.ndarray, quiet: bool = False) -> float:
    """Evaluate on validation set, return average reward."""
    trainer.policy.eval()
    total_reward = 0.0
    total_steps = 0

    policy = trainer.policy
    _obs_gpu = torch.empty(env.STATE_DIM, dtype=torch.float32, device=trainer.device)

    iterator = val_idx
    if not quiet:
        iterator = tqdm(val_idx, desc="  Validating", unit="ep", leave=False, ncols=80)

    with torch.inference_mode():
        for idx in iterator:
            events = event_seqs[idx]
            length = event_lens[idx]
            events_trimmed = events[:int(length)]
            if len(events_trimmed) == 0:
                continue

            obs, _ = env.reset(options={"events": events_trimmed})
            done = False
            hidden = policy.get_initial_hidden(1, trainer.device)

            while not done:
                _obs_gpu.copy_(torch.from_numpy(obs), non_blocking=True)
                obs_seq = _obs_gpu.unsqueeze(0).unsqueeze(0)  # type: ignore[call-arg]
                action, _, _, hidden = policy.get_action(
                    obs_seq, hidden=hidden, deterministic=True
                )
                obs, reward, done, _, _ = env.step(int(action.item()))  # type: ignore[arg-type]
                total_reward += reward
                total_steps += 1

    policy.train()
    return total_reward / max(total_steps, 1)


def plot_training_curve(epoch_rewards: list[float], val_rewards: list[float],
                        eval_every: int):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(epoch_rewards) + 1)
    ax.plot(epochs, epoch_rewards, label="Train Avg Reward", alpha=0.6)
    if val_rewards:
        val_epochs = [(i + 1) * eval_every for i in range(len(val_rewards))]
        ax.plot(val_epochs, val_rewards, "ro-", label="Val Avg Reward", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Reward")
    ax.set_title("PPO Training — Elevator Scheduling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(CHECKPOINT_DIR / "plots" / "training_curve.png"), dpi=120)
    plt.close(fig)
    print(f"Training curve saved to {CHECKPOINT_DIR}/plots/training_curve.png")


if __name__ == "__main__":
    main()
