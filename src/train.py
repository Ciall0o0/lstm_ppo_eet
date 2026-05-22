"""Training loop for LSTM+PPO elevator scheduling model."""

from __future__ import annotations

import os
import time
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_config, PROJ_ROOT
from data.dataset import load_raw_data, split_indices
from env.elevator_env import ElevatorEnv
from models.lstm_ppo import PPOTrainer

CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"
torch.backends.cudnn.benchmark = True

def setup_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR / "plots", exist_ok=True)


def main():
    cfg = load_config()
    setup_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    datasets_dir = PROJ_ROOT / cfg["data"]["datasets_dir"]
    raw = load_raw_data(str(datasets_dir))
    labels_raw = raw["labels"]["arr_0"]
    # Handle potential 2D labels
    labels = np.squeeze(labels_raw)
    event_seqs = raw["event_sequences"]["arr_0"]
    event_lens = raw["event_lengths"]["arr_0"]
    file_ids = raw["file_ids"]["arr_0"]

    train_idx, val_idx, test_idx = split_indices(
        labels, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["random_seed"]
    )
    print(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Create environment
    env_cfg = cfg.get("env", {})
    reward_cfg = cfg.get("reward", {})
    env_cfg["passenger_delivered"] = reward_cfg.get("passenger_delivered", 2.0)
    env_cfg["wait_time_per_sec"] = reward_cfg.get("wait_time_per_sec", -0.05)
    env_cfg["empty_distance_per_floor"] = reward_cfg.get("empty_distance_per_floor", -0.1)
    env_cfg["energy_per_start_stop"] = reward_cfg.get("energy_per_start_stop", -0.05)
    env_cfg["idle_penalty_per_sec"] = reward_cfg.get("idle_penalty_per_sec", 0.0)
    env = ElevatorEnv(env_cfg)

    # Create trainer
    trainer = PPOTrainer.from_config(env.STATE_DIM, env.action_space.n, cfg, device)

    training_cfg = cfg.get("training", {})
    total_epochs = training_cfg.get("total_epochs", 200)
    eval_every = training_cfg.get("eval_every", 10)
    early_stop_patience = training_cfg.get("early_stop_patience", 30)
    log_interval = training_cfg.get("log_interval", 100)

    # Training loop
    best_val_reward = -float("inf")
    patience_counter = 0
    epoch_rewards: list[float] = []
    val_rewards: list[float] = []
    all_metrics: list[dict] = []

    print(f"\n{'='*60}")
    print(f"Starting training: {total_epochs} epochs")
    print(f"{'='*60}\n")

    epoch_bar = tqdm(range(total_epochs), desc="Training", unit="epoch", ncols=120)
    for epoch in epoch_bar:
        epoch_start = time.time()
        epoch_total_reward = 0.0
        epoch_steps = 0
        n_episodes = 0

        # Shuffle train indices for this epoch
        epoch_indices = np.random.permutation(train_idx)

        episode_bar = tqdm(
            epoch_indices, desc=f"Epoch {epoch+1:>3d}", unit="ep",
            leave=False, ncols=120,
        )
        for idx in episode_bar:
            events = event_seqs[idx]
            length = event_lens[idx]
            events_trimmed = events[:int(length)]

            if len(events_trimmed) == 0:
                continue

            obs, _ = env.reset(options={"events": events_trimmed})
            done = False
            hidden = trainer.policy.get_initial_hidden(1, trainer.device)

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=trainer.device)
                obs_seq = obs_t.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                action, log_prob, value, hidden = trainer.policy.get_action(obs_seq, hidden=hidden)
                action_int = int(action.item())
                # log_prob, value kept as GPU tensors — no .item() sync

                next_obs, reward, done, _, info = env.step(action_int)

                trainer.buffer.add(obs_t, action_int, reward, value, log_prob, done)

                obs = next_obs
                epoch_total_reward += reward
                epoch_steps += 1

                if trainer.buffer.ptr >= trainer.rollout_steps:
                    trainer.update(last_obs=obs)

            n_episodes += 1

            # Update postfix with live metrics
            if n_episodes % log_interval == 0:
                avg_r = epoch_total_reward / max(epoch_steps, 1)
                episode_bar.set_postfix({
                    "Steps": epoch_steps,
                    "AvgR": f"{avg_r:.4f}",
                    "Buf": f"{trainer.buffer.ptr}/{trainer.rollout_steps}",
                })

        # Epoch summary
        avg_epoch_reward = epoch_total_reward / max(epoch_steps, 1)
        epoch_rewards.append(avg_epoch_reward)
        epoch_time = time.time() - epoch_start

        ppo_info = ""
        if trainer.stats:
            ppo_info = (f" | PPO: policy_loss={trainer.stats['policy_loss']:.4e} "
                        f"value_loss={trainer.stats['value_loss']:.4f} "
                        f"entropy={trainer.stats['entropy']:.4f}")
        epoch_bar.write(
            f"Epoch {epoch+1:>3d} done | Time {epoch_time:.1f}s | "
            f"Episodes {n_episodes} | Avg Reward {avg_epoch_reward:.4f}{ppo_info}"
        )

        # Validation
        if (epoch + 1) % eval_every == 0:
            val_reward = validate(trainer, env, event_seqs, event_lens, val_idx)
            val_rewards.append(val_reward)
            epoch_bar.write(f"  Validation avg reward: {val_reward:.4f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                patience_counter = 0
                trainer.save(str(CHECKPOINT_DIR / "best_model.pt"))
                epoch_bar.write(f"  -> New best model saved!")
            else:
                patience_counter += eval_every

            if patience_counter >= early_stop_patience:
                epoch_bar.write(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            trainer.save(str(CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pt"))

    # Final save and plot
    trainer.save(str(CHECKPOINT_DIR / "final_model.pt"))
    plot_training_curve(epoch_rewards, val_rewards, eval_every)
    print(f"\nTraining complete. Best val reward: {best_val_reward:.4f}")
    print(f"Model saved to {CHECKPOINT_DIR}/best_model.pt")


def validate(trainer: PPOTrainer, env: ElevatorEnv,
             event_seqs: np.ndarray, event_lens: np.ndarray,
             val_idx: np.ndarray) -> float:
    """Evaluate on validation set, return average reward."""
    trainer.policy.eval()
    total_reward = 0.0
    total_steps = 0

    # Pre-allocated GPU buffer avoids per-step torch.as_tensor allocation
    _obs_gpu = torch.empty(env.STATE_DIM, dtype=torch.float32, device=trainer.device)

    with torch.inference_mode():
        for idx in tqdm(val_idx, desc="Validating", unit="ep", leave=False, ncols=100):
            events = event_seqs[idx]
            length = event_lens[idx]
            events_trimmed = events[:int(length)]
            if len(events_trimmed) == 0:
                continue

            obs, _ = env.reset(options={"events": events_trimmed})
            done = False
            hidden = trainer.policy.get_initial_hidden(1, trainer.device)

            while not done:
                _obs_gpu.copy_(torch.from_numpy(obs), non_blocking=True)
                obs_seq = _obs_gpu.unsqueeze(0).unsqueeze(0)
                action, _, _, hidden = trainer.policy.get_action(
                    obs_seq, hidden=hidden, deterministic=True
                )
                obs, reward, done, _, _ = env.step(action.item())
                total_reward += reward
                total_steps += 1

    trainer.policy.train()
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
