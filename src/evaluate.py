"""Evaluate trained LSTM+PPO model across all test scenarios."""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_config, PROJ_ROOT
from data.dataset import load_raw_data, split_indices, SCENARIO_NAMES
from env.elevator_env import ElevatorEnv
from models.lstm_ppo import PPOTrainer

SCENARIO_LABELS = {k: v.replace("_", " ").title() for k, v in SCENARIO_NAMES.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(PROJ_ROOT / "checkpoints" / "best_model.pt"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config()
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    datasets_dir = PROJ_ROOT / cfg["data"]["datasets_dir"]
    raw = load_raw_data(str(datasets_dir))
    labels = np.squeeze(raw["labels"]["arr_0"])
    event_seqs = raw["event_sequences"]["arr_0"]
    event_lens = raw["event_lengths"]["arr_0"]
    file_ids = raw["file_ids"]["arr_0"]

    _, _, test_idx = split_indices(
        labels, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["random_seed"]
    )
    print(f"Test set: {len(test_idx)} files")

    # Load model
    env_cfg = cfg.get("env", {})
    env = ElevatorEnv(env_cfg)

    trainer = PPOTrainer.from_config(env.STATE_DIM, env.action_space.n, cfg, device)
    trainer.load(args.checkpoint)
    trainer.policy.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate per scenario
    scenario_metrics = defaultdict(list)
    total_reward = 0.0
    all_wait, all_ride, all_long, all_empty, all_eff, all_energy = [], [], [], [], [], []

    # Pre-allocated GPU buffer avoids per-step allocation
    _obs_gpu = torch.empty(env.STATE_DIM, dtype=torch.float32, device=trainer.device)

    with torch.inference_mode():
        for idx in test_idx:
            events = event_seqs[idx]
            length = event_lens[idx]
            events_trimmed = events[:int(length)]
            if len(events_trimmed) == 0:
                continue

            label = int(labels[idx])
            obs, _ = env.reset(options={"events": events_trimmed})
            done = False
            hidden = trainer.policy.get_initial_hidden(1, trainer.device)
            episode_reward = 0.0

            while not done:
                _obs_gpu.copy_(torch.from_numpy(obs), non_blocking=True)
                obs_seq = _obs_gpu.unsqueeze(0).unsqueeze(0)
                action, _, _, hidden = trainer.policy.get_action(
                    obs_seq, hidden=hidden, deterministic=True
                )
                obs, reward, done, _, _ = env.step(action.item())
                episode_reward += reward

            metrics = env.get_episode_metrics()
            metrics["reward"] = episode_reward
            metrics["scenario"] = label
            metrics["file"] = str(file_ids[idx])
            scenario_metrics[label].append(metrics)
            total_reward += episode_reward

            # Collect raw values for overall stats (avoid re-iteration later)
            all_wait.append(metrics["avg_wait_time"])
            all_ride.append(metrics["avg_ride_time"])
            all_long.append(metrics["long_wait_rate"])
            all_empty.append(metrics["empty_load_rate"])
            all_eff.append(metrics["operational_efficiency"])
            all_energy.append(metrics["energy_wh"])

    # Print results
    print(f"\n{'='*70}")
    print(f"Evaluation Results — {len(test_idx)} test files")
    print(f"{'='*70}")
    print(f"Avg reward: {total_reward / len(test_idx):.4f}\n")

    print(f"{'Scenario':<18} {'Files':>5} {'AvgWait':>8} {'AvgRide':>8} "
          f"{'LongWait%':>9} {'Empty%':>8} {'OpEff%':>8} {'EnergyWh':>9}")
    print("-" * 70)

    all_avgs = []
    for label in sorted(scenario_metrics.keys()):
        mlist = scenario_metrics[label]
        n = len(mlist)

        # Single-pass extraction into pre-allocated arrays
        waits = np.empty(n); rides = np.empty(n); longs = np.empty(n)
        empties = np.empty(n); ops = np.empty(n); energies = np.empty(n)
        rewards_arr = np.empty(n)
        for i, m in enumerate(mlist):
            waits[i] = m["avg_wait_time"]
            rides[i] = m["avg_ride_time"]
            longs[i] = m["long_wait_rate"]
            empties[i] = m["empty_load_rate"]
            ops[i] = m["operational_efficiency"]
            energies[i] = m["energy_wh"]
            rewards_arr[i] = m["reward"]

        avg_wait = float(waits.mean())
        avg_ride = float(rides.mean())
        long_wait = float(longs.mean()) * 100
        empty_rate = float(empties.mean()) * 100
        op_eff = float(ops.mean()) * 100
        energy = float(energies.mean())
        avg_reward = float(rewards_arr.mean())

        name = SCENARIO_LABELS.get(label, f"Scenario{label}")
        print(f"{name:<18} {n:>5} {avg_wait:>8.2f} {avg_ride:>8.2f} {long_wait:>8.1f}% "
              f"{empty_rate:>7.1f}% {op_eff:>7.1f}% {energy:>8.2f}")

        all_avgs.append({
            "scenario": name, "avg_wait": avg_wait, "avg_ride": avg_ride,
            "long_wait_pct": long_wait, "empty_pct": empty_rate,
            "op_eff_pct": op_eff, "energy_wh": energy, "avg_reward": avg_reward,
        })

    # Overall averages (from pre-collected lists, no re-iteration)
    overall_wait = np.mean(all_wait)
    overall_ride = np.mean(all_ride)
    overall_empty = np.mean(all_empty) * 100
    overall_eff = np.mean(all_eff) * 100

    print("-" * 70)
    print(f"{'OVERALL':<18} {len(test_idx):>5} {overall_wait:>8.2f} {overall_ride:>8.2f} "
          f"{'—':>9} {overall_empty:>7.1f}% {overall_eff:>7.1f}% {'—':>9}")

    # Save charts
    _plot_metrics(all_avgs)
    print(f"\nCharts saved to {PROJ_ROOT}/checkpoints/plots/")


def _plot_metrics(all_avgs: list[dict]):
    out_dir = PROJ_ROOT / "checkpoints" / "plots"
    os.makedirs(out_dir, exist_ok=True)

    scenarios = [d["scenario"] for d in all_avgs]
    x = range(len(scenarios))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("avg_wait", "Avg Wait Time (s)", axes[0, 0]),
        ("avg_ride", "Avg Ride Time (s)", axes[0, 1]),
        ("long_wait_pct", "Long Wait Rate (%)", axes[0, 2]),
        ("empty_pct", "Empty Load Rate (%)", axes[1, 0]),
        ("op_eff_pct", "Operational Efficiency (%)", axes[1, 1]),
        ("energy_wh", "Energy (Wh)", axes[1, 2]),
    ]

    for key, ylabel, ax in metrics:
        vals = [d[key] for d in all_avgs]
        bars = ax.bar(x, vals, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", fontsize=7)

    fig.suptitle("Elevator Scheduling — LSTM+PPO Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_dir / "evaluation_metrics.png"), dpi=120)
    plt.close(fig)

    # Reward bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    rewards = [d["avg_reward"] for d in all_avgs]
    ax2.bar(x, rewards, color="coral", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=30, ha="right")
    ax2.set_ylabel("Avg Episode Reward")
    ax2.set_title("Reward by Scenario")
    ax2.axhline(y=0, color="gray", linewidth=0.5)
    fig2.tight_layout()
    fig2.savefig(str(out_dir / "reward_by_scenario.png"), dpi=120)
    plt.close(fig2)


if __name__ == "__main__":
    main()
