"""Evaluate trained LSTM+PPO model across all test scenarios with baselines."""

from __future__ import annotations

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, PROJ_ROOT, get_device, merge_reward_config
from src.data.dataset import load_raw_data, split_indices, SCENARIO_NAMES
from src.env.elevator_env import ElevatorEnv
from src.models.lstm_ppo import PPOTrainer

SCENARIO_LABELS = {k: v.replace("_", " ").title() for k, v in SCENARIO_NAMES.items()}


def _run_episode(env, obs, policy, trainer, deterministic: bool = True) -> tuple[float, dict]:
    """Run a single episode and return (episode_reward, metrics_dict)."""
    done = False
    hidden = policy.get_initial_hidden(1, trainer.device)
    episode_reward = 0.0
    _obs_gpu = torch.empty(env.STATE_DIM, dtype=torch.float32, device=trainer.device)

    while not done:
        _obs_gpu.copy_(torch.from_numpy(obs), non_blocking=True)
        obs_seq = _obs_gpu.unsqueeze(0).unsqueeze(0)
        action, _, _, hidden = policy.get_action(
            obs_seq, hidden=hidden, deterministic=deterministic
        )
        obs, reward, done, _, _ = env.step(int(action.item()))
        episode_reward += reward

    metrics = env.get_episode_metrics()
    metrics["reward"] = episode_reward
    return episode_reward, metrics


def _run_baseline_episode(env_cfg, events_trimmed, action_fn):
    """Run a baseline episode with a custom action selection function."""
    env = ElevatorEnv(dict(env_cfg))
    obs, _ = env.reset(options={"events": events_trimmed})
    episode_reward = 0.0
    done = False

    while not done:
        action = action_fn(env)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward

    metrics = env.get_episode_metrics()
    metrics["reward"] = episode_reward
    return episode_reward, metrics


def _baseline_nearest_car(env_cfg, events_trimmed):
    """Nearest-Car baseline: assign pending call to nearest available elevator."""
    def _select(env):
        if not env.pending_calls:
            return 0
        call_floor = env.pending_calls[0]["floor"]
        best = min(range(len(env.elevators)),
                   key=lambda i: abs(env.elevators[i].current_floor - call_floor))
        return best
    return _run_baseline_episode(env_cfg, events_trimmed, _select)


def _baseline_random(env_cfg, events_trimmed):
    """Random baseline: assign pending call to a random elevator."""
    rng = np.random.RandomState(42)
    def _select(env):
        if not env.pending_calls:
            return 0
        return int(rng.randint(0, env.num_elevators))
    return _run_baseline_episode(env_cfg, events_trimmed, _select)


def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95):
    """Compute bootstrap confidence interval for the mean."""
    n = len(data)
    means = np.empty(n_bootstrap)
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, alpha * 100))
    hi = float(np.percentile(means, (1 - alpha) * 100))
    return lo, hi


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(PROJ_ROOT / "checkpoints" / "best_model.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of stochastic evaluation runs per test file")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline comparisons")
    parser.add_argument("--max-test", type=int, default=0,
                        help="Limit test files (0=all)")
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, dict):
        args = argparse.Namespace(**args)

    cfg = load_config()
    device = get_device(args.device)
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

    if args.max_test > 0:
        test_idx = test_idx[:args.max_test]
    print(f"Test set: {len(test_idx)} files")

    # Load model
    env_cfg = cfg.get("env", {})
    env_template = ElevatorEnv(merge_reward_config(env_cfg, cfg))

    trainer = PPOTrainer.from_config(env_template.STATE_DIM, env_template.action_space.n, cfg, device)
    trainer.load(args.checkpoint)
    trainer.policy.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate per scenario (model + baselines)
    ScenarioAccum = lambda: defaultdict(list)
    scenario_rewards = defaultdict(ScenarioAccum)
    all_metrics: dict[str, list] = defaultdict(list)

    with torch.inference_mode():
        for idx in test_idx:
            events = event_seqs[idx]
            length = event_lens[idx]
            events_trimmed = events[:int(length)]
            if len(events_trimmed) == 0:
                continue

            label = int(labels[idx])
            file_id = str(file_ids[idx])

            # Model evaluation (multiple runs)
            for run in range(args.n_runs):
                env = ElevatorEnv(merge_reward_config(dict(env_cfg), cfg))
                obs, _ = env.reset(options={"events": events_trimmed.copy()})
                ep_reward, metrics = _run_episode(
                    env, obs, trainer.policy, trainer, deterministic=(run == 0))
                metrics["file"] = file_id
                metrics["scenario"] = label
                metrics["run"] = run
                scenario_rewards[label]["model"].append(ep_reward)
                all_metrics[label].append(metrics)

            # Baselines (single deterministic run)
            if not args.no_baselines:
                baseline_cfg = merge_reward_config(dict(env_cfg), cfg)

                nc_reward, nc_metrics = _baseline_nearest_car(baseline_cfg, events_trimmed)
                scenario_rewards[label]["nearest_car"].append(nc_reward)
                nc_metrics["file"] = file_id
                nc_metrics["scenario"] = label
                all_metrics[f"{label}_nearest_car"].append(nc_metrics)

                rnd_reward, rnd_metrics = _baseline_random(baseline_cfg, events_trimmed)
                scenario_rewards[label]["random"].append(rnd_reward)
                rnd_metrics["file"] = file_id
                rnd_metrics["scenario"] = label
                all_metrics[f"{label}_random"].append(rnd_metrics)

    # Print results
    has_baselines = not args.no_baselines

    print(f"\n{'='*90}")
    print(f"Evaluation Results — {len(test_idx)} test files, {args.n_runs} runs each")
    print(f"{'='*90}")

    # Header
    if has_baselines:
        print(f"{'Scenario':<18} {'Files':>5} {'ModelR':>8} {'NC-R':>8} {'Rnd-R':>8} "
              f"{'AvgWait':>8} {'p95Wait':>8} {'Long%':>7} {'Empty%':>7} {'OpEff%':>7} {'Qual':>6}")
    else:
        print(f"{'Scenario':<18} {'Files':>5} {'ModelR':>8} {'AvgWait':>8} "
              f"{'p50Wait':>8} {'p95Wait':>8} {'Long%':>7} {'Empty%':>7} {'OpEff%':>7}")
    print("-" * 90)

    all_avgs = []
    for label in sorted(all_metrics.keys(), key=str):
        # Skip baseline pseudo-labels for iteration; process with real labels
        if isinstance(label, str) and "_" in str(label):
            continue

        mlist = all_metrics[label]
        if not mlist:
            continue

        n_files = len(set(m["file"] for m in mlist))

        waits = np.array([m["avg_wait_time"] for m in mlist])
        longs = np.array([m["long_wait_rate"] for m in mlist])
        empties = np.array([m["empty_load_rate"] for m in mlist])
        ops = np.array([m["operational_efficiency"] for m in mlist])
        rewards_arr = np.array([m["reward"] for m in mlist])

        avg_wait = float(waits.mean())
        p50_wait = float(np.percentile(waits, 50))
        p95_wait = float(np.percentile(waits, 95))
        long_wait = float(longs.mean()) * 100
        empty_rate = float(empties.mean()) * 100
        op_eff = float(ops.mean()) * 100
        avg_reward = float(rewards_arr.mean())

        # Bootstrap CI for model reward
        model_rewards = np.array(scenario_rewards[label].get("model", []))
        if len(model_rewards) > 1:
            ci_lo, ci_hi = bootstrap_confidence_interval(model_rewards)
        else:
            ci_lo, ci_hi = avg_reward, avg_reward

        name = SCENARIO_LABELS.get(label, f"Scenario{label}")

        scheduling_quality = float(np.mean([m.get("scheduling_quality", 0) for m in mlist]))

        if has_baselines:
            nc_r = float(np.mean(scenario_rewards[label].get("nearest_car", [0])))
            rnd_r = float(np.mean(scenario_rewards[label].get("random", [0])))
            sq = scheduling_quality
            print(f"{name:<18} {n_files:>5} {avg_reward:>8.2f} {nc_r:>8.2f} {rnd_r:>8.2f} "
                  f"{avg_wait:>8.2f} {p95_wait:>8.2f} {long_wait:>6.1f}% "
                  f"{empty_rate:>6.1f}% {op_eff:>6.1f}% {sq:>6.3f}")
        else:
            print(f"{name:<18} {n_files:>5} {avg_reward:>8.2f} "
                  f"{avg_wait:>8.2f} {p50_wait:>8.2f} {p95_wait:>8.2f} "
                  f"{long_wait:>6.1f}% {empty_rate:>6.1f}% {op_eff:>6.1f}%")

        all_avgs.append({
            "scenario": name, "n_files": n_files,
            "avg_wait": avg_wait, "p50_wait": p50_wait, "p95_wait": p95_wait,
            "long_wait_pct": long_wait, "empty_pct": empty_rate,
            "op_eff_pct": op_eff, "avg_reward": avg_reward,
            "reward_ci_lo": ci_lo, "reward_ci_hi": ci_hi,
            "scheduling_quality": scheduling_quality,
        })

        # Print CI info
        if len(model_rewards) > 1:
            print(f"  Model reward 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")

    # Overall summary
    all_waits = np.concatenate([np.array([m["avg_wait_time"] for m in all_metrics[l]])
                                for l in all_metrics if isinstance(l, (int, np.integer))])
    all_rewards = np.concatenate([np.array([m["reward"] for m in all_metrics[l]])
                                  for l in all_metrics if isinstance(l, (int, np.integer))])

    print("-" * 90)
    print(f"{'OVERALL':<18} {len(test_idx):>5} {all_rewards.mean():>8.2f} "
          f"{all_waits.mean():>8.2f} {np.percentile(all_waits, 50):>8.2f} "
          f"{np.percentile(all_waits, 95):>8.2f} "
          f"{'—':>6} {'—':>6} {'—':>6}")

    # Save charts
    _plot_metrics(all_avgs, has_baselines)
    if has_baselines:
        _plot_baseline_comparison(all_avgs, scenario_rewards)
    print(f"\nCharts saved to {PROJ_ROOT}/checkpoints/plots/")


def _plot_metrics(all_avgs: list[dict], has_baselines: bool = False):
    out_dir = PROJ_ROOT / "checkpoints" / "plots"
    os.makedirs(out_dir, exist_ok=True)

    scenarios = [d["scenario"] for d in all_avgs]
    x = range(len(scenarios))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    metrics = [
        ("avg_wait", "Avg Wait Time (s)", axes[0, 0]),
        ("p95_wait", "P95 Wait Time (s)", axes[0, 1]),
        ("long_wait_pct", "Long Wait Rate (%)", axes[0, 2]),
        ("empty_pct", "Empty Load Rate (%)", axes[1, 0]),
        ("op_eff_pct", "Operational Efficiency (%)", axes[1, 1]),
        ("avg_reward", "Avg Reward", axes[1, 2]),
        ("scheduling_quality", "Scheduling Quality", axes[2, 0]),
    ]
    # Hide unused subplots
    for r in range(3):
        for c in range(3):
            if not any(a is axes[r, c] for _, _, a in metrics):
                axes[r, c].set_visible(False)

    for key, ylabel, ax in metrics:
        vals = [d[key] for d in all_avgs]
        bars = ax.bar(x, vals, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{val:.1f}", ha="center", fontsize=7)

    fig.suptitle("Elevator Scheduling — LSTM+PPO Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_dir / "evaluation_metrics.png"), dpi=120)
    plt.close(fig)


def _plot_baseline_comparison(all_avgs: list[dict], scenario_rewards: dict):
    out_dir = PROJ_ROOT / "checkpoints" / "plots"
    os.makedirs(out_dir, exist_ok=True)

    scenarios = [d["scenario"] for d in all_avgs]
    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    model_vals = [d["avg_reward"] for d in all_avgs]
    ax.bar(x - width, model_vals, width, label="LSTM+PPO", color="steelblue", alpha=0.85)

    nc_vals = []
    rnd_vals = []
    for d in all_avgs:
        lbl = list(SCENARIO_LABELS.keys())[list(SCENARIO_LABELS.values()).index(d["scenario"])] \
            if d["scenario"] in SCENARIO_LABELS.values() else None
        if lbl is not None:
            nc_vals.append(float(np.mean(scenario_rewards[lbl].get("nearest_car", [0]))))
            rnd_vals.append(float(np.mean(scenario_rewards[lbl].get("random", [0]))))
        else:
            nc_vals.append(0)
            rnd_vals.append(0)

    ax.bar(x, nc_vals, width, label="Nearest Car", color="darkorange", alpha=0.85)
    ax.bar(x + width, rnd_vals, width, label="Random", color="lightcoral", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Reward by Scenario — Model vs Baselines")
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(out_dir / "baseline_comparison.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
