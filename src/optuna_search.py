"""Optuna hyperparameter search for LSTM+PPO elevator scheduling.

Rewritten to match the current dual-optimizer architecture with:
- Separate actor/critic gradient clipping
- Assignment reward shaping (proximity, direction, load, estimated_wait)
- LSTM burn-in for temporal context
- KL early stopping

Search space calibrated against proven working values from training runs.

Time-saving strategies:
- Per-trial early stopping (patience=10 on val reward)
- Reduced episodes per epoch (16 instead of 48)
- Reduced n_trials (20 instead of 30)
- Aggressive MedianPruner
"""

import gc
import json
import os
import time
import warnings

import numpy as np
import torch
import optuna
from optuna.trial import TrialState

from src.utils import load_config, PROJ_ROOT, get_device, merge_reward_config
from src.data.dataset import load_raw_data, split_indices
from src.env.elevator_env import ElevatorEnv
from src.models.lstm_ppo import PPOTrainer
from src.runner import MultiEnvRunner

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

# ---- constants ----
NUM_ENVS = 16
MAX_EPISODES_PER_EPOCH = 16  # reduced from 48 — 3x faster per epoch
N_EPOCHS = 50                 # long enough to detect late collapse
PATIENCE = 10                 # per-trial early stopping on val reward


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_data() -> dict:
    """Pre-load data into memory cache."""
    cfg = load_config()
    datasets_dir = PROJ_ROOT / cfg["data"]["datasets_dir"]
    raw = load_raw_data(str(datasets_dir))
    labels = np.squeeze(raw["labels"]["arr_0"])
    event_seqs = raw["event_sequences"]["arr_0"]
    event_lens = raw["event_lengths"]["arr_0"]
    train_idx, val_idx, _ = split_indices(
        labels, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"],
        cfg["data"]["random_seed"])
    return {
        "cfg": cfg,
        "labels": labels,
        "event_seqs": event_seqs,
        "event_lens": event_lens,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def _validate(trainer, env_template, event_seqs, event_lens, val_idx,
              num_envs=16, runner=None):
    """Batched validation using MultiEnvRunner for parallel GPU inference."""
    trainer.policy.eval()
    own_runner = runner is None
    if own_runner:
        n = min(num_envs, len(val_idx))
        runner = MultiEnvRunner(env_template, n, trainer.device)
    else:
        n = min(num_envs, len(val_idx), runner.num_envs)
    runner.done = [True] * runner.num_envs
    runner._active_count = 0
    runner._active = []

    total_reward = 0.0
    total_steps = 0
    ptr = 0

    for i in range(n):
        if ptr >= len(val_idx):
            break
        idx = val_idx[ptr]
        ptr += 1
        events_trimmed = event_seqs[idx][:int(event_lens[idx])]
        if len(events_trimmed) > 0:
            runner.reset_env(i, events_trimmed, trainer.policy)

    while not runner.all_done:
        r, s, _ = runner.step_all(trainer.policy, deterministic=True)
        total_reward += r
        total_steps += s

        for i in range(n):
            if runner.done[i] and ptr < len(val_idx):
                idx = val_idx[ptr]
                ptr += 1
                events_trimmed = event_seqs[idx][:int(event_lens[idx])]
                if len(events_trimmed) > 0:
                    runner.reset_env(i, events_trimmed, trainer.policy)

    if own_runner:
        runner.close()
    trainer.policy.train()
    return total_reward / max(total_steps, 1)


def _cleanup(runner, device):
    runner.close()
    gc.collect()


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, data: dict) -> float:
    device = torch.device(get_device())

    _cfg = data["cfg"]
    _event_seqs = data["event_seqs"]
    _event_lens = data["event_lens"]
    _train_idx = data["train_idx"]
    _val_idx = data["val_idx"]

    print(f"\n{'='*60}\n[Trial {trial.number:2d}] starting\n{'='*60}", flush=True)

    # ---- PPO hyperparameters (narrowed around proven working values) ----
    # Working baseline: lr=3e-4, value_loss_coef=0.7, entropy 0.01→0.001,
    # ppo_epochs=10, batch_size=8192, kl_target=0.01
    lr = trial.suggest_float("learning_rate", 2e-4, 5e-4, log=True)
    entropy_start = trial.suggest_float("entropy_coef_start", 0.005, 0.02)
    entropy_end = trial.suggest_float("entropy_coef_end", 0.001, min(0.01, entropy_start * 0.5))
    entropy_floor = trial.suggest_float("entropy_floor", 0.0, 0.01)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.15, 0.25)
    value_loss_coef = trial.suggest_float("value_loss_coef", 0.5, 1.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 3e-4, log=True)
    ppo_epochs = trial.suggest_int("ppo_epochs", 5, 10)
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192])
    seq_len = trial.suggest_categorical("seq_len", [32, 64])
    kl_target = trial.suggest_float("kl_target", 0.005, 0.02)

    # ---- Model architecture ----
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [128, 256])
    lstm_layers = 2  # fixed: 2 layers works well
    burn_in_steps = trial.suggest_categorical("burn_in_steps", [4, 8, 16])
    actor_hidden = trial.suggest_categorical("actor_hidden", [32, 64, 128])
    critic_hidden = trial.suggest_categorical("critic_hidden", [32, 64, 128])
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])

    # ---- Assignment reward shaping (critical for learning) ----
    assignment_proximity = trial.suggest_float("assignment_proximity", -3.0, -0.5)
    assignment_direction_align = trial.suggest_float("assignment_direction_align", 0.5, 3.0)
    assignment_load_balance = trial.suggest_float("assignment_load_balance", -3.0, -0.5)
    assignment_estimated_wait = trial.suggest_float("assignment_estimated_wait", -0.05, -0.005)

    # ---- Build env with searched reward params ----
    env_cfg = dict(_cfg.get("env", {}))
    reward_cfg = dict(_cfg.get("reward", {}))
    reward_cfg["assignment_proximity"] = assignment_proximity
    reward_cfg["assignment_direction_align"] = assignment_direction_align
    reward_cfg["assignment_load_balance"] = assignment_load_balance
    reward_cfg["assignment_estimated_wait"] = assignment_estimated_wait
    # Merge reward keys into env config
    for k, v in reward_cfg.items():
        env_cfg[k] = v
    env_template = ElevatorEnv(env_cfg)

    # Use smaller rollout for optuna search (config has 32768 for final training)
    rollout_steps = min(_cfg["ppo"]["rollout_steps"], 8192)

    trainer = PPOTrainer(
        state_dim=env_template.STATE_DIM,
        action_dim=env_template.action_space.n,
        lr=lr,
        gamma=_cfg["ppo"]["gamma"],
        gae_lambda=_cfg["ppo"]["gae_lambda"],
        clip_epsilon=clip_epsilon,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_start,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        seq_len=seq_len,
        burn_in_steps=burn_in_steps,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=0.1,  # fixed
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        weight_decay=weight_decay,
        activation=activation,
        device=device,
        num_envs=NUM_ENVS,
        use_amp=True,
        compile_policy=False,
        kl_early_stop=True,
        kl_target=kl_target,
        normalize_advantage=False,  # disabled: works better with dual optimizers
        normalize_rewards=False,
        actor_dropout=0.0,
        critic_dropout=0.0,
        use_layer_norm=True,
    )

    # LR scheduler: cosine with warmup (one per optimizer)
    warmup_epochs = 3
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.actor_optimizer, T_max=max(1, N_EPOCHS - warmup_epochs), eta_min=lr * 0.1)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.critic_optimizer, T_max=max(1, N_EPOCHS - warmup_epochs), eta_min=lr * 0.1)

    runner = MultiEnvRunner(env_template, NUM_ENVS, device)
    val_runner = MultiEnvRunner(env_template, NUM_ENVS, device)

    # Per-trial early stopping state
    best_val = -float("inf")
    no_improve = 0

    try:
        for epoch in range(N_EPOCHS):
            t0 = time.perf_counter()
            epoch_indices = np.random.permutation(_train_idx)

            n_updates = 0
            episode_count = 0
            train_reward = 0.0
            train_steps = 0
            feed_ptr = 0
            max_episodes = min(MAX_EPISODES_PER_EPOCH, len(_train_idx))

            # Feed initial episodes
            for i in range(NUM_ENVS):
                if episode_count >= max_episodes or feed_ptr >= len(epoch_indices):
                    runner.done[i] = True
                    continue
                while feed_ptr < len(epoch_indices) and episode_count < max_episodes:
                    idx = epoch_indices[feed_ptr]
                    feed_ptr += 1
                    events_trimmed = _event_seqs[idx][:int(_event_lens[idx])]
                    if len(events_trimmed) > 0:
                        runner.reset_env(i, events_trimmed, trainer.policy)
                        episode_count += 1
                        break
                else:
                    runner.done[i] = True

            trainer.policy.eval()

            # Collect rollouts
            while (feed_ptr < len(epoch_indices) and episode_count < max_episodes) or not runner.all_done:
                total_r, steps, _ = runner.step_all(trainer.policy, trainer.buffer)
                train_reward += total_r
                train_steps += steps

                for i in range(NUM_ENVS):
                    if runner.done[i] and feed_ptr < len(epoch_indices) and episode_count < max_episodes:
                        idx = epoch_indices[feed_ptr]
                        feed_ptr += 1
                        events_trimmed = _event_seqs[idx][:int(_event_lens[idx])]
                        if len(events_trimmed) > 0:
                            runner.reset_env(i, events_trimmed, trainer.policy)
                            episode_count += 1

                if trainer.buffer.is_ready(rollout_steps):
                    trainer.policy.train()
                    trainer.update(last_obs_per_env=runner.get_last_obs_per_env())
                    trainer.policy.eval()
                    n_updates += 1

            # Flush remaining buffer
            if trainer.buffer.size() >= rollout_steps // 2:
                trainer.policy.train()
                trainer.update(last_obs_per_env=runner.get_last_obs_per_env())
                trainer.policy.eval()
                n_updates += 1

            # Entropy anneal
            progress = epoch / max(N_EPOCHS - 1, 1)
            current_entropy = entropy_start + (entropy_end - entropy_start) * progress
            measured_entropy = trainer.stats.get("entropy", None)
            if entropy_floor > 0 and measured_entropy is not None and measured_entropy < entropy_floor:
                pass  # freeze entropy coef
            else:
                trainer.set_entropy_coef(current_entropy)

            # LR warmup + scheduler step
            if n_updates > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
                    if epoch < warmup_epochs:
                        warmup_ratio = 0.1
                        ramp = (epoch + 1) / max(warmup_epochs, 1)
                        warmup_lr = lr * (warmup_ratio + (1.0 - warmup_ratio) * ramp)
                        for pg in trainer.actor_optimizer.param_groups:
                            pg['lr'] = warmup_lr
                        for pg in trainer.critic_optimizer.param_groups:
                            pg['lr'] = warmup_lr
                    actor_scheduler.step()
                    critic_scheduler.step()

            # Logging
            loss_str = ""
            if trainer.stats:
                v_loss = trainer.stats.get("value_loss", float("nan"))
                p_loss = trainer.stats.get("policy_loss", float("nan"))
                ent = trainer.stats.get("entropy", float("nan"))
                clip = trainer.stats.get("clip_frac", 0.0)
                if not (np.isnan(v_loss) and np.isnan(p_loss)):
                    loss_str = (f" | v={v_loss:.2f} p={p_loss:.4f} "
                                f"ent={ent:.3f} clip={clip:.2f}")

            elapsed = time.perf_counter() - t0
            avg_reward = train_reward / max(train_steps, 1)
            print(f"  [Trial {trial.number:2d}] ep {epoch+1:>2d}/{N_EPOCHS} | "
                  f"{episode_count} ep | {n_updates} upd{loss_str} | "
                  f"r={avg_reward:.4f} | {elapsed:.1f}s", flush=True)

            # Validation + pruning + per-trial early stopping
            if epoch >= 5 and (epoch + 1) % 3 == 0:
                val_r = _validate(trainer, env_template, _event_seqs, _event_lens,
                                  _val_idx, num_envs=NUM_ENVS, runner=val_runner)
                trial.report(val_r, epoch)

                # Per-trial early stopping
                if val_r > best_val:
                    best_val = val_r
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= PATIENCE:
                    print(f"  [Trial {trial.number:2d}] early stop at ep {epoch+1} "
                          f"(no improve for {PATIENCE} checks, best={best_val:.4f})",
                          flush=True)
                    trial.set_user_attr("val_reward", float(best_val))
                    return best_val

                # Optuna pruner
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if trainer.stats and np.isnan(trainer.stats.get("value_loss", 0)):
                return -float("inf")

        # Final validation
        val_reward = _validate(trainer, env_template, _event_seqs, _event_lens,
                               _val_idx, runner=val_runner)
        trial.set_user_attr("val_reward", float(val_reward))

        print(f"[Trial {trial.number:2d}] done — val_reward={val_reward:.4f}\n", flush=True)
        return val_reward
    finally:
        _cleanup(runner, device)
        _cleanup(val_runner, device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _save_study_results(study):
    """Save best params and full study trials to checkpoints."""
    def _convert(v):
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print("No completed trials to save.", flush=True)
        return

    results = {
        "best_value": float(study.best_value),
        "best_params": {k: _convert(v) for k, v in study.best_params.items()},
    }
    out_path = PROJ_ROOT / "checkpoints" / "optuna_best_params.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved best params to {out_path}", flush=True)

    trials_data = []
    for t in study.trials:
        if t.state == TrialState.COMPLETE:
            trials_data.append({
                "number": t.number,
                "value": float(t.value) if t.value is not None else None,
                "params": {k: _convert(v) for k, v in t.params.items()},
                "duration": t.duration.total_seconds() if t.duration else None,
            })
    study_path = PROJ_ROOT / "checkpoints" / "optuna_study.json"
    with open(study_path, "w") as f:
        json.dump(trials_data, f, indent=2)
    print(f"Saved full study to {study_path}", flush=True)


def main(n_trials: int = 20):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"GPU: {gpu_name}  |  Free: {free_gb:.1f} GB  |  Trials: {n_trials}",
              flush=True)

    data = _load_data()
    print(f"Loaded {len(data['train_idx'])} train / {len(data['val_idx'])} val files",
          flush=True)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=3,
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    try:
        study.optimize(lambda trial: objective(trial, data),
                       n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...", flush=True)
        _save_study_results(study)
        os._exit(130)

    print("\n" + "=" * 60, flush=True)
    print("Best trial:", flush=True)
    print(f"  Val reward: {study.best_value:.4f}", flush=True)
    print("  Params:", flush=True)
    for k, v in study.best_params.items():
        print(f"    {k}: {v}", flush=True)
    print("=" * 60, flush=True)

    _save_study_results(study)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
