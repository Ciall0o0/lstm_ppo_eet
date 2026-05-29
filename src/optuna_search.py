"""Optuna hyperparameter search for LSTM+PPO elevator scheduling.

AMP is enabled by default for faster GPU inference/training via Tensor Cores.
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

torch.backends.cudnn.benchmark = False  # variable-length RNN: benchmark adds overhead
torch.set_float32_matmul_precision("high")

# ---- constants used inside objective() ----
NUM_ENVS = 32             # match train.py num_envs
MAX_EPISODES_PER_EPOCH = 16  # reduced for faster search
PRUNING_VAL_SIZE = 20     # overridden in objective to use full val set


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
    # Reset done flags for reuse
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
    n_epochs = 30

    _cfg = data["cfg"]
    _event_seqs = data["event_seqs"]
    _event_lens = data["event_lens"]
    _train_idx = data["train_idx"]
    _val_idx = data["val_idx"]

    print(f"\n{'='*60}\n[Trial {trial.number:2d}] starting\n{'='*60}", flush=True)

    lr = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    entropy_start = trial.suggest_float("entropy_coef_start", 0.02, 0.10)
    entropy_end = trial.suggest_float("entropy_coef_end", 0.005, entropy_start)
    max_grad_norm = trial.suggest_float("max_grad_norm", 1.0, 3.0)
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.15, 0.3)
    value_loss_coef = trial.suggest_float("value_loss_coef", 0.2, 0.8)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    ppo_epochs = trial.suggest_int("ppo_epochs", 4, 8)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    seq_len = trial.suggest_categorical("seq_len", [16, 32, 64])

    # Fixed architecture params (not searched)
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [128, 256])
    lstm_layers = 1
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.15)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    actor_hidden = hidden_dim
    critic_hidden = hidden_dim
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])

    env_cfg = merge_reward_config(dict(_cfg.get("env", {})), _cfg)
    # Reward weights are read from config, not searched — they are a design choice
    env_template = ElevatorEnv(env_cfg)

    rollout_steps = _cfg["ppo"]["rollout_steps"]
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
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        weight_decay=weight_decay,
        activation=activation,
        device=device,
        num_envs=NUM_ENVS,
        use_amp=True,
        compile_policy=False,  # torch.compile applied below
        kl_early_stop=_cfg["ppo"].get("kl_early_stop", True),
        kl_target=_cfg["ppo"].get("kl_target", 0.01),
        normalize_advantage=_cfg["ppo"].get("normalize_advantage", True),
    )

    # LR scheduler: cosine with warmup (matches train.py pattern)
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=5e-5)

    # Enable torch.compile (overhead amortized over 20 epochs)
    if trainer.device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            trainer.policy = torch.compile(trainer.policy)  # type: ignore[assignment]
        except Exception:
            pass

    runner = MultiEnvRunner(env_template, NUM_ENVS, device)
    val_runner = MultiEnvRunner(env_template, NUM_ENVS, device)

    try:
        for epoch in range(n_epochs):
            t0 = time.perf_counter()
            epoch_indices = np.random.permutation(_train_idx)

            n_updates = 0
            episode_count = 0
            train_reward = 0.0
            train_steps = 0
            feed_ptr = 0
            max_episodes = min(MAX_EPISODES_PER_EPOCH, len(_train_idx))

            for i in range(NUM_ENVS):
                if episode_count >= max_episodes or feed_ptr >= len(epoch_indices):
                    # No more episodes — mark remaining envs as done
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
                    # Couldn't find a valid episode for this env — mark as done
                    runner.done[i] = True

            trainer.policy.eval()

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

            if trainer.buffer.size() >= rollout_steps // 2:
                trainer.policy.train()
                trainer.update(last_obs_per_env=runner.get_last_obs_per_env())
                trainer.policy.eval()
                n_updates += 1

            progress = epoch / max(n_epochs - 1, 1)
            trainer.set_entropy_coef(entropy_start + (entropy_end - entropy_start) * progress)

            # LR warmup: linear ramp from lr*0.1 to lr
            if n_updates > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")
                    if epoch < warmup_epochs:
                        warmup_ratio = 0.1
                        ramp = (epoch + 1) / max(warmup_epochs, 1)
                        warmup_lr = lr * (warmup_ratio + (1.0 - warmup_ratio) * ramp)
                        for param_group in trainer.optimizer.param_groups:
                            param_group['lr'] = warmup_lr
                        scheduler.step()
                    else:
                        scheduler.step()

            loss_str = ""
            if trainer.stats:
                v_loss = trainer.stats.get("value_loss", float("nan"))
                p_loss = trainer.stats.get("policy_loss", float("nan"))
                if not (np.isnan(v_loss) and np.isnan(p_loss)):
                    loss_str = f" | v_loss={v_loss:.3f} p_loss={p_loss:.3f}"

            elapsed = time.perf_counter() - t0
            avg_reward = train_reward / max(train_steps, 1)
            print(f"  [Trial {trial.number:2d}] epoch {epoch + 1}/{n_epochs} | "
                  f"{episode_count} episodes | {n_updates} updates{loss_str} | "
                  f"avg_r={avg_reward:.4f} | {elapsed:.1f}s", flush=True)

            # Skip validation in first 5 epochs (pruner warmup + warmup period)
            if epoch >= 5:
                prune_val_r = _validate(trainer, env_template, _event_seqs, _event_lens,
                                        _val_idx, num_envs=NUM_ENVS, runner=val_runner)
                trial.report(prune_val_r, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if trainer.stats and np.isnan(trainer.stats.get("value_loss", 0)):
                return -float("inf")

        val_reward = _validate(trainer, env_template, _event_seqs, _event_lens, _val_idx,
                               runner=val_runner)
        trial.set_user_attr("val_reward", float(val_reward))

        print(f"[Trial {trial.number:2d}] done — val_reward={val_reward:.3f}\n", flush=True)
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

    # Only save if there's at least one completed trial
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


def main(n_trials: int = 30):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"GPU: {gpu_name}  |  Free: {free_gb:.1f} GB  |  Trials: {n_trials}",
              flush=True)

    # Load data once — shared across all trials
    data = _load_data()
    print(f"Loaded {len(data['train_idx'])} train / {len(data['val_idx'])} val files",
          flush=True)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    try:
        study.optimize(lambda trial: objective(trial, data),
                       n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...", flush=True)
        _save_study_results(study)
        # Force exit — don't wait for optuna's internal cleanup threads
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
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()
    main(n_trials=args.n_trials)
