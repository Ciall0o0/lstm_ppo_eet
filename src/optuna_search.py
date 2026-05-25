import copy
import gc
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import optuna
from optuna.trial import TrialState

from src.utils import load_config, PROJ_ROOT, get_device, merge_reward_config
from src.data.dataset import split_indices
from src.env.elevator_env import ElevatorEnv
from src.models.lstm_ppo import PPOTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

NUM_ENVS = 32
ROLLOUT_STEPS_CAP = 512
MAX_EPISODES_PER_EPOCH = 24


class MultiEnvRunner:
    """Manages N parallel ElevatorEnv instances, batching GPU forward passes."""

    def __init__(self, env_template, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.state_dim = env_template.STATE_DIM
        self.envs = [copy.deepcopy(env_template) for _ in range(num_envs)]

        self.obs: list = [None] * num_envs
        self.hidden: list = [None] * num_envs
        self.done: list[bool] = [True] * num_envs
        self._is_lstm: bool | None = None

        self._obs_gpu = torch.empty(num_envs, self.state_dim, dtype=torch.float32, device=device)
        self._obs_pinned = torch.empty(num_envs, self.state_dim, dtype=torch.float32,
                                       pin_memory=(device.type == "cuda"))
        self._executor = ThreadPoolExecutor(max_workers=num_envs)

    def reset_env(self, i: int, events_trimmed, policy):
        obs, _ = self.envs[i].reset(options={"events": events_trimmed})
        self.obs[i] = obs
        self.hidden[i] = policy.get_initial_hidden(1, self.device)
        self.done[i] = False
        if self._is_lstm is None:
            self._is_lstm = isinstance(self.hidden[i], tuple)

    def step_all(self, policy, buffer) -> tuple[float, int, int]:
        active = [i for i in range(self.num_envs) if not self.done[i]]
        if not active:
            return 0.0, 0, 0

        n_active = len(active)

        obs_stack = np.stack([self.obs[i] for i in active])
        self._obs_pinned[:n_active].copy_(torch.from_numpy(obs_stack))
        self._obs_gpu[:n_active].copy_(self._obs_pinned[:n_active], non_blocking=True)

        batched_hidden = self._batch_hidden(active)
        obs_seq = self._obs_gpu[:n_active].unsqueeze(1)

        with torch.no_grad():
            actions, log_probs, values, new_hidden = policy.get_action(
                obs_seq, hidden=batched_hidden)

        self._unbatch_hidden(active, new_hidden)

        action_ints: list[int] = actions.squeeze(-1).tolist()

        total_reward = 0.0
        total_steps = 0
        n_done = 0

        envs = self.envs
        _obs = self.obs
        _done_flags = self.done
        _obs_gpu = self._obs_gpu

        futures = {}
        for j, i in enumerate(active):
            futures[self._executor.submit(envs[i].step, action_ints[j])] = (j, i)

        for future in as_completed(futures):
            j, i = futures[future]
            next_obs, reward, done, _, _ = future.result()

            buffer.add_fast(i, _obs_gpu[j], action_ints[j], reward,
                            values[j], log_probs[j], done)

            _obs[i] = next_obs
            total_reward += reward
            total_steps += 1

            if done:
                _done_flags[i] = True
                n_done += 1

        return total_reward, total_steps, n_done

    def _batch_hidden(self, active: list[int]):
        if self._is_lstm:
            h_list = [self.hidden[i][0] for i in active]
            c_list = [self.hidden[i][1] for i in active]
            return (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))
        else:
            return torch.cat([self.hidden[i] for i in active], dim=1)

    def _unbatch_hidden(self, active: list[int], new_hidden):
        if self._is_lstm:
            new_h, new_c = new_hidden
            for j, i in enumerate(active):
                self.hidden[i] = (new_h[:, j:j + 1, :], new_c[:, j:j + 1, :])
        else:
            for j, i in enumerate(active):
                self.hidden[i] = new_hidden[:, j:j + 1, :]

    @property
    def all_done(self) -> bool:
        return all(self.done)

    def get_last_obs_per_env(self) -> list:
        return [self.obs[i] if not self.done[i] else None for i in range(self.num_envs)]


# Load data once — immutable across trials
_cfg = load_config()
_datasets_dir = PROJ_ROOT / _cfg["data"]["datasets_dir"]
_ds = str(_datasets_dir)
_LABELS = np.squeeze(np.load(f"{_ds}/labels.npz", allow_pickle=True)["arr_0"])
_EVENT_SEQS = np.load(f"{_ds}/event_sequences.npz", allow_pickle=True)["arr_0"]
_EVENT_LENS = np.load(f"{_ds}/event_lengths.npz", allow_pickle=True)["arr_0"]
_TRAIN_IDX, _VAL_IDX, _ = split_indices(
    _LABELS, _cfg["data"]["train_ratio"], _cfg["data"]["val_ratio"],
    _cfg["data"]["random_seed"])


def _validate(trainer, env, event_seqs, event_lens, val_idx):
    trainer.policy.eval()
    total_reward = 0.0
    total_steps = 0
    obs_gpu = torch.empty(env.STATE_DIM, dtype=torch.float32, device=trainer.device)
    with torch.inference_mode():
        for idx in val_idx:
            events = event_seqs[idx]
            length = event_lens[idx]
            events_trimmed = events[:int(length)]
            if len(events_trimmed) == 0:
                continue
            obs, _ = env.reset(options={"events": events_trimmed})
            done = False
            hidden = trainer.policy.get_initial_hidden(1, trainer.device)
            while not done:
                obs_gpu.copy_(torch.from_numpy(obs), non_blocking=True)
                obs_seq = obs_gpu.unsqueeze(0).unsqueeze(0)
                action, _, _, hidden = trainer.policy.get_action(
                    obs_seq, hidden=hidden, deterministic=True)
                obs, reward, done, _, _ = env.step(action.item())
                total_reward += reward
                total_steps += 1
    trainer.policy.train()
    return total_reward / max(total_steps, 1)


def objective(trial: optuna.Trial) -> float:
    device = torch.device(get_device())
    n_epochs = 5

    print(f"\n{'='*60}\n[Trial {trial.number:2d}] starting\n{'='*60}", flush=True)

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    entropy_start = trial.suggest_float("entropy_coef_start", 0.01, 0.20)
    entropy_end = trial.suggest_float("entropy_coef_end", 0.001, 0.05)
    max_grad_norm = trial.suggest_float("max_grad_norm", 1.0, 10.0)
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
    value_loss_coef = trial.suggest_float("value_loss_coef", 0.1, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    ppo_epochs = trial.suggest_int("ppo_epochs", 4, 15)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    seq_len = trial.suggest_categorical("seq_len", [16, 32, 64])

    lstm_hidden = trial.suggest_categorical("lstm_hidden", [128, 256, 384])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.0, 0.3)
    actor_hidden = trial.suggest_categorical("actor_hidden", [64, 128, 256])
    critic_hidden = trial.suggest_categorical("critic_hidden", [64, 128, 256])
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
    encoder_type = trial.suggest_categorical("encoder_type", ["lstm", "gru"])

    env_cfg = merge_reward_config(dict(_cfg.get("env", {})), _cfg)
    env_cfg.update({
        "passenger_delivered": trial.suggest_float("passenger_delivered", 10.0, 100.0),
        "wait_time_per_sec": trial.suggest_float("wait_time_per_sec", -0.05, -0.005),
        "empty_distance_per_floor": trial.suggest_float("empty_distance_per_floor", -0.1, -0.005),
        "assignment_dist_per_floor": trial.suggest_float("assignment_dist_per_floor", -3.0, -0.1),
        "energy_per_start_stop": trial.suggest_float("energy_per_start_stop", -0.05, -0.001),
        "idle_penalty_per_sec": trial.suggest_float("idle_penalty_per_sec", -0.02, 0.0),
    })
    env_template = ElevatorEnv(env_cfg)

    rollout_steps = min(ROLLOUT_STEPS_CAP, _cfg["ppo"]["rollout_steps"])
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
        encoder_type=encoder_type,
        activation=activation,
        device=device,
        num_envs=NUM_ENVS,
        use_amp=(device.type == "cuda"),
        kl_early_stop=False,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=n_epochs, eta_min=lr * 0.1)

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        epoch_indices = np.random.permutation(_TRAIN_IDX)
        runner = MultiEnvRunner(env_template, NUM_ENVS, device)

        n_updates = 0
        episode_count = 0
        train_reward = 0.0
        train_steps = 0
        feed_ptr = 0
        max_episodes = min(MAX_EPISODES_PER_EPOCH, len(_TRAIN_IDX))

        for i in range(NUM_ENVS):
            while feed_ptr < len(epoch_indices) and episode_count < max_episodes:
                idx = epoch_indices[feed_ptr]
                feed_ptr += 1
                events_trimmed = _EVENT_SEQS[idx][:int(_EVENT_LENS[idx])]
                if len(events_trimmed) > 0:
                    runner.reset_env(i, events_trimmed, trainer.policy)
                    episode_count += 1
                    break

        while (feed_ptr < len(epoch_indices) and episode_count < max_episodes) or not runner.all_done:
            total_r, steps, _ = runner.step_all(trainer.policy, trainer.buffer)
            train_reward += total_r
            train_steps += steps

            for i in range(NUM_ENVS):
                if runner.done[i] and feed_ptr < len(epoch_indices) and episode_count < max_episodes:
                    idx = epoch_indices[feed_ptr]
                    feed_ptr += 1
                    events_trimmed = _EVENT_SEQS[idx][:int(_EVENT_LENS[idx])]
                    if len(events_trimmed) > 0:
                        runner.reset_env(i, events_trimmed, trainer.policy)
                        episode_count += 1

            if trainer.buffer.is_ready(rollout_steps):
                trainer.update(last_obs_per_env=runner.get_last_obs_per_env())
                n_updates += 1

        if trainer.buffer.size() >= trainer.batch_size:
            trainer.update(last_obs_per_env=runner.get_last_obs_per_env())
            n_updates += 1

        runner._executor.shutdown(wait=True)

        progress = epoch / max(n_epochs - 1, 1)
        trainer.set_entropy_coef(entropy_start + (entropy_end - entropy_start) * progress)
        if n_updates > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")
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

        trial.report(avg_reward, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if trainer.stats and np.isnan(trainer.stats.get("value_loss", 0)):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return -float("inf")

    val_reward = _validate(trainer, env_template, _EVENT_SEQS, _EVENT_LENS, _VAL_IDX)
    trial.set_user_attr("val_reward", float(val_reward))

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[Trial {trial.number:2d}] done — val_reward={val_reward:.3f}\n", flush=True)
    return val_reward


def main():
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=30, show_progress_bar=False)

    print("\n" + "=" * 60, flush=True)
    print("Best trial:", flush=True)
    print(f"  Val reward: {study.best_value:.4f}", flush=True)
    print("  Params:", flush=True)
    for k, v in study.best_params.items():
        print(f"    {k}: {v}", flush=True)
    print("=" * 60, flush=True)

    # Save best params
    def _convert(v):
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v

    results = {
        "best_value": float(study.best_value),
        "best_params": {k: _convert(v) for k, v in study.best_params.items()},
    }
    out_path = PROJ_ROOT / "checkpoints" / "optuna_best_params.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved best params to {out_path}", flush=True)

    # Save full study trials for analysis
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


if __name__ == "__main__":
    main()
