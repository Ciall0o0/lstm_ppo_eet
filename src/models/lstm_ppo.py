"""PPO trainer with LSTM/GRU support, GAE advantage estimation, AMP, and KL early stopping."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .policy import LSTMActorCritic


class RolloutBuffer:
    """Stores trajectories on GPU partitioned by env for correct per-trajectory GAE."""

    def __init__(self, max_steps: int, state_dim: int, num_envs: int = 1, device=None):
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.per_env = max_steps // num_envs
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs = torch.zeros(max_steps, state_dim, device=device)
        self.actions = torch.zeros(max_steps, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(max_steps, device=device)
        self.values = torch.zeros(max_steps, device=device)
        self.log_probs = torch.zeros(max_steps, device=device)
        self.dones = torch.zeros(max_steps, device=device)
        self.masks = torch.ones(max_steps, device=device)
        self.head = [i * self.per_env for i in range(num_envs)]
        self._size = 0

    def add(self, env_id: int, obs, action: int, reward: float,
            value, log_prob, done: bool, mask: float = 1.0):
        ptr = self.head[env_id]
        end = (env_id + 1) * self.per_env
        if ptr >= end:
            return  # type: ignore[unreachable]
        if isinstance(obs, np.ndarray):
            self.obs[ptr] = torch.as_tensor(obs, device=self.device)
        else:
            self.obs[ptr] = obs.detach().clone().to(self.device)
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.values[ptr] = value.detach() if isinstance(value, torch.Tensor) else value
        self.log_probs[ptr] = log_prob.detach() if isinstance(log_prob, torch.Tensor) else log_prob
        self.dones[ptr] = float(done)
        self.masks[ptr] = mask
        self.head[env_id] = ptr + 1
        self._size += 1

    def add_fast(self, env_id: int, obs_gpu: torch.Tensor, action: int, reward: float,
                 value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """Fast path: GPU tensor obs, pre-extracted values (no isinstance checks)."""
        ptr = self.head[env_id]
        end = (env_id + 1) * self.per_env
        if ptr >= end:
            return  # partition full, drop
        self.obs[ptr] = obs_gpu
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.values[ptr] = value
        self.log_probs[ptr] = log_prob
        self.dones[ptr] = float(done)
        self.masks[ptr] = 1.0
        self.head[env_id] = ptr + 1
        self._size += 1

    def get_env_slice(self, env_id: int) -> tuple[int, int]:
        start = env_id * self.per_env
        end = self.head[env_id]
        return start, end

    def env_size(self, env_id: int) -> int:
        return self.head[env_id] - env_id * self.per_env

    def size(self) -> int:
        return self._size

    def is_ready(self, min_total: int) -> bool:
        return self._size >= min_total

    def clear(self):
        self.head = [i * self.per_env for i in range(self.num_envs)]
        self._size = 0


class PPOTrainer:
    """PPO trainer with recurrent encoder, GAE, AMP, and KL early stopping."""

    def __init__(
        self,
        state_dim: int = 73,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        seq_len: int = 32,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
        lstm_dropout: float = 0.0,
        weight_decay: float = 0.0,
        num_envs: int = 1,
        compile_policy: bool = False,
        device: str = "cuda",
        encoder_type: str = "lstm",
        activation: str = "relu",
        use_head_layernorm: bool = False,
        use_amp: bool = False,
        kl_target: float = 0.01,
        kl_early_stop: bool = True,
        normalize_advantage: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.seq_len = seq_len
        self.num_envs = num_envs
        self.kl_target = kl_target
        self.kl_early_stop = kl_early_stop
        self.normalize_advantage = normalize_advantage
        self.use_amp = use_amp
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = LSTMActorCritic(
            state_dim=state_dim, action_dim=action_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            actor_hidden=actor_hidden, critic_hidden=critic_hidden,
            encoder_type=encoder_type,
            activation=activation,
            use_head_layernorm=use_head_layernorm,
        ).to(self.device)

        if compile_policy and hasattr(torch, 'compile'):
            try:
                self.policy = torch.compile(self.policy)  # type: ignore[assignment]
            except Exception:
                pass

        use_fused = self.device.type == 'cuda'
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=lr,  # type: ignore
            weight_decay=weight_decay, fused=use_fused,
        )

        self.scaler = torch.amp.GradScaler('cuda') if (use_amp and self.device.type == 'cuda') else None

        self.buffer = RolloutBuffer(rollout_steps, state_dim, num_envs=num_envs, device=self.device)
        self.stats: dict = {}

    def set_entropy_coef(self, coef: float):
        self.entropy_coef = coef

    @classmethod
    def from_config(cls, state_dim: int, action_dim: int,
                    cfg: dict, device: str = "cuda") -> "PPOTrainer":
        ppo_cfg = cfg.get("ppo", {})
        model_cfg = cfg.get("model", {})
        training_cfg = cfg.get("training", {})
        num_envs = training_cfg.get("num_envs", 1)
        rollout_steps = ppo_cfg.get("rollout_steps", 2048)
        if rollout_steps % num_envs != 0:
            rollout_steps = (rollout_steps // num_envs) * num_envs
        return cls(
            state_dim=state_dim, action_dim=action_dim,
            lr=ppo_cfg.get("learning_rate", 3e-4),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_epsilon=ppo_cfg.get("clip_epsilon", 0.2),
            value_loss_coef=ppo_cfg.get("value_loss_coef", 0.5),
            entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            ppo_epochs=ppo_cfg.get("ppo_epochs", 10),
            batch_size=ppo_cfg.get("batch_size", 64),
            rollout_steps=rollout_steps,
            seq_len=ppo_cfg.get("seq_len", 32),
            lstm_hidden=model_cfg.get("lstm_hidden", 128),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            lstm_dropout=model_cfg.get("lstm_dropout", 0.0),
            actor_hidden=model_cfg.get("actor_hidden", 64),
            critic_hidden=model_cfg.get("critic_hidden", 64),
            weight_decay=ppo_cfg.get("weight_decay", 0.0),
            num_envs=num_envs,
            compile_policy=ppo_cfg.get("compile_policy", False),
            device=device,
            encoder_type=model_cfg.get("encoder_type", "lstm"),
            activation=model_cfg.get("activation", "relu"),
            use_head_layernorm=model_cfg.get("use_head_layernorm", False),
            use_amp=ppo_cfg.get("use_amp", False),
            kl_target=ppo_cfg.get("kl_target", 0.01),
            kl_early_stop=ppo_cfg.get("kl_early_stop", True),
            normalize_advantage=ppo_cfg.get("normalize_advantage", True),
        )

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, masks: torch.Tensor,
                    last_value: torch.Tensor | float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        T = len(rewards)
        next_values = torch.empty_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = last_value
        next_values = next_values * (1.0 - dones)

        deltas = rewards + self.gamma * next_values - values

        advantages = torch.zeros(T, device=rewards.device)
        gae = 0.0
        discount = self.gamma * self.gae_lambda
        for t in reversed(range(T)):
            gae = deltas[t] + discount * (1.0 - dones[t]) * masks[t] * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_obs_per_env: list | None = None):
        total_steps = self.buffer.size()
        if total_steps < self.batch_size:
            return {}

        all_advantages = []
        all_returns = []
        all_obs_segs = []
        all_act_segs = []
        all_old_lp_segs = []
        all_old_v_segs = []
        all_mask_segs = []

        last_obs_indices = []
        last_obs_list = []
        for i in range(self.num_envs):
            if last_obs_per_env and last_obs_per_env[i] is not None:
                last_obs_indices.append(i)
                last_obs_list.append(last_obs_per_env[i])

        last_values = [torch.tensor(0.0, device=self.device)] * self.num_envs
        if last_obs_list:
            with torch.no_grad():
                obs_stack = np.stack(last_obs_list)
                o_batch = torch.as_tensor(obs_stack, dtype=torch.float32, device=self.device)
                o_batch = o_batch.unsqueeze(1)
                _, v_batch, _ = self.policy.forward(o_batch)  # type: ignore
                for j, i in enumerate(last_obs_indices):
                    last_values[i] = v_batch[j].squeeze()

        for i in range(self.num_envs):
            start, end = self.buffer.get_env_slice(i)
            n = end - start
            if n < self.seq_len:
                continue

            rewards = self.buffer.rewards[start:end]
            values = self.buffer.values[start:end]
            dones = self.buffer.dones[start:end]
            masks = self.buffer.masks[start:end]

            adv, ret = self.compute_gae(rewards, values, dones, masks, last_values[i])

            usable = (n // self.seq_len) * self.seq_len
            n_seg = usable // self.seq_len

            all_obs_segs.append(self.buffer.obs[start:start + usable].view(n_seg, self.seq_len, -1))
            all_act_segs.append(self.buffer.actions[start:start + usable].view(n_seg, self.seq_len))
            all_old_lp_segs.append(self.buffer.log_probs[start:start + usable].view(n_seg, self.seq_len))
            all_old_v_segs.append(values[:usable].view(n_seg, self.seq_len))
            all_adv_segs = adv[:usable].view(n_seg, self.seq_len)
            all_ret_segs = ret[:usable].view(n_seg, self.seq_len)
            all_mask_segs.append(masks[:usable].view(n_seg, self.seq_len))
            all_advantages.append(all_adv_segs)
            all_returns.append(all_ret_segs)

        if not all_obs_segs:
            self.buffer.clear()
            return {}

        obs_segs = torch.cat(all_obs_segs)
        actions_segs = torch.cat(all_act_segs)
        old_lp_segs = torch.cat(all_old_lp_segs)
        old_v_segs = torch.cat(all_old_v_segs)
        adv_segs = torch.cat(all_advantages)
        ret_segs = torch.cat(all_returns)
        mask_segs = torch.cat(all_mask_segs)

        if self.normalize_advantage:
            adv_std, adv_mean = torch.std_mean(adv_segs, correction=1)
            adv_segs = (adv_segs - adv_mean) / (adv_std + 1e-8)

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        n_updates = 0

        n_segments = obs_segs.size(0)
        segments_per_mb = max(1, self.batch_size // self.seq_len)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_segments, device=self.device)

            for start in range(0, n_segments, segments_per_mb):
                batch_idx = perm[start:start + segments_per_mb]

                obs_b = obs_segs[batch_idx]
                act_b = actions_segs[batch_idx]
                old_lp_b = old_lp_segs[batch_idx]
                old_v_b = old_v_segs[batch_idx]
                adv_b = adv_segs[batch_idx]
                ret_b = ret_segs[batch_idx]
                mask_b = mask_segs[batch_idx]

                use_amp_step = self.scaler is not None
                with torch.amp.autocast('cuda', enabled=use_amp_step):
                    _, new_lp, new_v, ent, _ = self.policy.evaluate_actions(obs_b, act_b)

                    ratio = torch.exp(new_lp - old_lp_b)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                    policy_loss = -(torch.min(surr1, surr2) * mask_b).sum() / mask_b.sum()

                    v_clipped = old_v_b + torch.clamp(new_v - old_v_b, -self.clip_epsilon, self.clip_epsilon)
                    v_loss_unclipped = (new_v - ret_b) ** 2
                    v_loss_clipped = (v_clipped - ret_b) ** 2
                    value_loss = (torch.max(v_loss_unclipped, v_loss_clipped) * mask_b).sum() / mask_b.sum()

                    ent_loss = (ent * mask_b).sum() / mask_b.sum()

                    loss = (policy_loss + self.value_loss_coef * value_loss
                            - self.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_entropy += ent_loss.detach()
                n_updates += 1

            # KL early stopping: compute approximate KL after each PPO epoch
            if self.kl_early_stop and n_segments > 0:
                with torch.no_grad():
                    _, kl_logp, _, _, _ = self.policy.evaluate_actions(obs_segs, actions_segs)
                    approx_kl = (old_lp_segs - kl_logp).mean().abs()
                if approx_kl > self.kl_target * 1.5:
                    break

        self.buffer.clear()

        self.stats = {
            "policy_loss": (total_policy_loss / max(n_updates, 1)).item(),
            "value_loss": (total_value_loss / max(n_updates, 1)).item(),
            "entropy": (total_entropy / max(n_updates, 1)).item(),
            "n_updates": n_updates,
        }
        return self.stats

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "stats": self.stats,
            "encoder_type": self.policy.encoder_type,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.stats = ckpt.get("stats", {})


if __name__ == "__main__":
    print("Testing LSTM/GRU+PPO trainer initialization and forward/backward pass...")

    trainer = PPOTrainer(
        state_dim=73, action_dim=3, lstm_hidden=128, lstm_layers=2,
        rollout_steps=512, batch_size=64, seq_len=32, num_envs=2, device="cuda",
        encoder_type="lstm", activation="relu", use_amp=True,
    )

    env_dim = trainer.state_dim
    per_env = trainer.buffer.per_env
    for env_id in range(2):
        n = min(64, per_env)
        start = env_id * per_env
        end = start + n
        trainer.buffer.obs[start:end] = torch.randn(n, env_dim, device=trainer.device)
        trainer.buffer.actions[start:end] = torch.randint(0, 3, (n,), device=trainer.device)
        trainer.buffer.rewards[start:end] = torch.randn(n, device=trainer.device)
        trainer.buffer.values[start:end] = torch.randn(n, device=trainer.device)
        trainer.buffer.log_probs[start:end] = torch.randn(n, device=trainer.device)
        trainer.buffer.dones[start:end] = torch.zeros(n, device=trainer.device)
        trainer.buffer.masks[start:end] = torch.ones(n, device=trainer.device)
        trainer.buffer.head[env_id] = end
        trainer.buffer._size += n

    stats = trainer.update(last_obs_per_env=[None, None])
    print(f"PPO update stats: {stats}")

    trainer.save("/tmp/test_ppo.pt")
    trainer.load("/tmp/test_ppo.pt")
    print("Save/Load: OK")
    print("Model verification passed.")
