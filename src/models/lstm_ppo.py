"""PPO trainer with LSTM support and GAE advantage estimation."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .policy import LSTMActorCritic


class RolloutBuffer:
    """Stores trajectories on GPU for zero-copy PPO updates."""

    def __init__(self, max_steps: int, state_dim: int, seq_len: int = 32,
                 device: torch.device = torch.device("cpu")):
        self.max_steps = max_steps
        self.seq_len = seq_len
        self.device = device
        self.obs = torch.zeros(max_steps, state_dim, device=device)
        self.actions = torch.zeros(max_steps, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(max_steps, device=device)
        self.values = torch.zeros(max_steps, device=device)
        self.log_probs = torch.zeros(max_steps, device=device)
        self.dones = torch.zeros(max_steps, device=device)
        self.masks = torch.ones(max_steps, device=device)
        self.ptr = 0
        self.full = False

    def add(self, obs, action: int, reward: float,
            value, log_prob, done: bool, mask: float = 1.0):
        if self.ptr >= self.max_steps:
            self.full = True
            self.ptr = 0
        if isinstance(obs, np.ndarray):
            self.obs[self.ptr] = torch.as_tensor(obs, device=self.device)
        else:
            self.obs[self.ptr] = obs.detach().to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value.detach() if isinstance(value, torch.Tensor) else value
        self.log_probs[self.ptr] = log_prob.detach() if isinstance(log_prob, torch.Tensor) else log_prob
        self.dones[self.ptr] = float(done)
        self.masks[self.ptr] = mask
        self.ptr += 1

    def size(self) -> int:
        return self.max_steps if self.full else self.ptr

    def clear(self):
        self.ptr = 0
        self.full = False

    def get_batch(self) -> dict:
        s = self.size()
        return {
            "obs": self.obs[:s],
            "actions": self.actions[:s],
            "rewards": self.rewards[:s],
            "values": self.values[:s],
            "log_probs": self.log_probs[:s],
            "dones": self.dones[:s],
            "masks": self.masks[:s],
        }


class PPOTrainer:
    """PPO trainer with LSTM sequence processing and GAE."""

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
        device: str = "cpu",
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
        self.device = torch.device(device)

        self.policy = LSTMActorCritic(
            state_dim=state_dim, action_dim=action_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            actor_hidden=actor_hidden, critic_hidden=critic_hidden,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = RolloutBuffer(rollout_steps, state_dim, seq_len, device=self.device)
        self.stats: dict = {}

    @classmethod
    def from_config(cls, state_dim: int, action_dim: int,
                    cfg: dict, device: str = "cpu") -> "PPOTrainer":
        ppo_cfg = cfg.get("ppo", {})
        model_cfg = cfg.get("model", {})
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
            rollout_steps=ppo_cfg.get("rollout_steps", 2048),
            seq_len=ppo_cfg.get("seq_len", 32),
            lstm_hidden=model_cfg.get("lstm_hidden", 128),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            lstm_dropout=model_cfg.get("lstm_dropout", 0.0),
            actor_hidden=model_cfg.get("actor_hidden", 64),
            critic_hidden=model_cfg.get("critic_hidden", 64),
            device=device,
        )

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, masks: torch.Tensor,
                    last_value: torch.Tensor | float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE and returns. Deltas are vectorized; recurrence is serial."""
        T = len(rewards)

        # Vectorized: build next_values and compute deltas in one pass
        next_values = torch.empty_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = last_value
        next_values = next_values * (1.0 - dones)

        deltas = rewards + self.gamma * next_values - values

        # Sequential GAE recurrence (inherently serial)
        advantages = torch.zeros(T, device=rewards.device)
        gae = 0.0
        discount = self.gamma * self.gae_lambda
        for t in reversed(range(T)):
            gae = deltas[t] + discount * (1.0 - dones[t]) * masks[t] * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_obs: np.ndarray | None = None):
        """Perform PPO update on collected rollout data with batched segments."""
        batch = self.buffer.get_batch()
        total_steps = self.buffer.size()
        if total_steps < self.batch_size:
            return {}

        # Data already on device from GPU buffer — no .to(device) copies
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        values = batch["values"]
        old_log_probs = batch["log_probs"]
        dones = batch["dones"]
        masks = batch["masks"]

        # Compute last value for GAE (keep on GPU, no .item())
        last_value = torch.tensor(0.0, device=self.device)
        if last_obs is not None:
            with torch.no_grad():
                o = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
                o = o.unsqueeze(0).unsqueeze(0)
                _, v, _ = self.policy.forward(o)
                last_value = v.squeeze()

        advantages, returns = self.compute_gae(rewards, values, dones, masks, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # GPU-resident accumulators — sync only at the end
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        n_updates = 0

        # Reshape into segments: (n_segments, seq_len, D)
        usable_steps = (total_steps // self.seq_len) * self.seq_len
        n_segments = usable_steps // self.seq_len

        obs_segs = obs[:usable_steps].view(n_segments, self.seq_len, -1)
        actions_segs = actions[:usable_steps].view(n_segments, self.seq_len)
        old_lp_segs = old_log_probs[:usable_steps].view(n_segments, self.seq_len)
        adv_segs = advantages[:usable_steps].view(n_segments, self.seq_len)
        ret_segs = returns[:usable_steps].view(n_segments, self.seq_len)
        mask_segs = masks[:usable_steps].view(n_segments, self.seq_len)

        segments_per_mb = max(1, self.batch_size // self.seq_len)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_segments, device=self.device)

            for start in range(0, n_segments, segments_per_mb):
                batch_idx = perm[start:start + segments_per_mb]

                # Batch-index into pre-reshaped segments → (B, seq_len, D)
                obs_b = obs_segs[batch_idx]
                act_b = actions_segs[batch_idx]
                old_lp_b = old_lp_segs[batch_idx]
                adv_b = adv_segs[batch_idx]
                ret_b = ret_segs[batch_idx]
                mask_b = mask_segs[batch_idx]

                # Single batched forward pass
                _, new_lp, new_v, ent, _ = self.policy.evaluate_actions(obs_b, act_b)

                # PPO clipped loss (mask-weighted)
                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                policy_loss = -(torch.min(surr1, surr2) * mask_b).sum() / mask_b.sum()

                value_diff = (new_v - ret_b) * mask_b
                value_loss = (value_diff ** 2).sum() / mask_b.sum()

                ent_loss = (ent * mask_b).sum() / mask_b.sum()

                loss = (policy_loss + self.value_loss_coef * value_loss
                        - self.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_entropy += ent_loss.detach()
                n_updates += 1

        self.buffer.clear()

        self.stats = {
            "policy_loss": (total_policy_loss / max(n_updates, 1)).item(),
            "value_loss": (total_value_loss / max(n_updates, 1)).item(),
            "entropy": (total_entropy / max(n_updates, 1)).item(),
            "n_updates": n_updates,
        }
        return self.stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "stats": self.stats,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.stats = ckpt.get("stats", {})


if __name__ == "__main__":
    print("Testing LSTM+PPO trainer initialization and forward/backward pass...")

    trainer = PPOTrainer(
        state_dim=73, action_dim=3, lstm_hidden=128, lstm_layers=2,
        rollout_steps=512, batch_size=64, seq_len=32, device="cpu",
    )

    # Simulate a rollout
    env_dim = trainer.state_dim
    obs = torch.randn(128, env_dim, device=trainer.device)
    trainer.buffer.obs[:128] = obs
    trainer.buffer.actions[:128] = torch.randint(0, 3, (128,), device=trainer.device)
    trainer.buffer.rewards[:128] = torch.randn(128, device=trainer.device)
    trainer.buffer.values[:128] = torch.randn(128, device=trainer.device)
    trainer.buffer.log_probs[:128] = torch.randn(128, device=trainer.device)
    trainer.buffer.dones[:128] = torch.zeros(128, device=trainer.device)
    trainer.buffer.masks[:128] = torch.ones(128, device=trainer.device)
    trainer.buffer.ptr = 128

    stats = trainer.update(last_obs=obs[-1])
    print(f"PPO update stats: {stats}")

    # Test save/load
    trainer.save("/tmp/test_ppo.pt")
    trainer.load("/tmp/test_ppo.pt")
    print("Save/Load: OK")
    print("Phase 3 model verification passed.")
