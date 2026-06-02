"""Multi-env parallel runner for batched GPU inference and sequential env stepping."""

from __future__ import annotations

import copy

import numpy as np
import torch


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

        # Cached active tracking
        self._active_count = 0
        self._active: list[int] = []

        self._obs_gpu = torch.empty(num_envs, self.state_dim, dtype=torch.float32, device=device)
        self._use_pinned = device.type == 'cuda'
        if self._use_pinned:
            self._obs_pinned = torch.empty(num_envs, self.state_dim, dtype=torch.float32,
                                           pin_memory=True)
        else:
            self._obs_pinned = None

    def reset_env(self, i: int, events_trimmed, policy):
        if self.done[i]:
            self.done[i] = False
            self._active_count += 1
            self._active.append(i)
        obs, _ = self.envs[i].reset(options={"events": events_trimmed})
        self.obs[i] = obs
        self.hidden[i] = policy.get_initial_hidden(1, self.device)

    def step_all(self, policy, buffer=None, deterministic=False) -> tuple[float, int, int]:
        """Step all active envs with batched GPU inference.

        If buffer is provided, writes transitions to it (training mode).
        If deterministic=True, uses deterministic policy actions (validation mode).
        """
        active = self._active
        if not active:
            return 0.0, 0, 0

        # Skip envs whose buffer partition is full — mark them done so the
        # outer loop can re-feed them with a fresh episode.  Only check when
        # the global buffer is near capacity (avoids false positives after clear).
        if buffer is not None and buffer.is_ready(int(buffer.max_steps * 0.95)):
            still_active = []
            for i in active:
                if buffer.head[i] < (i + 1) * buffer.per_env:
                    still_active.append(i)
                else:
                    self.done[i] = True
            active = still_active
            if not active:
                self._active = []
                self._active_count = 0
                return 0.0, 0, 0

        n_active = len(active)

        obs_stack = np.stack([self.obs[i] for i in active])
        if self._use_pinned:
            self._obs_pinned[:n_active].copy_(torch.from_numpy(obs_stack))
            self._obs_gpu[:n_active].copy_(self._obs_pinned[:n_active], non_blocking=True)
        else:
            self._obs_gpu[:n_active].copy_(torch.from_numpy(obs_stack))

        batched_hidden = self._batch_hidden(active)
        obs_seq = self._obs_gpu[:n_active].unsqueeze(1)

        with torch.inference_mode():
            if buffer is None:
                actions, _, _, new_hidden = policy.get_action(
                    obs_seq, hidden=batched_hidden, deterministic=deterministic)
            else:
                actions, log_probs, values, new_hidden = policy.get_action(
                    obs_seq, hidden=batched_hidden, deterministic=deterministic)

        self._unbatch_hidden(active, new_hidden)
        action_ints: list[int] = actions.squeeze(-1).tolist()

        total_reward = 0.0
        total_steps = 0
        n_done = 0

        envs = self.envs
        _obs = self.obs
        _done_flags = self.done
        _obs_gpu = self._obs_gpu

        for j, i in enumerate(active):
            next_obs, reward, done, _, _ = envs[i].step(action_ints[j])

            if buffer is not None:
                buffer.add_fast(i, _obs_gpu[j], action_ints[j], reward,
                                values[j], log_probs[j], done)

            _obs[i] = next_obs
            total_reward += reward
            total_steps += 1

            if done:
                _done_flags[i] = True
                n_done += 1

        # Rebuild active list for next call
        self._active = [i for i in active if not _done_flags[i]]
        self._active_count = len(self._active)

        return total_reward, total_steps, n_done

    def _batch_hidden(self, active: list[int]):
        # self.hidden[i] is ((actor_h, actor_c), (critic_h, critic_c))
        actor_h = torch.cat([self.hidden[i][0][0] for i in active], dim=1)
        actor_c = torch.cat([self.hidden[i][0][1] for i in active], dim=1)
        critic_h = torch.cat([self.hidden[i][1][0] for i in active], dim=1)
        critic_c = torch.cat([self.hidden[i][1][1] for i in active], dim=1)
        return ((actor_h, actor_c), (critic_h, critic_c))

    def _unbatch_hidden(self, active: list[int], new_hidden):
        (new_ah, new_ac), (new_ch, new_cc) = new_hidden
        for j, i in enumerate(active):
            self.hidden[i] = (
                (new_ah[:, j:j + 1, :], new_ac[:, j:j + 1, :]),
                (new_ch[:, j:j + 1, :], new_cc[:, j:j + 1, :]),
            )

    @property
    def all_done(self) -> bool:
        return self._active_count == 0

    def get_last_obs_per_env(self) -> list:
        return [self.obs[i] if not self.done[i] else None for i in range(self.num_envs)]

    def close(self):
        pass
