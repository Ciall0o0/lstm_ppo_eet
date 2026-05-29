"""Real-time elevator scheduling inference using trained LSTM+PPO model."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.utils import load_config, PROJ_ROOT, get_device
from src.models.lstm_ppo import PPOTrainer
from src.env.elevator_env import ElevatorEnv, MAX_FLOOR


class ElevatorScheduler:
    """Online scheduler that uses the trained PPO policy for elevator dispatching."""

    def __init__(self, checkpoint_path: str | None = None, config_path: str | None = None):
        cfg = load_config(config_path)

        env_cfg = cfg.get("env", {})
        self.num_elevators = env_cfg.get("num_elevators", 3)
        self.device = torch.device(get_device())

        # Derive dimensions from config without constructing a full environment
        self.state_dim = ElevatorEnv._compute_state_dim(self.num_elevators)
        self.action_dim = self.num_elevators

        self._seq_len = cfg.get("ppo", {}).get("seq_len", 32)
        self._max_calls_norm = float(self.num_elevators * MAX_FLOOR)

        self.trainer = PPOTrainer.from_config(self.state_dim, self.action_dim, cfg, str(self.device))

        if checkpoint_path:
            self.load(checkpoint_path)
        else:
            default_ckpt = PROJ_ROOT / "checkpoints" / "best_model.pt"
            if default_ckpt.exists():
                self.load(str(default_ckpt))
                print(f"Loaded default checkpoint: {default_ckpt}")

        self.trainer.policy.eval()
        self.hidden: tuple | None = None
        self._history: deque[np.ndarray] = deque(maxlen=self._seq_len)

    def load(self, path: str):
        self.trainer.load(path)
        print(f"Model loaded from {path}")

    def reset(self):
        """Reset LSTM hidden state and history buffer."""
        self.hidden = self.trainer.policy.get_initial_hidden(1, self.device)
        self._history.clear()

    def query(self, state_dict: dict) -> int:
        """Query the model for the best elevator to dispatch.

        Args:
            state_dict: PLC state dictionary with keys:
                elevators: list of dicts with floor, direction, load_ratio, is_moving, door_open
                floor_up_calls: list of 10 bools
                floor_down_calls: list of 10 bools

        Returns:
            elevator_id: 0, 1, or 2
        """
        obs = self._build_obs(state_dict)
        self._history.append(obs)

        # Build sequence from history
        seq = np.stack(list(self._history))  # (T, D)

        obs_t = torch.as_tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, D)

        with torch.no_grad():
            action, _, _, self.hidden = self.trainer.policy.get_action(
                obs_t, hidden=self.hidden, deterministic=True
            )
            return int(action.item())

    def _build_obs(self, state_dict: dict) -> np.ndarray:
        """Convert structured PLC state dict to flat observation vector.

        Produces 104-dim vector matching ElevatorEnv._get_obs():
          Per elevator (20 dims): floor_1hot(10) + dir_1hot(3) + load_ratio(1)
                                  + is_moving(1) + door_open(1) + has_car_calls(1)
                                  + passenger_load_ratio(1) + target_floor(1) + dist_to_oldest(1)
          Global (44 dims): up_calls(10) + down_calls(10) + up_dest(10) + down_dest(10)
                            + elapsed(1) + n_pending(1) + time_delta(1) + floor_delta(1)
        Note: passenger_load_ratio, target_floor, dist_to_oldest, and call destinations
              are unavailable from PLC — padded with zeros.
        """
        elevators = state_dict.get("elevators", [])
        floor_up_calls = state_dict.get("floor_up_calls", [False] * MAX_FLOOR)
        floor_down_calls = state_dict.get("floor_down_calls", [False] * MAX_FLOOR)

        # Per-elevator features (20 dims each)
        elems = []
        for i in range(self.num_elevators):
            if i < len(elevators):
                el = elevators[i]
                floor_1hot = np.zeros(MAX_FLOOR, dtype=np.float32)
                f = int(el.get("floor", 1))
                if 1 <= f <= MAX_FLOOR:
                    floor_1hot[f - 1] = 1.0

                dir_vec = np.zeros(3, dtype=np.float32)
                d = el.get("direction", 0)  # -1=down, 0=idle, 1=up
                dir_vec[d + 1] = 1.0

                el_vec = np.concatenate([
                    floor_1hot,                    # 10
                    dir_vec,                        # 3
                    [float(el.get("load_ratio", 0.0))],  # 1
                    [1.0 if el.get("is_moving", False) else 0.0],   # 1
                    [1.0 if el.get("door_open", False) else 0.0],   # 1
                    [1.0 if el.get("has_car_calls", False) else 0.0],  # 1
                    [0.0],  # passenger_load_ratio (unavailable from PLC)
                    [0.0],  # target_floor / MAX_FLOOR (unavailable from PLC)
                    [0.0],  # dist_to_oldest (unavailable from PLC)
                ])
            else:
                el_vec = np.zeros(20, dtype=np.float32)
            elems.append(el_vec)

        # Up/down call indicators (20 dims)
        up_vec = np.array(floor_up_calls[:MAX_FLOOR], dtype=np.float32)
        down_vec = np.array(floor_down_calls[:MAX_FLOOR], dtype=np.float32)

        # Call destinations (20 dims) - unavailable from PLC, pad zeros
        up_dest_vec = np.zeros(MAX_FLOOR, dtype=np.float32)
        down_dest_vec = np.zeros(MAX_FLOOR, dtype=np.float32)

        # Global features (4 dims)
        n_active = int(sum(floor_up_calls) + sum(floor_down_calls))
        seq_progress = min(len(self._history) / max(self._history.maxlen, 1), 1.0)
        global_vec = np.array([
            seq_progress,                           # elapsed / max_time
            min(n_active / self._max_calls_norm, 1.0),  # n_pending
            0.0,  # time_delta (unavailable from PLC)
            0.0,  # floor_delta (unavailable from PLC)
        ], dtype=np.float32)

        return np.concatenate(
            elems + [up_vec, down_vec, up_dest_vec, down_dest_vec, global_vec]
        )

    def get_action_probs(self, state_dict: dict) -> dict[int, float]:
        """Return action probabilities for all elevators (useful for diagnostics)."""
        obs = self._build_obs(state_dict)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits, _, self.hidden = self.trainer.policy.forward(obs_t, hidden=self.hidden)
            probs = torch.softmax(logits.squeeze(), dim=-1).cpu().numpy()

        return {i: float(p) for i, p in enumerate(probs)}


if __name__ == "__main__":
    # Quick test with simulated state
    scheduler = ElevatorScheduler()
    scheduler.reset()

    mock_state = {
        "elevators": [
            {"floor": 1, "direction": 0, "load_ratio": 0.0, "is_moving": False, "door_open": False, "has_car_calls": False},
            {"floor": 5, "direction": 0, "load_ratio": 0.3, "is_moving": False, "door_open": False, "has_car_calls": True},
            {"floor": 8, "direction": -1, "load_ratio": 0.2, "is_moving": True, "door_open": False, "has_car_calls": False},
        ],
        "floor_up_calls": [True, False, False, False, False, False, False, False, False, False],
        "floor_down_calls": [False, False, False, False, False, False, True, False, False, False],
    }

    action = scheduler.query(mock_state)
    probs = scheduler.get_action_probs(mock_state)
    print(f"Recommended elevator: {action}")
    print(f"Action probabilities: {probs}")
