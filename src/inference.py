"""Real-time elevator scheduling inference using trained LSTM+PPO model."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.utils import load_config, PROJ_ROOT
from src.models.lstm_ppo import PPOTrainer
from src.env.elevator_env import MAX_FLOOR


class ElevatorScheduler:
    """Online scheduler that uses the trained PPO policy for elevator dispatching."""

    def __init__(self, checkpoint_path: str | None = None, config_path: str | None = None):
        cfg = load_config(config_path)

        self.num_elevators = cfg.get("env", {}).get("num_elevators", 3)
        self.state_dim = cfg.get("model", {}).get("state_dim", 73)
        self.action_dim = cfg.get("model", {}).get("action_dim", self.num_elevators)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self._history: deque[np.ndarray] = deque(maxlen=32)

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
        if len(self._history) > 0:
            seq = np.stack(list(self._history))  # (T, D)
        else:
            seq = obs[np.newaxis, :]

        obs_t = torch.as_tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, D)

        with torch.no_grad():
            action, _, _, self.hidden = self.trainer.policy.get_action(
                obs_t, hidden=self.hidden, deterministic=True
            )
            return int(action.item())

    def _build_obs(self, state_dict: dict) -> np.ndarray:
        """Convert structured PLC state dict to flat observation vector."""
        elevators = state_dict.get("elevators", [])
        floor_up_calls = state_dict.get("floor_up_calls", [False] * MAX_FLOOR)
        floor_down_calls = state_dict.get("floor_down_calls", [False] * MAX_FLOOR)

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
                    floor_1hot,
                    dir_vec,
                    [float(el.get("load_ratio", 0.0))],
                    [1.0 if el.get("is_moving", False) else 0.0],
                    [1.0 if el.get("door_open", False) else 0.0],
                    [1.0 if el.get("has_car_calls", False) else 0.0],
                ])
            else:
                el_vec = np.zeros(MAX_FLOOR + 3 + 1 + 1 + 1 + 1, dtype=np.float32)
            elems.append(el_vec)

        up_vec = np.array(floor_up_calls[:MAX_FLOOR], dtype=np.float32)
        down_vec = np.array(floor_down_calls[:MAX_FLOOR], dtype=np.float32)

        n_active = int(sum(floor_up_calls) + sum(floor_down_calls))
        seq_progress = min(len(self._history) / max(self._history.maxlen, 1), 1.0)
        global_vec = np.array([
            seq_progress,
            min(n_active / 30.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate(elems + [up_vec, down_vec, global_vec]).astype(np.float32)

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
            {"floor": 1, "direction": 0, "load_ratio": 0.0, "is_moving": False, "door_open": False},
            {"floor": 5, "direction": 0, "load_ratio": 0.3, "is_moving": False, "door_open": False},
            {"floor": 8, "direction": -1, "load_ratio": 0.2, "is_moving": True, "door_open": False},
        ],
        "floor_up_calls": [True, False, False, False, False, False, False, False, False, False],
        "floor_down_calls": [False, False, False, False, False, False, True, False, False, False],
    }

    action = scheduler.query(mock_state)
    probs = scheduler.get_action_probs(mock_state)
    print(f"Recommended elevator: {action}")
    print(f"Action probabilities: {probs}")
