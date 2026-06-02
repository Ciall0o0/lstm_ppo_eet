"""Shared utilities for the elevator scheduling project."""

from __future__ import annotations

import copy
import json
import random
import yaml
from pathlib import Path

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent

REWARD_KEYS = (
    "passenger_delivered", "wait_time_per_sec", "empty_distance_per_floor",
    "energy_per_start_stop", "idle_penalty_per_sec",
    "assignment_proximity", "assignment_direction_align", "assignment_load_balance",
    "assignment_estimated_wait",
    "normalize", "clip_range",
)


def load_config(path: str | None = None) -> dict:
    """Load YAML config, defaulting to config/config.yaml."""
    path = path or str(PROJ_ROOT / "config" / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(requested: str = "auto") -> str:
    """Resolve a device string: 'auto' → CUDA if available, else CPU."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def set_seed(seed: int):
    """Set random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


PPO_OPTUNA_KEYS = {
    "learning_rate", "entropy_coef_start", "entropy_coef_end", "entropy_floor",
    "max_grad_norm", "clip_epsilon", "value_loss_coef",
    "batch_size", "ppo_epochs", "weight_decay", "seq_len",
    "burn_in_steps", "kl_target",
}
MODEL_OPTUNA_KEYS = {
    "lstm_hidden", "lstm_layers", "lstm_dropout",
    "actor_hidden", "critic_hidden", "activation",
}
REWARD_OPTUNA_KEYS = {
    "assignment_proximity", "assignment_direction_align",
    "assignment_load_balance", "assignment_estimated_wait",
}


def load_optuna_params(cfg: dict, path: str | None = None) -> dict:
    """Load Optuna best params and merge into a config dict.

    Returns a new config dict (does not mutate the input).
    Falls back to original config if the file doesn't exist.
    """
    path = path or str(PROJ_ROOT / "checkpoints" / "optuna_best_params.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return cfg

    params = data.get("best_params", {})
    if not params:
        return cfg

    cfg = copy.deepcopy(cfg)

    # Merge PPO params
    ppo = cfg.setdefault("ppo", {})
    for k in PPO_OPTUNA_KEYS:
        if k in params:
            ppo[k] = params[k]

    # Merge model params
    model = cfg.setdefault("model", {})
    for k in MODEL_OPTUNA_KEYS:
        if k in params:
            model[k] = params[k]

    # Merge reward params
    reward = cfg.setdefault("reward", {})
    for k in REWARD_KEYS:
        if k in params:
            reward[k] = params[k]

    return cfg


def merge_reward_config(env_cfg: dict, cfg: dict) -> dict:
    """Merge reward keys from cfg['reward'] into a copy of env_cfg."""
    env_cfg = dict(env_cfg)
    reward_cfg = cfg.get("reward", {})
    for k in REWARD_KEYS:
        if k in reward_cfg:
            env_cfg[k] = reward_cfg[k]
    return env_cfg
