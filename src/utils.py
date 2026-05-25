"""Shared utilities for the elevator scheduling project."""

from __future__ import annotations

import yaml
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent

REWARD_KEYS = (
    "passenger_delivered", "wait_time_per_sec", "empty_distance_per_floor",
    "energy_per_start_stop", "idle_penalty_per_sec", "assignment_dist_per_floor",
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


def merge_reward_config(env_cfg: dict, cfg: dict) -> dict:
    """Copy reward keys from cfg['reward'] into env_cfg (mutates env_cfg)."""
    reward_cfg = cfg.get("reward", {})
    for k in REWARD_KEYS:
        if k in reward_cfg:
            env_cfg[k] = reward_cfg[k]
    return env_cfg
