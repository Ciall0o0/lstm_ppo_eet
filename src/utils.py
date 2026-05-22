"""Shared utilities for the elevator scheduling project."""

import yaml
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    """Load YAML config, defaulting to config/config.yaml."""
    path = path or str(PROJ_ROOT / "config" / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
