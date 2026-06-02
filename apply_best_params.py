"""Apply Optuna best params to config.yaml."""
import json
import sys
from pathlib import Path

import yaml

BEST_PARAMS_PATH = Path("checkpoints/optuna_best_params.json")
CONFIG_PATH = Path("config/config.yaml")

# Mapping from Optuna param names to config.yaml paths
PARAM_MAP = {
    # PPO params
    "learning_rate": ("ppo", "learning_rate"),
    "entropy_coef_start": ("ppo", "entropy_coef_start"),
    "entropy_coef_end": ("ppo", "entropy_coef_end"),
    "entropy_floor": ("ppo", "entropy_floor"),
    "max_grad_norm": ("ppo", "max_grad_norm"),
    "clip_epsilon": ("ppo", "clip_epsilon"),
    "value_loss_coef": ("ppo", "value_loss_coef"),
    "batch_size": ("ppo", "batch_size"),
    "ppo_epochs": ("ppo", "ppo_epochs"),
    "weight_decay": ("ppo", "weight_decay"),
    "seq_len": ("ppo", "seq_len"),
    "burn_in_steps": ("ppo", "burn_in_steps"),
    "kl_target": ("ppo", "kl_target"),
    # Model params
    "lstm_hidden": ("model", "lstm_hidden"),
    "actor_hidden": ("model", "actor_hidden"),
    "critic_hidden": ("model", "critic_hidden"),
    "activation": ("model", "activation"),
    # Reward params
    "assignment_proximity": ("reward", "assignment_proximity"),
    "assignment_direction_align": ("reward", "assignment_direction_align"),
    "assignment_load_balance": ("reward", "assignment_load_balance"),
    "assignment_estimated_wait": ("reward", "assignment_estimated_wait"),
}


def main():
    if not BEST_PARAMS_PATH.exists():
        print(f"ERROR: {BEST_PARAMS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with open(BEST_PARAMS_PATH) as f:
        data = json.load(f)

    best = data["best_params"]
    print(f"Best val_reward: {data['best_value']:.6f}")
    print(f"Params: {json.dumps(best, indent=2)}")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    updated = []
    for param_name, value in best.items():
        if param_name not in PARAM_MAP:
            print(f"  [skip] {param_name} (no mapping)")
            continue
        section, key = PARAM_MAP[param_name]
        cfg.setdefault(section, {})
        old = cfg[section].get(key, "(missing)")
        cfg[section][key] = value
        updated.append(f"  {section}.{key}: {old} → {value}")

    for line in updated:
        print(line)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nUpdated {CONFIG_PATH} ({len(updated)} params)")


if __name__ == "__main__":
    main()
