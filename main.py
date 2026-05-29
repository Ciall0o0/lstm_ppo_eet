"""Pipeline orchestrator: Optuna search -> Train -> Evaluate."""

import argparse

from src.utils import load_config, load_optuna_params


def main():
    parser = argparse.ArgumentParser(
        description="Elevator scheduling pipeline: Optuna -> Train -> Evaluate",
    )
    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna hyperparameter search")
    parser.add_argument("--optuna-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--train", dest="train", action="store_true", default=None,
                        help="Run training")
    parser.add_argument("--no-train", dest="train", action="store_false",
                        help="Skip training")
    parser.add_argument("--eval", dest="eval", action="store_true", default=None,
                        help="Run evaluation after training")
    parser.add_argument("--no-eval", dest="eval", action="store_false",
                        help="Skip evaluation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--no-optuna-params", action="store_true",
                        help="Ignore Optuna best params, train with default config")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for evaluation (default: best_model.pt)")
    args = parser.parse_args()

    # Default behavior: train + eval (skip optuna search)
    if args.train is None and args.eval is None and not args.optuna:
        args.train = True
        args.eval = True

    # Build ordered step list
    steps = []
    if args.optuna:
        steps.append("optuna")
    if args.train:
        steps.append("train")
    if args.eval:
        steps.append("eval")
    total = len(steps)

    step_labels = {"optuna": "Optuna hyperparameter search",
                   "train": "Training",
                   "eval": "Evaluation"}

    for i, step in enumerate(steps, 1):
        print("=" * 60)
        print(f"Step {i}/{total}: {step_labels[step]}")
        print("=" * 60)

        if step == "optuna":
            from src.optuna_search import main as optuna_main
            optuna_main(n_trials=args.optuna_trials)

        elif step == "train":
            cfg = load_config()

            # Apply Optuna best params unless explicitly disabled
            if not args.no_optuna_params:
                cfg_before = cfg
                cfg = load_optuna_params(cfg)
                if cfg is not cfg_before:
                    print("Applied Optuna best hyperparameters")
                    ppo = cfg.get("ppo", {})
                    model = cfg.get("model", {})
                    print(f"  LR: {ppo.get('learning_rate', '?')}")
                    print(f"  LSTM hidden={model.get('lstm_hidden', '?')}, "
                          f"layers={model.get('lstm_layers', '?')}")
                    print(f"  Batch: {ppo.get('batch_size', '?')}, "
                          f"PPO epochs: {ppo.get('ppo_epochs', '?')}")

            # Override epochs if specified
            if args.epochs is not None:
                cfg.setdefault("training", {})["total_epochs"] = args.epochs
                print(f"  Epochs: {args.epochs} (override)")

            from src.train import main as train_main
            train_main(cfg=cfg)

        elif step == "eval":
            eval_args = {}
            if args.checkpoint:
                eval_args["checkpoint"] = args.checkpoint

            from src.evaluate import main as eval_main
            eval_main(args=eval_args if eval_args else None)

        print()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
