from __future__ import annotations

import argparse
from pathlib import Path

from .baseline_training import train_baseline


RATING_OPPONENTS = ("rating-exact", "rating-noise-5", "rating-noise-20")
ORIGINAL_OPPONENTS = (
    "rating-noise",
    "budget-proportion",
    "aggressive-high-value",
    "deal-probability",
)
ORIGINAL_WEIGHTS = {
    "rating-noise": 0.20,
    "budget-proportion": 0.25,
    "aggressive-high-value": 0.25,
    "deal-probability": 0.30,
}


def train_deal_value(
    *,
    calibration_timesteps: int = 100_000,
    context_timesteps: int = 200_000,
    evaluation_interval: int = 25_000,
    evaluation_episodes: int = 100,
    environments: int = 8,
    rollout_steps: int = 1_024,
    batch_size: int = 512,
    seed: int = 101,
    history_path: str | Path = "data/training_history.sqlite3",
    model_directory: str | Path = "models",
) -> tuple[str, Path, str, Path]:
    """Train a value-calibrated policy, then expose it to budget context."""

    calibration_run, calibration_model = train_baseline(
        total_timesteps=calibration_timesteps,
        evaluation_interval=evaluation_interval,
        evaluation_episodes=evaluation_episodes,
        environments=environments,
        seed=seed,
        history_path=history_path,
        model_directory=model_directory,
        opponent_names=RATING_OPPONENTS,
        opponent_weights={name: 1 / 3 for name in RATING_OPPONENTS},
        rollout_steps=rollout_steps,
        batch_size=batch_size,
        learner_family="deal-value",
        training_phase="value-calibration",
        observation_mode="value-only",
        reward_mode="value-calibration",
    )
    context_run, context_model = train_baseline(
        total_timesteps=context_timesteps,
        evaluation_interval=evaluation_interval,
        evaluation_episodes=evaluation_episodes,
        environments=environments,
        seed=seed + 1,
        history_path=history_path,
        model_directory=model_directory,
        opponent_names=ORIGINAL_OPPONENTS,
        opponent_weights=ORIGINAL_WEIGHTS,
        start_model=calibration_model,
        rollout_steps=rollout_steps,
        batch_size=batch_size,
        learner_family="deal-value",
        training_phase="budget-context",
        observation_mode="full",
    )
    return calibration_run, calibration_model, context_run, context_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate rating-to-price behavior, then train with full auction context."
        )
    )
    parser.add_argument("--calibration-timesteps", type=int, default=100_000)
    parser.add_argument("--context-timesteps", type=int, default=200_000)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--history", default="data/training_history.sqlite3")
    parser.add_argument("--models", default="models")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    calibration_run, calibration_model, context_run, context_model = (
        train_deal_value(
            calibration_timesteps=arguments.calibration_timesteps,
            context_timesteps=arguments.context_timesteps,
            evaluation_interval=arguments.eval_interval,
            evaluation_episodes=arguments.eval_episodes,
            environments=arguments.environments,
            rollout_steps=arguments.rollout_steps,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
            history_path=arguments.history,
            model_directory=arguments.models,
        )
    )
    print(f"Calibration run {calibration_run}: {calibration_model}")
    print(f"Context run {context_run}: {context_model}")


if __name__ == "__main__":
    main()
