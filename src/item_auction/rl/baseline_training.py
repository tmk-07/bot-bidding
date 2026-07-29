from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .env import FixedOpponentEnv
from .history import TrainingHistory
from .opponents import OpponentEntry, OpponentPool, OpponentPolicy


DEFAULT_WEIGHTS = {
    "rating-noise": 0.20,
    "budget-proportion": 0.25,
    "aggressive-high-value": 0.25,
    "deal-probability": 0.30,
}


class WeightedOpponentFactory:
    """Stateful callable that selects a fresh policy for each episode."""

    def __init__(
        self,
        entries: Sequence[OpponentEntry],
        *,
        weights: dict[str, float],
        seed: int,
    ) -> None:
        self.entries = list(entries)
        self.weights = [weights.get(entry.name, 1.0) for entry in self.entries]
        self.rng = random.Random(seed)

    def __call__(self) -> OpponentPolicy:
        entry = self.rng.choices(self.entries, weights=self.weights, k=1)[0]
        return entry.factory()


def baseline_entries() -> list[OpponentEntry]:
    return OpponentPool.training_baselines().entries


def evaluate_checkpoint(
    model: Any,
    *,
    history: TrainingHistory,
    run_id: str,
    checkpoint_steps: int,
    episodes_per_opponent: int,
    seed: int,
    entries: Sequence[OpponentEntry] | None = None,
) -> None:
    """Play deterministic held-out games and persist every result and selection."""

    evaluation_entries = list(entries or baseline_entries())
    for opponent_index, entry in enumerate(evaluation_entries):
        for episode in range(episodes_per_opponent):
            game_seed = (
                seed
                + opponent_index * 1_000_003
                + episode
            )
            env = FixedOpponentEnv(entry.factory, randomize_seat=True)
            observation, _ = env.reset(seed=game_seed)
            terminated = False
            while not terminated:
                action, _ = model.predict(
                    observation,
                    action_masks=observation["action_mask"],
                    deterministic=True,
                )
                observation, _, terminated, truncated, _ = env.step(int(action))
                if truncated:
                    raise RuntimeError("Evaluation game was unexpectedly truncated")
            history.record_game(
                run_id=run_id,
                checkpoint_steps=checkpoint_steps,
                episode=episode,
                seed=game_seed,
                opponent_name=entry.name,
                learner_agent=env.learner_agent,
                engine=env.aec.engine,
            )
            env.close()


def train_baseline(
    *,
    total_timesteps: int = 100_000,
    evaluation_interval: int = 25_000,
    evaluation_episodes: int = 100,
    environments: int = 8,
    seed: int = 7,
    history_path: str | Path = "data/training_history.sqlite3",
    model_directory: str | Path = "models",
) -> tuple[str, Path]:
    """Train MaskablePPO against the four structured baseline strategies."""

    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    if evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive")
    if evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be positive")
    if environments < 1:
        raise ValueError("environments must be positive")

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install them with "
            "`pip install -e '.[train]'`."
        ) from exc

    entries = baseline_entries()
    config = {
        "opponents": [entry.name for entry in entries],
        "opponent_weights": DEFAULT_WEIGHTS,
        "evaluation_interval": evaluation_interval,
        "evaluation_episodes_per_opponent": evaluation_episodes,
        "environments": environments,
        "policy": "MultiInputPolicy",
        "learning_rate": 3e-4,
        "n_steps": 256,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "batch_size": 256,
        "network": [128, 128],
    }
    history = TrainingHistory(history_path)
    run_id = history.create_run(
        algorithm="MaskablePPO",
        total_timesteps=total_timesteps,
        seed=seed,
        config=config,
    )

    def make_environment(index: int) -> Callable[[], FixedOpponentEnv]:
        def factory() -> FixedOpponentEnv:
            opponent_factory = WeightedOpponentFactory(
                entries,
                weights=DEFAULT_WEIGHTS,
                seed=seed + index * 97_409,
            )
            return FixedOpponentEnv(opponent_factory, randomize_seat=True)

        return factory

    vector_env = VecMonitor(
        DummyVecEnv([make_environment(index) for index in range(environments)])
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        vector_env,
        learning_rate=3e-4,
        n_steps=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        batch_size=256,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        verbose=1,
    )

    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    final_path = model_directory / f"baseline-{run_id}"
    completed_steps = 0
    try:
        evaluate_checkpoint(
            model,
            history=history,
            run_id=run_id,
            checkpoint_steps=0,
            episodes_per_opponent=evaluation_episodes,
            seed=seed + 50_000_000,
            entries=entries,
        )
        while model.num_timesteps < total_timesteps:
            chunk = min(
                evaluation_interval,
                total_timesteps - model.num_timesteps,
            )
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            completed_steps = model.num_timesteps
            checkpoint_path = model_directory / (
                f"baseline-{run_id}-{completed_steps}"
            )
            model.save(checkpoint_path)
            evaluate_checkpoint(
                model,
                history=history,
                run_id=run_id,
                checkpoint_steps=completed_steps,
                episodes_per_opponent=evaluation_episodes,
                seed=seed + 50_000_000,
                entries=entries,
            )
        model.save(final_path)
        history.finish_run(
            run_id,
            status="completed",
            model_path=str(final_path.with_suffix(".zip")),
        )
    except Exception:
        history.finish_run(run_id, status="failed")
        raise
    finally:
        vector_env.close()
    return run_id, final_path.with_suffix(".zip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO against the structured baseline bots."
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--history", default="data/training_history.sqlite3"
    )
    parser.add_argument("--models", default="models")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    run_id, model_path = train_baseline(
        total_timesteps=arguments.timesteps,
        evaluation_interval=arguments.eval_interval,
        evaluation_episodes=arguments.eval_episodes,
        environments=arguments.environments,
        seed=arguments.seed,
        history_path=arguments.history,
        model_directory=arguments.models,
    )
    print(f"Training run {run_id} completed.")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
