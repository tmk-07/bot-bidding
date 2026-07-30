from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .actions import BidAction, action_to_bid
from .env import FixedOpponentEnv
from .history import TrainingHistory
from .opponents import (
    FrozenCheckpointPolicy,
    OpponentEntry,
    OpponentPool,
    OpponentPolicy,
    RatingNoisePolicy,
)


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
        exposure_counter: Counter[str] | None = None,
    ) -> None:
        self.entries = list(entries)
        self.weights = [weights.get(entry.name, 1.0) for entry in self.entries]
        self.rng = random.Random(seed)
        self.exposure_counter = exposure_counter

    def __call__(self) -> OpponentPolicy:
        entry = self.rng.choices(self.entries, weights=self.weights, k=1)[0]
        if self.exposure_counter is not None:
            self.exposure_counter[entry.name] += 1
        return entry.factory()


def baseline_entries() -> list[OpponentEntry]:
    return OpponentPool.training_baselines().entries


def available_entries() -> list[OpponentEntry]:
    """All structured policies available to a configurable curriculum."""

    entries = baseline_entries()
    entries.extend(
        [
            OpponentEntry(
                "rating-exact",
                lambda: RatingNoisePolicy(0.0, name="rating-exact"),
            ),
            OpponentEntry(
                "rating-noise-5",
                lambda: RatingNoisePolicy(0.05, name="rating-noise-5"),
            ),
            OpponentEntry(
                "rating-noise-20",
                lambda: RatingNoisePolicy(0.20, name="rating-noise-20"),
            ),
        ]
    )
    return entries


def selected_entries(
    *,
    opponent_names: Sequence[str] | None = None,
    frozen_models: Sequence[str | Path] | None = None,
    frozen_names: Sequence[str] | None = None,
) -> list[OpponentEntry]:
    available = {entry.name: entry for entry in available_entries()}
    names = list(opponent_names or available)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown baseline opponents: {', '.join(unknown)}")
    entries = [available[name] for name in names]
    checkpoints = list(frozen_models or [])
    names_for_frozen = list(frozen_names or [])
    if names_for_frozen and len(names_for_frozen) != len(checkpoints):
        raise ValueError(
            "frozen_names must contain one name for every frozen model"
        )
    for index, checkpoint in enumerate(checkpoints):
        path = Path(checkpoint)
        checkpoint_id = path.stem.removeprefix("baseline-")
        frozen_name = (
            names_for_frozen[index]
            if names_for_frozen
            else f"frozen-{checkpoint_id}"
        )
        policy = FrozenCheckpointPolicy(
            path,
            name=frozen_name,
        )
        entries.append(
            OpponentEntry(policy.name, lambda policy=policy: policy)
        )
    if not entries:
        raise ValueError("At least one opponent is required")
    return entries


def evaluate_checkpoint(
    model: Any,
    *,
    history: TrainingHistory,
    run_id: str,
    checkpoint_steps: int,
    episodes_per_opponent: int,
    seed: int,
    entries: Sequence[OpponentEntry] | None = None,
    observation_mode: str = "full",
    reward_mode: str = "standard",
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
            env = FixedOpponentEnv(
                entry.factory,
                randomize_seat=True,
                observation_mode=observation_mode,
                reward_mode=reward_mode,
            )
            observation, _ = env.reset(seed=game_seed)
            terminated = False
            learner_actions: list[dict[str, Any]] = []
            while not terminated:
                action, _ = model.predict(
                    observation,
                    action_masks=observation["action_mask"],
                    deterministic=True,
                )
                selected_action = int(action)
                learner_actions.append(
                    {
                        "auction_number": env.aec.engine.item_index + 1,
                        "rating": env.aec.current_rating,
                        "current_bid": env.aec.current_bid,
                        "own_budget": env.aec.budgets[env.learner_agent],
                        "opponent_budget": env.aec.budgets[env.opponent_agent],
                        "action_index": selected_action,
                        "action_name": BidAction(selected_action).name,
                        "target_bid": action_to_bid(
                            selected_action,
                            current_bid=env.aec.current_bid,
                            own_budget=env.aec.budgets[env.learner_agent],
                            opponent_budget=env.aec.budgets[env.opponent_agent],
                        ),
                    }
                )
                observation, _, terminated, truncated, _ = env.step(
                    selected_action
                )
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
                learner_actions=learner_actions,
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
    opponent_names: Sequence[str] | None = None,
    frozen_models: Sequence[str | Path] | None = None,
    frozen_names: Sequence[str] | None = None,
    start_model: str | Path | None = None,
    opponent_weights: dict[str, float] | None = None,
    rollout_steps: int = 1_024,
    batch_size: int = 512,
    learner_family: str = "iterated",
    training_phase: str = "context",
    observation_mode: str = "full",
    reward_mode: str = "standard",
) -> tuple[str, Path]:
    """Train MaskablePPO against selected baselines and frozen checkpoints."""

    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    if evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive")
    if evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be positive")
    if environments < 1:
        raise ValueError("environments must be positive")
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    rollout_size = rollout_steps * environments
    if rollout_size % batch_size:
        raise ValueError(
            "batch_size must evenly divide rollout_steps * environments"
        )

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install them with "
            "`pip install -e '.[train]'`."
        ) from exc

    entries = selected_entries(
        opponent_names=opponent_names,
        frozen_models=frozen_models,
        frozen_names=frozen_names,
    )
    if opponent_weights is None:
        resolved_weights = {
            entry.name: DEFAULT_WEIGHTS.get(entry.name, 0.30)
            for entry in entries
        }
    else:
        entry_names = {entry.name for entry in entries}
        unknown_weights = sorted(set(opponent_weights) - entry_names)
        if unknown_weights:
            raise ValueError(
                "Weights supplied for unavailable opponents: "
                + ", ".join(unknown_weights)
            )
        resolved_weights = {
            entry.name: opponent_weights.get(entry.name, 0.0)
            for entry in entries
        }
        if any(weight < 0 for weight in resolved_weights.values()):
            raise ValueError("Opponent weights must not be negative")
        if sum(resolved_weights.values()) <= 0:
            raise ValueError("At least one opponent weight must be positive")
    config = {
        "opponents": [entry.name for entry in entries],
        "opponent_weights": resolved_weights,
        "frozen_models": [str(Path(path)) for path in frozen_models or []],
        "frozen_names": list(frozen_names or []),
        "start_model": str(Path(start_model)) if start_model else None,
        "learner_family": learner_family,
        "training_phase": training_phase,
        "observation_mode": observation_mode,
        "reward_mode": reward_mode,
        "evaluation_interval": evaluation_interval,
        "evaluation_episodes_per_opponent": evaluation_episodes,
        "environments": environments,
        "policy": "MultiInputPolicy",
        "learning_rate": 3e-4,
        "n_steps": rollout_steps,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "batch_size": batch_size,
        "network": [128, 128],
    }
    history = TrainingHistory(history_path)
    run_id = history.create_run(
        algorithm="MaskablePPO",
        total_timesteps=total_timesteps,
        seed=seed,
        config=config,
        learner_family=learner_family,
    )
    training_exposure: Counter[str] = Counter()

    def make_environment(index: int) -> Callable[[], FixedOpponentEnv]:
        def factory() -> FixedOpponentEnv:
            opponent_factory = WeightedOpponentFactory(
                entries,
                weights=resolved_weights,
                seed=seed + index * 97_409,
                exposure_counter=training_exposure,
            )
            return FixedOpponentEnv(
                opponent_factory,
                randomize_seat=True,
                observation_mode=observation_mode,
                reward_mode=reward_mode,
            )

        return factory

    vector_env = VecMonitor(
        DummyVecEnv([make_environment(index) for index in range(environments)])
    )
    if start_model:
        start_path = Path(start_model)
        if not start_path.exists():
            raise FileNotFoundError(f"Starting checkpoint not found: {start_path}")
        model = MaskablePPO.load(
            start_path,
            env=vector_env,
            custom_objects={
                "n_steps": rollout_steps,
                "batch_size": batch_size,
            },
        )
        # Checkpoint labels and requested timesteps are local to this run, while
        # policy and optimizer parameters continue from the saved checkpoint.
        model.num_timesteps = 0
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            vector_env,
            learning_rate=3e-4,
            n_steps=rollout_steps,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            batch_size=batch_size,
            policy_kwargs={"net_arch": [128, 128]},
            seed=seed,
            verbose=1,
        )

    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    safe_family = learner_family.lower().replace(" ", "-").replace("_", "-")
    final_path = model_directory / f"{safe_family}-{run_id}"
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
            observation_mode=observation_mode,
            reward_mode=reward_mode,
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
                f"{safe_family}-{run_id}-{completed_steps}"
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
                observation_mode=observation_mode,
                reward_mode=reward_mode,
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
        history.record_training_exposure(run_id, dict(training_exposure))
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
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1_024,
        help="Learner decisions collected per environment before each PPO update.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learner-family", default="iterated")
    parser.add_argument("--training-phase", default="context")
    parser.add_argument(
        "--observation-mode",
        choices=["full", "value-only"],
        default="full",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["standard", "value-calibration"],
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=[entry.name for entry in available_entries()],
        help="Structured baseline opponents to include (default: all four).",
    )
    parser.add_argument(
        "--frozen-model",
        action="append",
        default=[],
        help="Saved MaskablePPO .zip to add as an immutable opponent.",
    )
    parser.add_argument(
        "--frozen-name",
        action="append",
        default=[],
        help="Readable name paired by position with each --frozen-model.",
    )
    parser.add_argument(
        "--start-model",
        help="Saved MaskablePPO .zip whose learner weights should be continued.",
    )
    parser.add_argument(
        "--opponent-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Episode sampling weight for a selected opponent.",
    )
    parser.add_argument(
        "--history", default="data/training_history.sqlite3"
    )
    parser.add_argument("--models", default="models")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    parsed_weights: dict[str, float] | None = None
    if arguments.opponent_weight:
        parsed_weights = {}
        for specification in arguments.opponent_weight:
            try:
                name, raw_weight = specification.rsplit("=", 1)
                parsed_weights[name] = float(raw_weight)
            except (ValueError, TypeError) as exc:
                raise SystemExit(
                    f"Invalid --opponent-weight {specification!r}; "
                    "expected NAME=WEIGHT"
                ) from exc
    run_id, model_path = train_baseline(
        total_timesteps=arguments.timesteps,
        evaluation_interval=arguments.eval_interval,
        evaluation_episodes=arguments.eval_episodes,
        environments=arguments.environments,
        seed=arguments.seed,
        history_path=arguments.history,
        model_directory=arguments.models,
        opponent_names=arguments.opponents,
        frozen_models=arguments.frozen_model,
        frozen_names=arguments.frozen_name,
        start_model=arguments.start_model,
        opponent_weights=parsed_weights,
        rollout_steps=arguments.rollout_steps,
        batch_size=arguments.batch_size,
        learner_family=arguments.learner_family,
        training_phase=arguments.training_phase,
        observation_mode=arguments.observation_mode,
        reward_mode=arguments.reward_mode,
    )
    print(f"Training run {run_id} completed.")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
