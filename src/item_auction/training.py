from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .agents import LinearBot, LinearPolicy, ValueBot
from .environment import AuctionConfig
from .simulation import play_game


def evaluate_policy(
    policy: LinearPolicy,
    *,
    episodes: int,
    seed: int,
    config: AuctionConfig | None = None,
) -> float:
    total = 0.0
    for episode in range(episodes):
        learner = LinearBot("learner", policy)
        opponents = [
            ValueBot("steady", 0.90, 0.32),
            ValueBot("patient", 0.82, 0.48),
            ValueBot("aggressive", 1.15, 0.28),
        ]
        result = play_game(
            [learner, *opponents], config=config, seed=seed + episode
        )
        best = max(result.scores.values())
        learner_score = result.scores["learner"]
        win_credit = 1.0 / len(result.winners) if "learner" in result.winners else 0.0
        roster_fill = len(result.rosters["learner"]) / (config or AuctionConfig()).roster_size
        total += win_credit * 2.0 + learner_score / max(1.0, best) + roster_fill * 0.25
    return total / episodes


def train_policy(
    *,
    generations: int = 20,
    population: int = 24,
    episodes: int = 30,
    seed: int = 1,
    config: AuctionConfig | None = None,
) -> tuple[LinearPolicy, list[float]]:
    """Train with a compact cross-entropy evolutionary search.

    This is dependency-free and easily replaceable with PPO/DQN later because
    the environment exposes explicit observations and actions.
    """

    if generations < 1 or population < 4 or episodes < 1:
        raise ValueError("generations >= 1, population >= 4, and episodes >= 1")
    rng = random.Random(seed)
    means = LinearPolicy().values()
    stds = [0.50, 0.55, 0.40, 0.35, 0.30, 0.30, 0.25]
    history: list[float] = []
    best = LinearPolicy()
    best_score = float("-inf")
    elite_count = max(2, population // 5)

    for generation in range(generations):
        candidates: list[tuple[float, list[float]]] = []
        for candidate_index in range(population):
            values = [
                rng.gauss(mean, std) for mean, std in zip(means, stds, strict=True)
            ]
            values[-1] = max(0.2, values[-1])
            policy = LinearPolicy.from_values(values)
            score = evaluate_policy(
                policy,
                episodes=episodes,
                seed=seed * 1_000_003 + generation * episodes,
                config=config,
            )
            candidates.append((score, values))
            if score > best_score:
                best_score, best = score, policy
        candidates.sort(reverse=True, key=lambda pair: pair[0])
        elites = [values for _, values in candidates[:elite_count]]
        means = [sum(row[i] for row in elites) / elite_count for i in range(len(means))]
        stds = [
            max(
                0.04,
                (
                    sum((row[i] - means[i]) ** 2 for row in elites) / elite_count
                )
                ** 0.5,
            )
            for i in range(len(stds))
        ]
        history.append(candidates[0][0])
    return best, history


def save_policy(policy: LinearPolicy, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(policy), indent=2) + "\n")


def load_policy(path: str | Path) -> LinearPolicy:
    return LinearPolicy(**json.loads(Path(path).read_text()))
