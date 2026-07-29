from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .actions import BidAction


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    item_rating: float
    current_bid: int
    own_budget: int
    opponent_budget: int
    own_score: float
    opponent_score: float
    own_items: int
    opponent_items: int
    items_remaining: int
    is_opening_turn: bool


class OpponentPolicy(Protocol):
    """Interface shared by baselines and future trained checkpoints."""

    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def start_auction(self, observation: PolicyObservation) -> None: ...

    def act(
        self, observation: PolicyObservation, action_mask: np.ndarray
    ) -> int: ...


class _MaximumPolicy:
    name = "maximum-policy"

    def __init__(self) -> None:
        self._rng = random.Random()
        self._maximum = 0

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._maximum = 0

    def _choose_maximum(self, observation: PolicyObservation) -> int:
        raise NotImplementedError

    def start_auction(self, observation: PolicyObservation) -> None:
        self._maximum = max(
            0,
            min(observation.own_budget, round(self._choose_maximum(observation))),
        )

    def act(
        self, observation: PolicyObservation, action_mask: np.ndarray
    ) -> int:
        if (
            observation.current_bid + 1 <= self._maximum
            and action_mask[BidAction.MIN_RAISE]
        ):
            return int(BidAction.MIN_RAISE)
        return int(BidAction.PASS)


class RatingNoisePolicy(_MaximumPolicy):
    """Maximum is item rating plus or minus 10%."""

    name = "rating-noise"

    def _choose_maximum(self, observation: PolicyObservation) -> int:
        noise = self._rng.uniform(-0.10, 0.10)
        return round(observation.item_rating * (1.0 + noise))


class BudgetProportionPolicy(_MaximumPolicy):
    """Rating maps directly to a percentage of the current budget, ±10%."""

    name = "budget-proportion"

    def _choose_maximum(self, observation: PolicyObservation) -> int:
        base = observation.own_budget * observation.item_rating / 100.0
        return round(base * self._rng.uniform(0.90, 1.10))


class AggressiveHighValuePolicy(_MaximumPolicy):
    """Spends heavily above rating 70 and almost nothing below it."""

    name = "aggressive-high-value"

    def _choose_maximum(self, observation: PolicyObservation) -> int:
        if observation.item_rating > 70:
            quality = (observation.item_rating - 70) / 30
            fraction = 0.55 + 0.40 * quality
            return round(
                observation.own_budget * fraction * self._rng.uniform(0.90, 1.10)
            )
        return round(observation.own_budget * self._rng.uniform(0.00, 0.05))


class RandomPassPolicy:
    """Makes an independent 50/50 pass-or-minimum-raise decision each turn."""

    name = "random-pass"

    def __init__(self, raise_probability: float = 0.50) -> None:
        self.raise_probability = raise_probability
        self._rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def start_auction(self, observation: PolicyObservation) -> None:
        pass

    def act(
        self, observation: PolicyObservation, action_mask: np.ndarray
    ) -> int:
        can_raise = bool(action_mask[BidAction.MIN_RAISE])
        if can_raise and self._rng.random() < self.raise_probability:
            return int(BidAction.MIN_RAISE)
        return int(BidAction.PASS)


class DealProbabilityPolicy(RandomPassPolicy):
    """Raises more often when rating is high relative to the visible price."""

    name = "deal-probability"

    def act(
        self, observation: PolicyObservation, action_mask: np.ndarray
    ) -> int:
        if not action_mask[BidAction.MIN_RAISE]:
            return int(BidAction.PASS)
        deal_margin = observation.item_rating - (observation.current_bid + 1)
        # Smoothly moves from near-certain pass on a bad deal to near-certain
        # raise on a strong deal. A $15 positive margin is roughly 82% raise.
        probability = 1.0 / (1.0 + math.exp(-deal_margin / 10.0))
        probability = min(0.97, max(0.03, probability))
        if self._rng.random() < probability:
            return int(BidAction.MIN_RAISE)
        return int(BidAction.PASS)


class CallablePolicy:
    """Adapter for trained models, checkpoints, or any prediction callable.

    The callable receives ``(observation, action_mask)`` and returns an action
    index. This keeps the environment independent of PPO, PyTorch, or any other
    training framework.
    """

    def __init__(
        self,
        name: str,
        predictor: Callable[[PolicyObservation, np.ndarray], int],
        reset_callback: Callable[[int | None], None] | None = None,
    ) -> None:
        self.name = name
        self._predictor = predictor
        self._reset_callback = reset_callback

    def reset(self, seed: int | None = None) -> None:
        if self._reset_callback:
            self._reset_callback(seed)

    def start_auction(self, observation: PolicyObservation) -> None:
        pass

    def act(
        self, observation: PolicyObservation, action_mask: np.ndarray
    ) -> int:
        action = int(self._predictor(observation, action_mask))
        if not 0 <= action < len(action_mask) or not action_mask[action]:
            raise ValueError(f"{self.name} returned illegal action {action}")
        return action


@dataclass(frozen=True, slots=True)
class OpponentEntry:
    name: str
    factory: Callable[[], OpponentPolicy]


class OpponentPool:
    """Registry that mixes baselines and future frozen trained policies."""

    def __init__(self, entries: list[OpponentEntry] | None = None) -> None:
        self.entries = list(entries or [])

    def add(self, name: str, factory: Callable[[], OpponentPolicy]) -> None:
        self.entries.append(OpponentEntry(name, factory))

    def sample(self, rng: random.Random) -> OpponentPolicy:
        if not self.entries:
            raise ValueError("Opponent pool is empty")
        return rng.choice(self.entries).factory()

    @classmethod
    def with_baselines(cls) -> OpponentPool:
        return cls(
            [
                OpponentEntry("rating-noise", RatingNoisePolicy),
                OpponentEntry("budget-proportion", BudgetProportionPolicy),
                OpponentEntry("aggressive-high-value", AggressiveHighValuePolicy),
                OpponentEntry("random-pass", RandomPassPolicy),
                OpponentEntry("deal-probability", DealProbabilityPolicy),
            ]
        )
