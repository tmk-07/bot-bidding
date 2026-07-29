from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .models import Observation


class Bot(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self, seed: int | None = None) -> None:
        """Reset episode-local state."""

    @abstractmethod
    def bid(self, observation: Observation) -> int:
        """Return the bot's maximum willingness to pay."""


class ValueBot(Bot):
    """A solid baseline that balances value, budget pace, and item scarcity."""

    def __init__(
        self,
        name: str,
        aggressiveness: float = 1.0,
        value_threshold: float = 0.35,
    ) -> None:
        super().__init__(name)
        self.aggressiveness = aggressiveness
        self.value_threshold = value_threshold

    def bid(self, observation: Observation) -> int:
        if observation.slots_left <= 0:
            return 0
        low, high = observation.value_range
        value = (observation.item.visible_value - low) / max(1.0, high - low)
        urgency = observation.slots_left / max(
            observation.slots_left, observation.auctions_left_including_current
        )
        threshold = max(0.05, self.value_threshold - urgency * 0.8)
        if value < threshold:
            return 0
        spendable = observation.max_legal_bid
        per_slot = observation.budget / observation.slots_left
        quality = (value - threshold) / max(0.01, 1.0 - threshold)
        bid = per_slot * (0.35 + 1.15 * quality) * self.aggressiveness
        return min(spendable, max(observation.min_price, round(bid)))


class RandomBot(Bot):
    def __init__(self, name: str, seed: int | None = None) -> None:
        super().__init__(name)
        self._base_seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(self._base_seed if seed is None else seed)

    def bid(self, observation: Observation) -> int:
        if observation.slots_left <= 0:
            return 0
        if self._rng.random() < 0.25:
            return 0
        cap = min(
            observation.max_legal_bid,
            max(observation.min_price, round(observation.budget / observation.slots_left)),
        )
        return self._rng.randint(observation.min_price, cap)


class RandomBudgetFractionBot(Bot):
    """Chooses a random percentage of remaining cash as its maximum bid."""

    def __init__(self, name: str, seed: int | None = None) -> None:
        super().__init__(name)
        self._base_seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(self._base_seed if seed is None else seed)

    def bid(self, observation: Observation) -> int:
        if observation.budget < observation.min_price:
            return 0
        percentage = self._rng.random()
        return min(observation.max_legal_bid, round(observation.budget * percentage))


@dataclass(slots=True)
class LinearPolicy:
    """Small serializable policy suitable for evolutionary training."""

    intercept: float = -0.55
    value: float = 1.45
    urgency: float = 0.55
    budget_pace: float = 0.20
    opponent_pressure: float = 0.15
    price_history: float = -0.10
    scale: float = 1.0

    def values(self) -> list[float]:
        return [
            self.intercept,
            self.value,
            self.urgency,
            self.budget_pace,
            self.opponent_pressure,
            self.price_history,
            self.scale,
        ]

    @classmethod
    def from_values(cls, values: list[float]) -> LinearPolicy:
        return cls(*values)


class LinearBot(Bot):
    """Trainable bot whose features use only the bot's observation."""

    def __init__(self, name: str, policy: LinearPolicy | None = None) -> None:
        super().__init__(name)
        self.policy = policy or LinearPolicy()

    def bid(self, observation: Observation) -> int:
        if observation.slots_left <= 0:
            return 0
        low, high = observation.value_range
        value = (observation.item.visible_value - low) / max(1.0, high - low)
        urgency = observation.slots_left / max(
            observation.slots_left, observation.auctions_left_including_current
        )
        starting_budget = observation.budget + sum(x.price for x in observation.roster)
        target_remaining = observation.slots_left / observation.roster_target
        budget_pace = observation.budget / max(1, starting_budget) - target_remaining
        opponent_pressure = (
            sum(o.budget for o in observation.opponents)
            / max(1, len(observation.opponents) * starting_budget)
            if observation.opponents
            else 0.0
        )
        sold = [auction.price for auction in observation.history if auction.winner]
        price_history = (
            sum(sold[-5:]) / len(sold[-5:]) / max(1, starting_budget)
            if sold
            else 0.0
        )
        p = self.policy
        desire = (
            p.intercept
            + p.value * value
            + p.urgency * urgency
            + p.budget_pace * budget_pace
            + p.opponent_pressure * opponent_pressure
            + p.price_history * price_history
        )
        if desire <= 0:
            return 0
        willingness = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, desire))))
        per_slot = observation.budget / observation.slots_left
        bid = round(per_slot * 1.6 * willingness * max(0.1, p.scale))
        if bid < observation.min_price:
            return 0
        return min(observation.max_legal_bid, bid)
