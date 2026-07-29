from __future__ import annotations

import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from .models import AuctionResult, GameResult, Item, RosterEntry


@dataclass(frozen=True, slots=True)
class EngineTransition:
    resolved: bool
    auction: AuctionResult | None


class AuctionEngine:
    """Canonical two-player, turn-based ascending-auction engine.

    Both the human UI and RL environments delegate all bidding, turn order,
    budget accounting, item resolution, and scoring to this class.
    """

    def __init__(
        self,
        agents: tuple[str, str] = ("player_0", "player_1"),
        *,
        budget: int = 500,
        pool_size: int = 20,
        value_min: int = 1,
        value_max: int = 100,
        items: Iterable[Item] | None = None,
    ) -> None:
        if len(set(agents)) != 2:
            raise ValueError("AuctionEngine requires two unique agent names")
        if budget < 1:
            raise ValueError("budget must be positive")
        if pool_size < 1:
            raise ValueError("pool_size must be positive")
        if value_min > value_max:
            raise ValueError("value_min must not exceed value_max")
        if items is None and pool_size > value_max - value_min + 1:
            raise ValueError("Pool is too large to sample ratings without replacement")
        self.agents = agents
        self.budget_start = budget
        self.pool_size = pool_size
        self.value_min = value_min
        self.value_max = value_max
        self._item_template = tuple(items) if items is not None else None
        if self._item_template is not None:
            if len(self._item_template) != pool_size:
                raise ValueError(f"Custom pool must contain exactly {pool_size} items")
            ids = [item.id for item in self._item_template]
            if len(set(ids)) != len(ids):
                raise ValueError("Custom item IDs must be unique; duplicates are not allowed")
        self._rng = random.Random()
        self._seed: int | None = None
        self._items: tuple[Item, ...] = ()
        self.item_index = 0
        self.budgets: dict[str, int] = {}
        self.scores: dict[str, float] = {}
        self.rosters: dict[str, list[RosterEntry]] = {}
        self.current_bid = 0
        self.leader: str | None = None
        self.first_pass: str | None = None
        self.current_opener = agents[0]
        self.agent_selection: str | None = agents[0]
        self.last_offers: dict[str, int] = {}
        self.history: list[AuctionResult] = []
        self.done = True
        self.winner: str | None = None

    @staticmethod
    def _other_from(agents: tuple[str, str], agent: str) -> str:
        if agent == agents[0]:
            return agents[1]
        if agent == agents[1]:
            return agents[0]
        raise ValueError(f"Unknown agent: {agent}")

    def other(self, agent: str) -> str:
        return self._other_from(self.agents, agent)

    @property
    def current_item(self) -> Item | None:
        return None if self.done else self._items[self.item_index]

    @property
    def items_remaining(self) -> int:
        return max(0, self.pool_size - self.item_index)

    def _make_items(self) -> tuple[Item, ...]:
        if self._item_template is not None:
            items = list(self._item_template)
            self._rng.shuffle(items)
            return tuple(items)
        ratings = self._rng.sample(
            range(self.value_min, self.value_max + 1), self.pool_size
        )
        return tuple(
            Item(
                id=f"rating-{rating}",
                name=f"Item {index + 1}",
                value=float(rating),
            )
            for index, rating in enumerate(ratings)
        )

    def reset(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._items = self._make_items()
        self.item_index = 0
        self.budgets = {agent: self.budget_start for agent in self.agents}
        self.scores = {agent: 0.0 for agent in self.agents}
        self.rosters = {agent: [] for agent in self.agents}
        self.history = []
        self.current_opener = self.agents[0]
        self.agent_selection = self.current_opener
        self.current_bid = 0
        self.leader = None
        self.first_pass = None
        self.last_offers = {agent: 0 for agent in self.agents}
        self.done = False
        self.winner = None

    def _require_turn(self, agent: str) -> None:
        if self.done:
            raise RuntimeError("Draft is complete")
        if agent != self.agent_selection:
            raise ValueError(f"It is {self.agent_selection}'s turn, not {agent}'s")

    def place_bid(self, agent: str, amount: int) -> EngineTransition:
        self._require_turn(agent)
        amount = int(amount)
        if amount <= self.current_bid:
            raise ValueError(f"Bid must be at least ${self.current_bid + 1}")
        if amount > self.budgets[agent]:
            raise ValueError("Bid exceeds remaining budget")
        self.current_bid = amount
        self.leader = agent
        self.first_pass = None
        self.last_offers[agent] = amount
        self.agent_selection = self.other(agent)
        return EngineTransition(False, None)

    def pass_turn(self, agent: str) -> EngineTransition:
        self._require_turn(agent)
        if self.leader is not None:
            return self._resolve(self.leader)
        if self.first_pass is None:
            self.first_pass = agent
            self.agent_selection = self.other(agent)
            return EngineTransition(False, None)
        return self._resolve(None)

    def _resolve(self, winner: str | None) -> EngineTransition:
        item = self.current_item
        price = self.current_bid if winner else 0
        if winner:
            self.budgets[winner] -= price
            self.scores[winner] += item.value
            self.rosters[winner].append(RosterEntry(item, price))
        auction = AuctionResult.create(item, winner, price, self.last_offers)
        self.history.append(auction)
        self.item_index += 1
        if self.item_index >= self.pool_size:
            self.done = True
            self.agent_selection = None
            score_values = list(self.scores.values())
            if score_values[0] > score_values[1]:
                self.winner = self.agents[0]
            elif score_values[1] > score_values[0]:
                self.winner = self.agents[1]
            else:
                self.winner = None
        else:
            self.current_opener = self.other(self.current_opener)
            self.agent_selection = self.current_opener
            self.current_bid = 0
            self.leader = None
            self.first_pass = None
            self.last_offers = {agent: 0 for agent in self.agents}
        return EngineTransition(True, auction)

    def result(self) -> GameResult:
        if not self.done:
            raise RuntimeError("Draft is still in progress")
        top_score = max(self.scores.values())
        return GameResult(
            seed=self._seed,
            rosters=MappingProxyType(
                {agent: tuple(roster) for agent, roster in self.rosters.items()}
            ),
            budgets=MappingProxyType(dict(self.budgets)),
            scores=MappingProxyType(dict(self.scores)),
            auctions=tuple(self.history),
            winners=tuple(
                agent for agent, score in self.scores.items() if score == top_score
            ),
        )
