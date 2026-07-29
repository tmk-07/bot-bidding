from __future__ import annotations

import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .models import (
    AuctionResult,
    GameResult,
    Item,
    Observation,
    PublicBotState,
    RevealedItem,
    RosterEntry,
)


@dataclass(frozen=True, slots=True)
class AuctionConfig:
    budget: int = 250
    roster_size: int = 5
    pool_size: int = 30
    require_full_rosters: bool = True
    min_price: int = 1
    value_min: int = 1
    value_max: int = 100

    def validate(self, player_count: int) -> None:
        if player_count < 2:
            raise ValueError("At least two bots are required")
        if self.require_full_rosters and (
            self.budget < self.roster_size * self.min_price
        ):
            raise ValueError("Budget cannot cover the minimum roster cost")
        if self.require_full_rosters and (
            self.pool_size < player_count * self.roster_size
        ):
            raise ValueError(
                "Pool must contain at least player_count * roster_size items"
            )
        if self.value_min > self.value_max:
            raise ValueError("value_min must not exceed value_max")


@dataclass(slots=True)
class _BotState:
    budget: int
    roster: list[RosterEntry]


class SequentialAuctionEnv:
    """Gym-like multi-agent environment for a sequential item auction.

    Call ``reset`` to reveal the first item, then call ``step`` with one maximum
    bid per bot. The highest bidder wins and pays one more than the second bid
    (bounded by its own bid), which is the outcome of an English auction.
    """

    def __init__(
        self,
        bot_names: Sequence[str],
        config: AuctionConfig | None = None,
        items: Iterable[Item] | None = None,
    ) -> None:
        if len(set(bot_names)) != len(bot_names):
            raise ValueError("Bot names must be unique")
        self.bot_names = tuple(bot_names)
        self.config = config or AuctionConfig()
        self.config.validate(len(self.bot_names))
        self._item_template = tuple(items) if items is not None else None
        if (
            self.config.require_full_rosters
            and self._item_template is not None
            and (
            len(self._item_template) < len(self.bot_names) * self.config.roster_size
            )
        ):
            raise ValueError("Custom item pool is too small to fill every roster")
        self._rng = random.Random()
        self._seed: int | None = None
        self._items: tuple[Item, ...] = ()
        self._states: dict[str, _BotState] = {}
        self._history: list[AuctionResult] = []
        self._index = 0
        self._done = True

    @property
    def current_item(self) -> Item | None:
        return None if self._done else self._items[self._index]

    @property
    def done(self) -> bool:
        return self._done

    def _default_items(self) -> tuple[Item, ...]:
        value_count = self.config.value_max - self.config.value_min + 1
        if self.config.pool_size > value_count:
            raise ValueError("Pool is too large to sample values without replacement")
        values = self._rng.sample(
            range(self.config.value_min, self.config.value_max + 1),
            self.config.pool_size,
        )
        return tuple(
            Item(
                id=f"item-{index + 1}",
                name=f"Item {index + 1}",
                value=value,
            )
            for index, value in enumerate(values)
        )

    def reset(self, seed: int | None = None) -> Mapping[str, Observation]:
        self._seed = seed
        self._rng = random.Random(seed)
        if self._item_template is None:
            items = list(self._default_items())
        else:
            items = list(self._item_template)
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError("Custom item IDs must be unique")
            self._rng.shuffle(items)
        self._items = tuple(items)
        self._states = {
            name: _BotState(self.config.budget, []) for name in self.bot_names
        }
        self._history = []
        self._index = 0
        self._done = False
        return self.observations()

    def max_legal_bid(self, bot_name: str) -> int:
        state = self._states[bot_name]
        if not self.config.require_full_rosters:
            return state.budget
        slots_after_win = self.config.roster_size - len(state.roster) - 1
        if slots_after_win < 0:
            return 0
        return max(0, state.budget - slots_after_win * self.config.min_price)

    def _observation(self, bot_name: str) -> Observation:
        state = self._states[bot_name]
        opponents = tuple(
            PublicBotState(
                name=name,
                budget=other.budget,
                roster_size=len(other.roster),
                score=sum(entry.item.value for entry in other.roster),
            )
            for name, other in self._states.items()
            if name != bot_name
        )
        return Observation(
            bot_name=bot_name,
            item=RevealedItem.from_item(self._items[self._index]),
            budget=state.budget,
            roster=tuple(state.roster),
            roster_size=len(state.roster),
            roster_target=self.config.roster_size,
            auction_index=self._index,
            total_auctions=len(self._items),
            opponents=opponents,
            history=tuple(self._history),
            min_price=self.config.min_price,
            max_legal_bid=self.max_legal_bid(bot_name),
            value_range=(self.config.value_min, self.config.value_max),
        )

    def observations(self) -> Mapping[str, Observation]:
        if self._done:
            return MappingProxyType({})
        return MappingProxyType(
            {
                name: self._observation(name)
                for name in self.bot_names
                if len(self._states[name].roster) < self.config.roster_size
            }
        )

    def step(
        self, bids: Mapping[str, int | float]
    ) -> tuple[Mapping[str, Observation], AuctionResult, bool]:
        if self._done:
            raise RuntimeError("Game is done; call reset()")
        eligible = {
            name
            for name in self.bot_names
            if len(self._states[name].roster) < self.config.roster_size
        }
        clean_bids: dict[str, int] = {}
        for name in self.bot_names:
            raw = bids.get(name, 0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"Bid for {name} must be numeric")
            bid = max(0, int(raw))
            clean_bids[name] = min(bid, self.max_legal_bid(name)) if name in eligible else 0

        positive = [name for name in eligible if clean_bids[name] >= self.config.min_price]
        winner: str | None = None
        price = 0
        if positive:
            top_bid = max(clean_bids[name] for name in positive)
            tied = [name for name in positive if clean_bids[name] == top_bid]
            winner = self._rng.choice(tied)
            losing_bids = [clean_bids[name] for name in positive if name != winner]
            runner_up = max(losing_bids, default=0)
            price = min(top_bid, max(self.config.min_price, runner_up + 1))
        return self.resolve_current(winner, price, clean_bids)

    def resolve_current(
        self,
        winner: str | None,
        price: int,
        bids: Mapping[str, int],
    ) -> tuple[Mapping[str, Observation], AuctionResult, bool]:
        """Finalize the current item at an explicit price.

        ``step`` uses this after calculating the proxy-auction outcome. A
        turn-based auction controller can use it after an ascending bid sequence.
        """

        if self._done:
            raise RuntimeError("Game is done; call reset()")
        if winner is None:
            if price != 0:
                raise ValueError("An unsold item must have price 0")
        else:
            if winner not in self._states:
                raise ValueError(f"Unknown winner: {winner}")
            if price < self.config.min_price:
                raise ValueError("Winning price is below the minimum price")
            if price > self.max_legal_bid(winner):
                raise ValueError("Winning price exceeds the winner's legal bid")
            state = self._states[winner]
            if len(state.roster) >= self.config.roster_size:
                raise ValueError("Winner's roster is already full")
            state.budget -= price
            state.roster.append(RosterEntry(self._items[self._index], price))

        clean_bids = {
            name: max(0, int(bids.get(name, 0))) for name in self.bot_names
        }
        result = AuctionResult.create(self._items[self._index], winner, price, clean_bids)
        self._history.append(result)
        self._index += 1
        all_full = self.config.require_full_rosters and all(
            len(state.roster) >= self.config.roster_size
            for state in self._states.values()
        )
        self._done = all_full or self._index >= len(self._items)
        return self.observations(), result, self._done

    def result(self) -> GameResult:
        if not self._done:
            raise RuntimeError("Game is still in progress")
        rosters = {
            name: tuple(state.roster) for name, state in self._states.items()
        }
        scores = {
            name: sum(entry.item.value for entry in roster)
            for name, roster in rosters.items()
        }
        top_score = max(scores.values())
        return GameResult(
            seed=self._seed,
            rosters=MappingProxyType(rosters),
            budgets=MappingProxyType(
                {name: state.budget for name, state in self._states.items()}
            ),
            scores=MappingProxyType(scores),
            auctions=tuple(self._history),
            winners=tuple(name for name, score in scores.items() if score == top_score),
        )
