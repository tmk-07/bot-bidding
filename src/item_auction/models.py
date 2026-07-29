from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Item:
    """An auctionable item.

    ``value`` is the score awarded to its owner. ``estimate`` is what bots see.
    They are identical by default, but separating them makes noisy information
    and hidden true values possible without changing the environment API.
    """

    id: str
    name: str
    value: float
    estimate: float | None = None
    category: str = "default"

    @property
    def visible_value(self) -> float:
        return self.value if self.estimate is None else self.estimate


@dataclass(frozen=True, slots=True)
class RevealedItem:
    """The safe, public view of the item currently on the auction block."""

    id: str
    name: str
    visible_value: float
    category: str

    @classmethod
    def from_item(cls, item: Item) -> RevealedItem:
        return cls(item.id, item.name, item.visible_value, item.category)


@dataclass(frozen=True, slots=True)
class RosterEntry:
    item: Item
    price: int


@dataclass(frozen=True, slots=True)
class PublicBotState:
    name: str
    budget: int
    roster_size: int
    score: float


@dataclass(frozen=True, slots=True)
class AuctionResult:
    item: Item
    winner: str | None
    price: int
    bids: Mapping[str, int]

    @classmethod
    def create(
        cls, item: Item, winner: str | None, price: int, bids: Mapping[str, int]
    ) -> AuctionResult:
        return cls(item, winner, price, MappingProxyType(dict(bids)))


@dataclass(frozen=True, slots=True)
class Observation:
    """The complete information available to one bot for the current auction."""

    bot_name: str
    item: RevealedItem
    budget: int
    roster: tuple[RosterEntry, ...]
    roster_size: int
    roster_target: int
    auction_index: int
    total_auctions: int
    opponents: tuple[PublicBotState, ...]
    history: tuple[AuctionResult, ...]
    min_price: int
    max_legal_bid: int
    value_range: tuple[float, float]

    @property
    def slots_left(self) -> int:
        return self.roster_target - self.roster_size

    @property
    def auctions_left_including_current(self) -> int:
        return self.total_auctions - self.auction_index


@dataclass(frozen=True, slots=True)
class GameResult:
    seed: int | None
    rosters: Mapping[str, tuple[RosterEntry, ...]]
    budgets: Mapping[str, int]
    scores: Mapping[str, float]
    auctions: tuple[AuctionResult, ...]
    winners: tuple[str, ...]
