from __future__ import annotations

import random
from dataclasses import dataclass

from .core import AuctionEngine
from .models import AuctionResult, GameResult, Item


@dataclass(frozen=True, slots=True)
class BidResponse:
    message: str
    resolved: bool
    auction: AuctionResult | None = None


class AscendingAuctionDuel:
    """Human-facing controller over the canonical AuctionEngine."""

    def __init__(
        self,
        human_name: str = "You",
        bot_name: str = "Random Bot",
        *,
        budget: int = 500,
        pool_size: int = 20,
        items: list[Item] | None = None,
    ) -> None:
        self.human_name = human_name
        self.bot_name = bot_name
        self.engine = AuctionEngine(
            (human_name, bot_name),
            budget=budget,
            pool_size=pool_size,
            items=items,
        )
        self.env = self.engine  # Backward-compatible alias for callers.
        self._bot_max = 0
        self._rng = random.Random()

    @property
    def done(self) -> bool:
        return self.engine.done

    @property
    def current_item(self) -> Item | None:
        return self.engine.current_item

    @property
    def current_price(self) -> int:
        return self.engine.current_bid

    @property
    def leader(self) -> str | None:
        return self.engine.leader

    @property
    def minimum_raise(self) -> int:
        return self.engine.current_bid + 1

    @property
    def human_budget(self) -> int:
        return self.engine.budgets[self.human_name]

    @property
    def bot_budget(self) -> int:
        return self.engine.budgets[self.bot_name]

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.engine.reset(seed)
        self._prepare_item()

    def _prepare_item(self) -> None:
        if self.engine.done:
            self._bot_max = 0
            return
        self._bot_max = round(self.bot_budget * self._rng.random())
        if self.engine.agent_selection == self.bot_name:
            if self._bot_max >= 1 and self.bot_budget >= 1:
                self.engine.place_bid(self.bot_name, 1)
            else:
                self.engine.pass_turn(self.bot_name)

    def _after_resolution(self, response: BidResponse) -> BidResponse:
        self._prepare_item()
        return response

    def human_raise(self, amount: int) -> BidResponse:
        transition = self.engine.place_bid(self.human_name, int(amount))
        bot_raise = int(amount) + 1
        if bot_raise <= self._bot_max and bot_raise <= self.bot_budget:
            self.engine.place_bid(self.bot_name, bot_raise)
            return BidResponse(
                message=f"{self.bot_name} raises to ${bot_raise}.",
                resolved=False,
            )

        item_name = self.engine.current_item.name
        transition = self.engine.pass_turn(self.bot_name)
        return self._after_resolution(
            BidResponse(
                message=f"{self.bot_name} holds. You win {item_name} for ${amount}.",
                resolved=True,
                auction=transition.auction,
            )
        )

    def human_pass(self) -> BidResponse:
        item_name = self.engine.current_item.name
        if self.engine.leader == self.bot_name:
            price = self.engine.current_bid
            transition = self.engine.pass_turn(self.human_name)
            return self._after_resolution(
                BidResponse(
                    message=f"You pass. {self.bot_name} wins {item_name} for ${price}.",
                    resolved=True,
                    auction=transition.auction,
                )
            )

        transition = self.engine.pass_turn(self.human_name)
        if transition.resolved:
            return self._after_resolution(
                BidResponse(
                    message=f"Both players pass on {item_name}.",
                    resolved=True,
                    auction=transition.auction,
                )
            )
        if self._bot_max >= 1 and self.bot_budget >= 1:
            self.engine.place_bid(self.bot_name, 1)
            transition = self.engine.pass_turn(self.human_name)
            return self._after_resolution(
                BidResponse(
                    message=f"You pass. {self.bot_name} takes {item_name} for $1.",
                    resolved=True,
                    auction=transition.auction,
                )
            )
        transition = self.engine.pass_turn(self.bot_name)
        return self._after_resolution(
            BidResponse(
                message=f"Both players pass on {item_name}.",
                resolved=True,
                auction=transition.auction,
            )
        )

    def result(self) -> GameResult:
        return self.engine.result()
