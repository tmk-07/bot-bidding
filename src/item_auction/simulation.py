from __future__ import annotations

from collections.abc import Sequence

from .agents import Bot
from .environment import AuctionConfig, SequentialAuctionEnv
from .models import GameResult, Item


def play_game(
    bots: Sequence[Bot],
    *,
    config: AuctionConfig | None = None,
    items: Sequence[Item] | None = None,
    seed: int | None = None,
) -> GameResult:
    env = SequentialAuctionEnv([bot.name for bot in bots], config, items)
    observations = env.reset(seed)
    for index, bot in enumerate(bots):
        bot.reset(None if seed is None else seed * 1009 + index)
    while not env.done:
        bids = {
            bot.name: bot.bid(observations[bot.name])
            for bot in bots
            if bot.name in observations
        }
        observations, _, _ = env.step(bids)
    return env.result()
