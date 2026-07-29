from __future__ import annotations

from item_auction.agents import LinearBot, ValueBot
from item_auction.environment import AuctionConfig, SequentialAuctionEnv
from item_auction.simulation import play_game


def test_observation_does_not_expose_future_items() -> None:
    env = SequentialAuctionEnv(["a", "b"], AuctionConfig(pool_size=10))
    observations = env.reset(seed=9)
    observation = observations["a"]
    assert observation.item.name == "Item 1"
    assert not hasattr(observation.item, "value")
    assert not hasattr(observation, "items")
    assert observation.history == ()


def test_bid_is_clamped_and_reserves_minimum_for_open_slots() -> None:
    env = SequentialAuctionEnv(["a", "b"], AuctionConfig(pool_size=10))
    env.reset(seed=4)
    _, result, _ = env.step({"a": 10_000, "b": 1})
    assert result.bids["a"] == 246
    assert result.price == 2


def test_seeded_game_is_reproducible_and_valid() -> None:
    def run():
        return play_game(
            [LinearBot("linear"), ValueBot("steady")],
            config=AuctionConfig(pool_size=12),
            seed=123,
        )

    first, second = run(), run()
    assert first.scores == second.scores
    assert first.budgets == second.budgets
    assert first.auctions == second.auctions
    for name in first.rosters:
        assert len(first.rosters[name]) == 5
        assert first.budgets[name] >= 0
        assert sum(entry.price for entry in first.rosters[name]) + first.budgets[name] == 250


def test_custom_pool_is_shuffled_without_being_leaked() -> None:
    from item_auction.models import Item

    items = [Item(str(i), f"custom-{i}", i + 1) for i in range(10)]
    env = SequentialAuctionEnv(["a", "b"], AuctionConfig(pool_size=10), items)
    first = env.reset(seed=1)["a"].item
    second = env.reset(seed=2)["a"].item
    assert first != second


def test_open_roster_duel_can_end_twenty_to_zero() -> None:
    config = AuctionConfig(
        budget=500,
        roster_size=20,
        pool_size=20,
        require_full_rosters=False,
    )
    env = SequentialAuctionEnv(["human", "bot"], config)
    observations = env.reset(seed=11)
    assert observations["human"].max_legal_bid == 500
    for _ in range(20):
        observations, _, _ = env.step({"human": 500, "bot": 0})
    result = env.result()
    assert len(result.rosters["human"]) == 20
    assert len(result.rosters["bot"]) == 0
    assert len(result.auctions) == 20
    assert result.budgets["human"] == 480
