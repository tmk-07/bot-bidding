import pytest

from item_auction import AscendingAuctionDuel, AuctionEngine, Item
from item_auction.rl import AuctionAECEnv


def test_default_pool_samples_unique_ratings_without_replacement() -> None:
    engine = AuctionEngine(pool_size=20)
    engine.reset(seed=42)
    ratings = [item.value for item in engine._items]
    ids = [item.id for item in engine._items]
    assert len(ratings) == len(set(ratings)) == 20
    assert len(ids) == len(set(ids)) == 20


def test_custom_pool_rejects_duplicate_item_ids() -> None:
    items = [
        Item(id="duplicate", name="First", value=10),
        Item(id="duplicate", name="Second", value=20),
    ]
    with pytest.raises(ValueError, match="unique"):
        AuctionEngine(pool_size=2, items=items)


def test_streamlit_controller_and_rl_wrapper_share_canonical_engine() -> None:
    duel = AscendingAuctionDuel()
    rl_env = AuctionAECEnv()
    assert isinstance(duel.engine, AuctionEngine)
    assert isinstance(rl_env.engine, AuctionEngine)


def test_engine_alternates_openers_and_accounts_for_visible_bid() -> None:
    engine = AuctionEngine(pool_size=2)
    engine.reset(seed=7)
    assert engine.agent_selection == "player_0"
    engine.place_bid("player_0", 10)
    transition = engine.pass_turn("player_1")
    assert transition.auction.winner == "player_0"
    assert transition.auction.price == 10
    assert engine.agent_selection == "player_1"
    assert engine.budgets["player_0"] == 490
