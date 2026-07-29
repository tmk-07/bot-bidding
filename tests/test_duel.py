import pytest

from item_auction import AscendingAuctionDuel


def test_bot_raises_then_human_can_pass() -> None:
    duel = AscendingAuctionDuel()
    duel.reset(seed=42)
    response = duel.human_raise(10)
    assert not response.resolved
    assert duel.leader == "Random Bot"
    assert duel.current_price == 11

    response = duel.human_pass()
    assert response.resolved
    assert response.auction.winner == "Random Bot"
    assert response.auction.price == 11


def test_human_jump_bid_can_make_bot_hold() -> None:
    duel = AscendingAuctionDuel()
    duel.reset(seed=1)  # Bot max is about 13.4% of $500.
    response = duel.human_raise(100)
    assert response.resolved
    assert response.auction.winner == "You"
    assert response.auction.price == 100


def test_bid_must_raise_visible_price_and_fit_budget() -> None:
    duel = AscendingAuctionDuel()
    duel.reset(seed=42)
    duel.human_raise(10)
    with pytest.raises(ValueError, match="at least"):
        duel.human_raise(11)
    with pytest.raises(ValueError, match="budget"):
        duel.human_raise(501)


def test_passing_before_an_opening_bid_costs_bot_one_dollar() -> None:
    duel = AscendingAuctionDuel()
    duel.reset(seed=42)
    response = duel.human_pass()
    assert response.auction.winner == "Random Bot"
    assert response.auction.price == 1
