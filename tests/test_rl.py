import random

import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test

from item_auction.core import AuctionEngine
from item_auction.rl import (
    ACTION_COUNT,
    AggressiveHighValuePolicy,
    AuctionAECEnv,
    BidAction,
    BudgetProportionPolicy,
    CallablePolicy,
    DealProbabilityPolicy,
    FixedOpponentEnv,
    OpponentPool,
    RandomPassPolicy,
    RatingNoisePolicy,
    TrainingHistory,
    action_to_bid,
    legal_action_mask,
    rating_band,
)


def test_budget_aware_actions_use_both_players_cash() -> None:
    assert (
        action_to_bid(
            BidAction.OWN_50,
            current_bid=20,
            own_budget=200,
            opponent_budget=80,
        )
        == 100
    )
    assert (
        action_to_bid(
            BidAction.OPP_50,
            current_bid=20,
            own_budget=200,
            opponent_budget=80,
        )
        == 40
    )
    assert (
        action_to_bid(
            BidAction.OPP_ALL_IN_PLUS_ONE,
            current_bid=20,
            own_budget=200,
            opponent_budget=80,
        )
        == 81
    )
    mask = legal_action_mask(
        current_bid=90, own_budget=100, opponent_budget=80
    )
    assert mask.shape == (ACTION_COUNT,)
    assert mask[BidAction.PASS]
    assert mask[BidAction.MIN_RAISE]
    assert not mask[BidAction.OPP_100]


def test_pettingzoo_and_gymnasium_api_compliance() -> None:
    api_test(AuctionAECEnv(), num_cycles=500, verbose_progress=False)
    check_env(FixedOpponentEnv(RatingNoisePolicy), skip_render_check=True)


def test_every_baseline_completes_full_episodes() -> None:
    policies = [
        RatingNoisePolicy,
        BudgetProportionPolicy,
        AggressiveHighValuePolicy,
        RandomPassPolicy,
        DealProbabilityPolicy,
    ]
    for index, policy in enumerate(policies):
        env = FixedOpponentEnv(policy, randomize_seat=True)
        observation, info = env.reset(seed=100 + index)
        terminated = False
        steps = 0
        while not terminated:
            mask = observation["action_mask"]
            legal = np.flatnonzero(mask)
            # A simple learner that raises the minimum when possible.
            action = (
                int(BidAction.MIN_RAISE)
                if mask[BidAction.MIN_RAISE]
                else int(BidAction.PASS)
            )
            observation, reward, terminated, truncated, info = env.step(action)
            assert not truncated
            steps += 1
            assert steps < 2_000
        assert len(env.aec.auction_history) == 20
        assert sum(env.aec.item_counts.values()) <= 20
        assert sum(env.aec.budgets.values()) >= 0
        assert info["opponent"] == policy.name


def test_future_checkpoint_can_join_opponent_pool_through_adapter() -> None:
    def predict(observation, mask):
        if mask[BidAction.MIN_RAISE] and observation.current_bid < 10:
            return int(BidAction.MIN_RAISE)
        return int(BidAction.PASS)

    pool = OpponentPool.with_baselines()
    pool.add("checkpoint-v1", lambda: CallablePolicy("checkpoint-v1", predict))
    assert len(pool.entries) == 6
    sampled_names = {
        pool.sample(random.Random(seed)).name for seed in range(100)
    }
    assert "checkpoint-v1" in sampled_names


def test_training_pool_excludes_pure_random_policy() -> None:
    names = {entry.name for entry in OpponentPool.training_baselines().entries}
    assert names == {
        "rating-noise",
        "budget-proportion",
        "aggressive-high-value",
        "deal-probability",
    }


def test_training_history_records_players_and_every_selection(tmp_path) -> None:
    history = TrainingHistory(tmp_path / "history.sqlite3")
    run_id = history.create_run(
        algorithm="test",
        total_timesteps=10,
        seed=5,
        config={"purpose": "test"},
    )
    engine = AuctionEngine(pool_size=2)
    engine.reset(seed=42)
    engine.place_bid("player_0", 12)
    engine.pass_turn("player_1")
    engine.pass_turn("player_1")
    engine.place_bid("player_0", 7)
    engine.pass_turn("player_1")
    assert engine.done

    history.record_game(
        run_id=run_id,
        checkpoint_steps=10,
        episode=0,
        seed=42,
        opponent_name="rating-noise",
        learner_agent="player_0",
        engine=engine,
        learner_actions=[
            {
                "auction_number": 1,
                "rating": 82,
                "current_bid": 0,
                "own_budget": 500,
                "opponent_budget": 500,
                "action_index": int(BidAction.MIN_RAISE),
                "action_name": BidAction.MIN_RAISE.name,
                "target_bid": 1,
            }
        ],
    )
    summary = history.player_summary(run_id, 10)
    selections = history.selections(run_id, 10)
    ratings = history.rating_summary(run_id, 10)

    assert len(summary) == 2
    assert len(selections) == 2
    assert sum(row["items"] for row in ratings) == 2
    assert selections[0]["winner"] in {"RL Bot", "rating-noise", "Unsold"}
    assert rating_band(100) == "100"
    assert rating_band(9) == "1-9"
    assert history.action_progress(run_id) == [
        {
            "checkpoint_steps": 10,
            "action_name": "MIN_RAISE",
            "uses": 1,
            "action_pct": 100.0,
        }
    ]


def test_observation_contains_current_state_but_not_future_pool() -> None:
    env = AuctionAECEnv()
    env.reset(seed=42)
    observation = env.observe("player_0")
    assert observation["observation"].shape == (12,)
    assert observation["action_mask"].shape == (ACTION_COUNT,)
    assert observation["observation"][0] == env.current_rating / 100
