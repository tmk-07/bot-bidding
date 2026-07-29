import random

import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test

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
    action_to_bid,
    legal_action_mask,
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


def test_observation_contains_current_state_but_not_future_pool() -> None:
    env = AuctionAECEnv()
    env.reset(seed=42)
    observation = env.observe("player_0")
    assert observation["observation"].shape == (12,)
    assert observation["action_mask"].shape == (ACTION_COUNT,)
    assert observation["observation"][0] == env.current_rating / 100
