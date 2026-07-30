import random

import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test

from item_auction.core import AuctionEngine
from item_auction.rl.baseline_training import (
    available_entries,
    evaluate_policy_checkpoint,
    selected_entries,
)
from item_auction.rl import (
    ACTION_COUNT,
    AggressiveHighValuePolicy,
    AuctionAECEnv,
    BidAction,
    BudgetProportionPolicy,
    CallablePolicy,
    DealProbabilityPolicy,
    FixedOpponentEnv,
    MarketPressurePolicy,
    OpponentPool,
    PolicyObservation,
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
    available_names = {entry.name for entry in available_entries()}
    assert {
        "rating-exact",
        "rating-noise-5",
        "rating-noise-20",
        "market-pressure",
    } <= available_names


def test_value_only_training_hides_budget_and_match_context() -> None:
    opponent = lambda: CallablePolicy(
        "pass",
        lambda observation, mask: int(BidAction.PASS),
    )
    env = FixedOpponentEnv(
        opponent,
        randomize_seat=False,
        observation_mode="value-only",
    )
    observation, _ = env.reset(seed=12)
    values = observation["observation"]
    assert values[0] > 0
    assert np.all(values[[2, 3, 4, 5, 6, 7, 8, 11]] == 0)
    assert observation["action_mask"][BidAction.PASS]


def test_value_calibration_penalizes_passing_below_rating() -> None:
    opponent = lambda: CallablePolicy(
        "pass",
        lambda observation, mask: int(BidAction.PASS),
    )
    standard = FixedOpponentEnv(opponent, randomize_seat=False)
    calibrated = FixedOpponentEnv(
        opponent,
        randomize_seat=False,
        reward_mode="value-calibration",
    )
    standard.reset(seed=12)
    calibrated.reset(seed=12)
    _, standard_reward, *_ = standard.step(BidAction.PASS)
    _, calibrated_reward, *_ = calibrated.step(BidAction.PASS)
    assert calibrated_reward == standard_reward - 0.10


def test_market_pressure_policy_uses_public_context() -> None:
    policy = MarketPressurePolicy()
    observation = PolicyObservation(
        item_rating=80,
        current_bid=79,
        own_budget=300,
        opponent_budget=350,
        own_score=200,
        opponent_score=260,
        own_items=4,
        opponent_items=5,
        items_remaining=5,
        is_opening_turn=False,
    )
    mask = np.ones(ACTION_COUNT, dtype=np.int8)

    assert policy.reservation_price(observation) == 81
    assert policy.act(observation, mask) == BidAction.MIN_RAISE


def test_scripted_policy_evaluation_records_full_history(tmp_path) -> None:
    history = TrainingHistory(tmp_path / "policy-history.sqlite3")
    run_id = history.create_run(
        algorithm="TunedPolicy",
        total_timesteps=0,
        seed=5,
        config={"training_phase": "held-out-benchmark"},
        learner_family="market-generalist",
    )
    evaluate_policy_checkpoint(
        MarketPressurePolicy,
        history=history,
        run_id=run_id,
        checkpoint_steps=0,
        episodes_per_opponent=2,
        seed=90,
        entries=selected_entries(opponent_names=["rating-exact"]),
    )

    assert len(history.selections(run_id, 0)) == 40
    assert sum(row["games"] for row in history.player_summary(run_id, 0)) == 4


def test_incremental_learner_only_exposes_pass_and_minimum_raise() -> None:
    opponent = lambda: CallablePolicy(
        "pass",
        lambda observation, mask: int(BidAction.PASS),
    )
    env = FixedOpponentEnv(
        opponent,
        randomize_seat=False,
        learner_action_mode="incremental",
    )
    observation, _ = env.reset(seed=12)
    assert observation["action_mask"][BidAction.PASS]
    assert observation["action_mask"][BidAction.MIN_RAISE]
    assert not observation["action_mask"][BidAction.OWN_10 :].any()
    assert np.array_equal(observation["action_mask"], env.action_masks())


def test_training_history_records_players_and_every_selection(tmp_path) -> None:
    history = TrainingHistory(tmp_path / "history.sqlite3")
    run_id = history.create_run(
        algorithm="test",
        total_timesteps=10,
        seed=5,
        config={"purpose": "test"},
        learner_family="deal-value",
    )
    assert history.runs()[0]["learner_family"] == "deal-value"
    engine = AuctionEngine(pool_size=2)
    engine.reset(seed=42)
    engine.place_bid("player_0", 12)
    engine.pass_turn("player_1")
    engine.place_bid("player_1", 7)
    engine.pass_turn("player_0")
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
    pricing = history.pricing_behavior(run_id, 10)

    assert len(summary) == 2
    assert len(selections) == 2
    assert sum(row["items"] for row in ratings) == 2
    assert sum(row["items"] for row in pricing) == 2
    assert all("avg_learner_offer" in row for row in pricing)
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
    curve = history.bid_curve(run_id)
    assert len(curve) == 2
    assert sum(row["stop_samples"] for row in curve) == 1
    history.record_training_exposure(
        run_id,
        {"rating-exact": 8, "frozen-test": 2},
    )
    exposure = history.training_exposure(run_id)
    assert exposure[0] == {
        "opponent": "rating-exact",
        "training_matches": 8,
        "actual_share_pct": 80.0,
    }
    history.record_game(
        run_id=run_id,
        checkpoint_steps=20,
        episode=1,
        seed=43,
        opponent_name="rating-noise",
        learner_agent="player_0",
        engine=engine,
    )
    history.select_checkpoint(
        run_id,
        20,
        model_path="models/candidate.zip",
    )
    selected_run = history.runs()[0]
    assert selected_run["selected_checkpoint_steps"] == 20
    assert selected_run["model_path"] == "models/candidate.zip"
    removed = history.truncate_run(
        run_id,
        10,
        model_path="models/best.zip",
    )
    assert removed == 1
    assert history.checkpoints(run_id) == [10]
    assert history.training_exposure(run_id) == []


def test_observation_contains_current_state_but_not_future_pool() -> None:
    env = AuctionAECEnv()
    env.reset(seed=42)
    observation = env.observe("player_0")
    assert observation["observation"].shape == (12,)
    assert observation["action_mask"].shape == (ACTION_COUNT,)
    assert observation["observation"][0] == env.current_rating / 100
