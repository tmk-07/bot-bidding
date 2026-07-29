from item_auction.training import train_policy


def test_small_training_run_completes() -> None:
    policy, history = train_policy(
        generations=2, population=4, episodes=2, seed=7
    )
    assert len(history) == 2
    assert policy.scale >= 0.2
