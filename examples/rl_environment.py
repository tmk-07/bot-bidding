import random

import numpy as np

from item_auction.rl import (
    BidAction,
    CallablePolicy,
    FixedOpponentEnv,
    OpponentPool,
)


pool = OpponentPool.with_baselines()

# A future frozen model can be registered without changing the environment.
# Replace this callable with model.predict(...) after training.
pool.add(
    "example-checkpoint",
    lambda: CallablePolicy(
        "example-checkpoint",
        lambda observation, mask: (
            int(BidAction.MIN_RAISE)
            if mask[BidAction.MIN_RAISE] and observation.current_bid < 25
            else int(BidAction.PASS)
        ),
    ),
)

rng = random.Random(42)
env = FixedOpponentEnv(lambda: pool.sample(rng))
observation, info = env.reset(seed=42)
done = False

while not done:
    legal_actions = np.flatnonzero(observation["action_mask"])
    action = int(rng.choice(legal_actions))
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(info)
