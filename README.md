# Item Auction

A fast simulation environment for training bots to draft a team of **5 items**
with a **$250 budget**. Items have values from 1–100 and are revealed one at a
time, so agents cannot inspect the future pool.

Each bot submits its maximum bid for the current item. The highest bidder wins
and pays one more than the second-highest bid (up to its own bid), which
efficiently reproduces the outcome of an ascending English auction. The game
reserves enough cash for every bot to buy its remaining roster slots for at
least $1.

## Visual dashboard

The easiest way to explore the simulation is the local Streamlit dashboard:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[ui,dev]'
streamlit run streamlit_app.py
```

Streamlit prints a local address, normally
[`http://localhost:8501`](http://localhost:8501), and usually opens it in your
browser automatically. Stop it with `Ctrl+C` in the terminal.

The dashboard's primary mode is a head-to-head blind auction:

- you versus a bot that bids a random percentage of its remaining budget;
- exactly 20 sequentially revealed items rated from 1–100;
- $500 starting budgets and no mandated roster split;
- turn-by-turn ascending bids: raise the visible price or pass;
- live budgets, rosters, scores, and completed auction history;

Default pools are sampled without replacement: all 20 ratings are unique within
a draft. Custom engine pools also require unique item IDs.

## RL environment

Install the environment dependencies:

```bash
pip install -e '.[rl,dev]'
```

`item_auction.core.AuctionEngine` is the canonical turn-based engine used by
both the Streamlit game and all RL wrappers. `AuctionAECEnv` exposes that same
engine through PettingZoo, while `FixedOpponentEnv` is a Gymnasium adapter that
automatically plays opponent turns, making it suitable for a single learning
policy.

The observation contains the current rating, current bid, both remaining
budgets, both scores and roster sizes, items remaining, leader state, recent
prices, and a legal-action mask. It never contains future item values.

The 13 actions include pass, minimum raise, percentage targets based on the
acting bot's remaining budget, percentage targets based on the opponent's
remaining budget, and an opponent-all-in-plus-$1 target.

Five baseline opponents are included:

- `RatingNoisePolicy`: bids up to rating ±10%;
- `BudgetProportionPolicy`: bids up to rating percent of remaining budget ±10%;
- `AggressiveHighValuePolicy`: spends heavily above rating 70;
- `RandomPassPolicy`: randomly passes or raises;
- `DealProbabilityPolicy`: raise probability increases with rating-to-price value.

`OpponentPool` stores factories rather than live objects, so every episode gets
fresh policy state. Add a trained checkpoint later through `CallablePolicy`:

```python
pool.add(
    "ppo-v1",
    lambda: CallablePolicy(
        "ppo-v1",
        lambda observation, mask: my_model_predict(observation, mask),
    ),
)
```

See [`examples/rl_environment.py`](examples/rl_environment.py) for a complete
untrained episode. No RL model or training loop is included yet.

## Command-line quick start

No runtime dependencies are required:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

item-auction simulate --seed 42
item-auction train --generations 20 --population 24 --episodes 30
```

Run the tests with:

```bash
pip install -e '.[dev]'
pytest
```

## Environment API

```python
from item_auction import AuctionConfig, SequentialAuctionEnv

env = SequentialAuctionEnv(
    ["bot-a", "bot-b", "bot-c", "bot-d"],
    AuctionConfig(budget=250, roster_size=5, pool_size=30),
)
observations = env.reset(seed=42)

while not env.done:
    bids = {
        name: my_policy(observation)
        for name, observation in observations.items()
    }
    observations, auction, done = env.step(bids)

result = env.result()
```

An observation includes only:

- the currently revealed item and its visible value;
- the bot's own cash and roster;
- opponents' public cash, roster size, and score;
- completed auction history;
- legal bid and game-progress information.

It never includes the remaining item sequence.

## Included bots and training

- `ValueBot`: a hand-built baseline balancing item quality, budget pace, and
  late-game urgency.
- `RandomBot`: a reproducible weak baseline.
- `LinearBot`: a small serializable policy using observation-only features.
- `train_policy`: dependency-free evolutionary training against three different
  `ValueBot` personalities.

Training writes a JSON policy that can be loaded with
`item_auction.training.load_policy`. The environment's explicit
`reset`/`step` interface also makes it straightforward to wrap for Gymnasium or
plug into PPO/DQN later.

## Custom items, categories, and uncertainty

Pass your own list of `Item` objects to `SequentialAuctionEnv` or `play_game`.
See [`examples/custom_items.py`](examples/custom_items.py).

```python
Item(
    id="mystery-box",
    name="Mystery Box",
    value=91,       # final score; hidden while this item is being bid on
    estimate=70,    # value shown to bots during the auction
    category="tech",
)
```

`category` is already part of every item. `estimate` can differ from the true
`value`; the current observation contains a safe `RevealedItem` without the
true value, so uncertainty experiments do not require an environment rewrite.
The true value is considered realized once the completed auction enters the
public history.
Future category scoring can be added by replacing the score aggregation in
`SequentialAuctionEnv.result()` or by computing a custom reward from the
returned rosters.
