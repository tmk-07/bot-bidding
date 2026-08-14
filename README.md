# Bot Bidding — Auction Strategy Lab

A sequential auction game and reinforcement-learning environment for building, training, and comparing bidding strategies.

**Repository:** [github.com/tmk-07/bot-bidding](https://github.com/tmk-07/bot-bidding)

## Overview

I built this project to explore how different strategies perform in a sequential auction where future items are unknown.

How the game works  Two players receive a $500 budget and bid on 20 items. Each item has a rating from 1–100, and items are revealed one at a time without replacement. Players can see the current item, bids, budgets, scores, and auction history—but cannot see upcoming items. The team with the highest sum of item ratings is the winner.

The project includes:

- A playable Streamlit game
- A shared auction engine
- Scripted bidding strategies
- Reinforcement-learning environments
- PPO training and evaluation
- Frozen checkpoint opponents
- Historical performance and behavior analysis

## Features

- Play against multiple trained and scripted bots
- 20 sequentially revealed items
- Unique ratings sampled without replacement
- $500 budget for each player
- Turn-by-turn ascending bidding
- Hidden future auction pool
- Shared engine for gameplay and training
- Gymnasium and PettingZoo environments
- MaskablePPO reinforcement-learning support
- Fixed evaluation leagues
- Best-checkpoint promotion
- SQLite training-history storage
- Win-rate, score, budget, roster, pricing, and action analysis
- Rating-band price comparisons
- Individual auction and selection history

## Technical Architecture

The project uses one canonical `AuctionEngine` for both the playable Streamlit game and RL simulations. This keeps bidding rules, budgets, scoring, item generation, and auction resolution consistent across every part of the project.

The RL system wraps the engine with:

- A PettingZoo AEC environment for turn-based multi-agent interaction
- A Gymnasium environment for training one learner against a fixed opponent
- Legal-action masks for budget-aware bidding
- Configurable scripted and frozen-checkpoint opponents
- MaskablePPO for reinforcement-learning experiments

Training runs are evaluated against fixed held-out opponent leagues. Every evaluation stores results in SQLite, including player scores, budgets, drafted items, final prices, offers, winners, and learner actions.

## Technology

- Python
- Streamlit
- Gymnasium
- PettingZoo
- Stable-Baselines3
- SB3 Contrib MaskablePPO
- NumPy
- Pandas
- Altair
- SQLite
- Pytest
- GitHub

## Project Structure

```text
.
├── data/
│   └── training_history.sqlite3    # Local training and evaluation history
├── examples/
│   └── rl_environment.py           # Example RL environment usage
├── models/                         # Trained checkpoints and active models
├── src/
│   └── item_auction/
│       ├── core.py                 # Canonical auction engine
│       ├── duel.py                 # Human-versus-bot game controller
│       ├── environment.py          # General simulation environment
│       ├── models.py               # Shared auction data models
│       └── rl/
│           ├── actions.py          # Budget-aware RL action definitions
│           ├── env.py              # PettingZoo and Gymnasium wrappers
│           ├── opponents.py        # Scripted and trained bot adapters
│           ├── baseline_training.py
│           ├── deal_value_training.py
│           └── history.py          # SQLite evaluation storage
├── tests/                          # Engine, controller, and RL tests
├── streamlit_app.py                # Game and analytics dashboard
├── pyproject.toml                  # Package and dependency configuration
└── README.md                       # Project documentation
```

## Running the App

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[ui,train,dev]'
```

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Open the local address printed by Streamlit, normally:

```text
http://localhost:8501
```

Stop the app with `Ctrl+C`.

## Current Bots

### Market Generalist v1

Uses the visible rating as a fair-price anchor and makes small adjustments based on:

- Score difference
- Budget difference
- Auctions remaining
- Current market price

It applies the same strategy to every opponent and does not know which strategy it is facing.

### Iterated Bot v5

A MaskablePPO policy trained through several opponent leagues and frozen-checkpoint curricula. It uses the complete public game state to select among legal budget-aware actions.

### Deal-Value Bot v2

A MaskablePPO policy originally trained around price discovery. It evaluates the current deal relative to item value, then adjusts for budgets, scores, rosters, and remaining auctions.

### Scripted Strategies

The environment also includes:

- Rating Exact
- Rating Noise
- Budget Proportion
- Aggressive High Value
- Deal Probability
- Random Pass
- Market Pressure

## Training a Bot

Install the training dependencies:

```bash
pip install -e '.[train,dev]'
```

Run baseline PPO training:

```bash
item-auction-train-baseline \
  --timesteps 100000 \
  --eval-interval 25000 \
  --eval-episodes 100 \
  --environments 8
```

Train against a custom opponent curriculum:

```bash
item-auction-train-baseline \
  --opponents market-pressure rating-exact rating-noise-20 \
  --opponent-weight market-pressure=0.50 \
  --opponent-weight rating-exact=0.30 \
  --opponent-weight rating-noise-20=0.20 \
  --timesteps 300000
```

A saved PPO model can also be included as a frozen opponent:

```bash
item-auction-train-baseline \
  --start-model models/active-iterated.zip \
  --frozen-model models/active-iterated.zip \
  --frozen-name "Frozen Iterated Bot"
```

## Evaluation and Promotion

Every training run evaluates the unchanged starting policy at step zero and then evaluates periodic checkpoints against a fixed held-out league.

Tracked metrics include:

- Win credit
- Average total score
- Average score margin
- Items drafted
- Budget used
- Final auction prices
- Learner and opponent offers
- Rating-band ownership
- Action distribution
- Individual selections

The final checkpoint is not promoted automatically. The best checkpoint must outperform the current active model without creating unacceptable regressions against important opponents.

Superseded runs remain available for historical analysis.

## Testing

Run the complete test suite with:

```bash
pytest
```

The tests cover:

- Auction rules
- Budget accounting
- Item uniqueness
- Human-versus-bot interactions
- Legal action masks
- Scripted policies
- RL environment compliance
- Training-history storage
- Checkpoint promotion
- Policy evaluation

## GitHub Workflow

Typical update workflow:

```bash
git add .
git commit -m "Update auction environment"
git push
```

Training databases and model checkpoints are stored locally by default and excluded from Git because they can become large.
