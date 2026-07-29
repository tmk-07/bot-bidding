from __future__ import annotations

import argparse
from pathlib import Path

from .agents import LinearBot, ValueBot
from .simulation import play_game
from .training import save_policy, train_policy


def _print_game(seed: int) -> None:
    bots = [
        LinearBot("Linear"),
        ValueBot("Steady", 0.95, 0.35),
        ValueBot("Patient", 0.82, 0.48),
        ValueBot("Aggressive", 1.15, 0.27),
    ]
    result = play_game(bots, seed=seed)
    print(f"Sequential item auction (seed {seed})")
    print("-" * 72)
    for auction in result.auctions:
        if auction.winner:
            print(
                f"{auction.item.name:>8} value={auction.item.value:>3.0f}  "
                f"won by {auction.winner:<10} for ${auction.price}"
            )
    print("-" * 72)
    for name in sorted(result.scores, key=result.scores.get, reverse=True):
        roster = ", ".join(
            f"{entry.item.value:.0f}(${entry.price})"
            for entry in result.rosters[name]
        )
        print(
            f"{name:<10} score={result.scores[name]:>5.0f} "
            f"cash=${result.budgets[name]:>3}  [{roster}]"
        )
    print(f"Winner: {', '.join(result.winners)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainable sequential item auction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate", help="Run and print one game")
    simulate.add_argument("--seed", type=int, default=42)
    train = subparsers.add_parser("train", help="Train a linear bidding bot")
    train.add_argument("--generations", type=int, default=20)
    train.add_argument("--population", type=int, default=24)
    train.add_argument("--episodes", type=int, default=30)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--output", type=Path, default=Path("trained_policy.json"))
    args = parser.parse_args()

    if args.command == "simulate":
        _print_game(args.seed)
    else:
        policy, history = train_policy(
            generations=args.generations,
            population=args.population,
            episodes=args.episodes,
            seed=args.seed,
        )
        save_policy(policy, args.output)
        print(f"Best fitness by generation: {', '.join(f'{x:.3f}' for x in history)}")
        print(f"Saved trained policy to {args.output}")


if __name__ == "__main__":
    main()
