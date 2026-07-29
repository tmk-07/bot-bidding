from item_auction import AuctionConfig, Item, LinearBot, ValueBot
from item_auction.simulation import play_game


items = [
    Item(
        id=f"tool-{i}",
        name=f"Tool {i}",
        value=(i * 37) % 100 + 1,
        category=("power" if i % 2 else "speed"),
    )
    for i in range(24)
]

result = play_game(
    [
        LinearBot("learner"),
        ValueBot("balanced"),
        ValueBot("patient", aggressiveness=0.85, value_threshold=0.50),
        ValueBot("bold", aggressiveness=1.15, value_threshold=0.25),
    ],
    config=AuctionConfig(pool_size=len(items)),
    items=items,
    seed=42,
)

print(result.scores)
