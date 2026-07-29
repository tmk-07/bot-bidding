from __future__ import annotations

import math
from enum import IntEnum

import numpy as np


class BidAction(IntEnum):
    """Budget-aware discrete actions available to every trained policy."""

    PASS = 0
    MIN_RAISE = 1
    OWN_10 = 2
    OWN_25 = 3
    OWN_50 = 4
    OWN_75 = 5
    OWN_ALL_IN = 6
    OPP_10 = 7
    OPP_25 = 8
    OPP_50 = 9
    OPP_75 = 10
    OPP_100 = 11
    OPP_ALL_IN_PLUS_ONE = 12


ACTION_COUNT = len(BidAction)

_OWN_FRACTIONS = {
    BidAction.OWN_10: 0.10,
    BidAction.OWN_25: 0.25,
    BidAction.OWN_50: 0.50,
    BidAction.OWN_75: 0.75,
    BidAction.OWN_ALL_IN: 1.00,
}
_OPP_FRACTIONS = {
    BidAction.OPP_10: 0.10,
    BidAction.OPP_25: 0.25,
    BidAction.OPP_50: 0.50,
    BidAction.OPP_75: 0.75,
    BidAction.OPP_100: 1.00,
}


def action_to_bid(
    action: int | BidAction,
    *,
    current_bid: int,
    own_budget: int,
    opponent_budget: int,
) -> int | None:
    """Map a semantic action to a concrete total bid.

    Percentage actions target a percentage of a player's *remaining total
    budget*, rather than adding a percentage to the current price. Returning
    ``None`` means pass. An amount at or below the current bid is illegal and
    is masked by :func:`legal_action_mask`.
    """

    selected = BidAction(int(action))
    if selected == BidAction.PASS:
        return None
    if selected == BidAction.MIN_RAISE:
        return current_bid + 1
    if selected in _OWN_FRACTIONS:
        return min(own_budget, math.ceil(own_budget * _OWN_FRACTIONS[selected]))
    if selected in _OPP_FRACTIONS:
        target = math.ceil(opponent_budget * _OPP_FRACTIONS[selected])
        return min(own_budget, target)
    if selected == BidAction.OPP_ALL_IN_PLUS_ONE:
        return min(own_budget, opponent_budget + 1)
    raise ValueError(f"Unsupported action: {selected}")


def legal_action_mask(
    *,
    current_bid: int,
    own_budget: int,
    opponent_budget: int,
) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[BidAction.PASS] = 1
    for action in list(BidAction)[1:]:
        amount = action_to_bid(
            action,
            current_bid=current_bid,
            own_budget=own_budget,
            opponent_budget=opponent_budget,
        )
        if amount is not None and current_bid < amount <= own_budget:
            mask[action] = 1
    return mask
