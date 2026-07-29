from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from ..core import AuctionEngine
from .actions import ACTION_COUNT, BidAction, action_to_bid, legal_action_mask
from .opponents import OpponentPolicy, PolicyObservation


AGENTS = ("player_0", "player_1")
OBSERVATION_SIZE = 12


class AuctionAECEnv(AECEnv):
    """PettingZoo wrapper around the canonical AuctionEngine."""

    metadata = {
        "name": "item_auction_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        *,
        budget: int = 500,
        pool_size: int = 20,
        value_min: int = 1,
        value_max: int = 100,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.possible_agents = list(AGENTS)
        self.budget_start = budget
        self.pool_size = pool_size
        self.value_min = value_min
        self.value_max = value_max
        self.render_mode = render_mode
        self.engine = AuctionEngine(
            AGENTS,
            budget=budget,
            pool_size=pool_size,
            value_min=value_min,
            value_max=value_max,
        )
        self._action_spaces = {
            agent: spaces.Discrete(ACTION_COUNT) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(OBSERVATION_SIZE,),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.MultiBinary(ACTION_COUNT),
                }
            )
            for agent in self.possible_agents
        }
        self.winner: str | None = None

    @property
    def budgets(self) -> dict[str, int]:
        return self.engine.budgets

    @property
    def scores(self) -> dict[str, float]:
        return self.engine.scores

    @property
    def rosters(self):
        return self.engine.rosters

    @property
    def item_counts(self) -> dict[str, int]:
        return {agent: len(roster) for agent, roster in self.engine.rosters.items()}

    @property
    def current_bid(self) -> int:
        return self.engine.current_bid

    @property
    def leader(self) -> str | None:
        return self.engine.leader

    @property
    def _item_index(self) -> int:
        return self.engine.item_index

    @property
    def _first_pass(self) -> str | None:
        return self.engine.first_pass

    @property
    def auction_history(self) -> list[dict[str, Any]]:
        return [
            {
                "rating": auction.item.value,
                "winner": auction.winner,
                "price": auction.price,
            }
            for auction in self.engine.history
        ]

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    @staticmethod
    def other(agent: str) -> str:
        return AGENTS[1] if agent == AGENTS[0] else AGENTS[0]

    @property
    def current_rating(self) -> int:
        item = self.engine.current_item
        if item is None:
            return 0
        return int(item.value)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.agents = list(self.possible_agents)
        self.engine.reset(seed)
        self.agent_selection = self.engine.agent_selection
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.winner = None

    def _policy_observation(self, agent: str) -> PolicyObservation:
        opponent = self.other(agent)
        return PolicyObservation(
            item_rating=float(self.current_rating),
            current_bid=self.current_bid,
            own_budget=self.budgets[agent],
            opponent_budget=self.budgets[opponent],
            own_score=self.scores[agent],
            opponent_score=self.scores[opponent],
            own_items=self.item_counts[agent],
            opponent_items=self.item_counts[opponent],
            items_remaining=self.engine.items_remaining,
            is_opening_turn=(
                self.leader is None and agent == self.engine.current_opener
            ),
        )

    def policy_observation(self, agent: str) -> PolicyObservation:
        return self._policy_observation(agent)

    def action_mask(self, agent: str) -> np.ndarray:
        opponent = self.other(agent)
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            return np.zeros(ACTION_COUNT, dtype=np.int8)
        return legal_action_mask(
            current_bid=self.current_bid,
            own_budget=self.budgets[agent],
            opponent_budget=self.budgets[opponent],
        )

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        if self.engine.done:
            return {
                "observation": np.zeros(OBSERVATION_SIZE, dtype=np.float32),
                "action_mask": np.zeros(ACTION_COUNT, dtype=np.int8),
            }
        opponent = self.other(agent)
        max_score = self.pool_size * self.value_max
        recent = self.engine.history[-5:]
        recent_mean = (
            sum(auction.price for auction in recent) / len(recent) if recent else 0.0
        )
        values = np.asarray(
            [
                self.current_rating / self.value_max,
                self.current_bid / self.budget_start,
                self.budgets[agent] / self.budget_start,
                self.budgets[opponent] / self.budget_start,
                self.scores[agent] / max_score,
                self.scores[opponent] / max_score,
                self.item_counts[agent] / self.pool_size,
                self.item_counts[opponent] / self.pool_size,
                self.engine.items_remaining / self.pool_size,
                float(self.leader == agent),
                float(self.leader is not None),
                recent_mean / self.budget_start,
            ],
            dtype=np.float32,
        )
        return {"observation": values, "action_mask": self.action_mask(agent)}

    def step(self, action: int) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        mask = self.action_mask(agent)
        action = int(action)
        if not 0 <= action < ACTION_COUNT or not mask[action]:
            raise ValueError(f"Illegal action {action} for {agent}")

        if action == BidAction.PASS:
            transition = self.engine.pass_turn(agent)
        else:
            opponent = self.other(agent)
            amount = action_to_bid(
                action,
                current_bid=self.current_bid,
                own_budget=self.budgets[agent],
                opponent_budget=self.budgets[opponent],
            )
            if amount is None:
                raise ValueError("Raise action resolved to pass")
            transition = self.engine.place_bid(agent, amount)

        if transition.resolved and transition.auction.winner:
            item_winner = transition.auction.winner
            item_loser = self.other(item_winner)
            shaped = transition.auction.item.value / (
                self.pool_size * self.value_max
            )
            self.rewards[item_winner] += shaped
            self.rewards[item_loser] -= shaped

        if self.engine.done:
            self.winner = self.engine.winner
            if self.winner:
                loser = self.other(self.winner)
                self.rewards[self.winner] += 1.0
                self.rewards[loser] -= 1.0
            for player in self.agents:
                self.terminations[player] = True
            self.agent_selection = agent
        else:
            self.agent_selection = self.engine.agent_selection
        self._accumulate_rewards()

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (
            f"item={min(self.engine.item_index + 1, self.pool_size)}/{self.pool_size} "
            f"rating={self.current_rating or '-'} bid=${self.current_bid} "
            f"leader={self.leader} budgets={self.budgets} scores={self.scores}"
        )

    def close(self) -> None:
        pass


class FixedOpponentEnv(gym.Env):
    """Single-learner Gymnasium adapter over the canonical engine."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_factory: Callable[[], OpponentPolicy],
        *,
        learner_agent: str = AGENTS[0],
        randomize_seat: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.opponent_factory = opponent_factory
        self.learner_agent = learner_agent
        self.randomize_seat = randomize_seat
        self.render_mode = render_mode
        self.aec = AuctionAECEnv(render_mode=render_mode)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = self.aec.observation_space(AGENTS[0])
        self._rng = random.Random()
        self.opponent: OpponentPolicy | None = None
        self._opponent_item_index = -1

    @property
    def opponent_agent(self) -> str:
        return self.aec.other(self.learner_agent)

    def _play_opponent_turns(self) -> float:
        reward = 0.0
        while (
            not self.aec.terminations[self.learner_agent]
            and self.aec.agent_selection != self.learner_agent
        ):
            agent = self.aec.agent_selection
            observation = self.aec.policy_observation(agent)
            if self._opponent_item_index != self.aec._item_index:
                self.opponent.start_auction(observation)
                self._opponent_item_index = self.aec._item_index
            action = self.opponent.act(observation, self.aec.action_mask(agent))
            self.aec.step(action)
            reward += self.aec.rewards[self.learner_agent]
        return reward

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self._rng = random.Random(seed)
        self.learner_agent = (
            self._rng.choice(AGENTS) if self.randomize_seat else self.learner_agent
        )
        self.opponent = self.opponent_factory()
        self.opponent.reset(None if seed is None else seed + 1)
        self._opponent_item_index = -1
        self.aec.reset(seed=seed)
        self._play_opponent_turns()
        return self.aec.observe(self.learner_agent), {
            "opponent": self.opponent.name,
            "learner_agent": self.learner_agent,
        }

    def step(self, action: int):
        self.aec.step(int(action))
        reward = self.aec.rewards[self.learner_agent]
        reward += self._play_opponent_turns()
        terminated = self.aec.terminations[self.learner_agent]
        observation = self.aec.observe(self.learner_agent)
        info = {
            "winner": self.aec.winner,
            "scores": dict(self.aec.scores),
            "budgets": dict(self.aec.budgets),
            "opponent": self.opponent.name,
        }
        return observation, reward, terminated, False, info

    def render(self):
        return self.aec.render()
