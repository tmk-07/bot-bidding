from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from ..core import AuctionEngine


SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    learner_family TEXT NOT NULL DEFAULT 'iterated',
    total_timesteps INTEGER NOT NULL,
    selected_checkpoint_steps INTEGER,
    seed INTEGER NOT NULL,
    model_path TEXT,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES training_runs(id),
    checkpoint_steps INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    opponent_name TEXT NOT NULL,
    learner_agent TEXT NOT NULL,
    winner_role TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player_results (
    game_id INTEGER NOT NULL REFERENCES evaluation_games(id),
    player_role TEXT NOT NULL,
    player_name TEXT NOT NULL,
    score REAL NOT NULL,
    items_drafted INTEGER NOT NULL,
    budget_start INTEGER NOT NULL,
    budget_used INTEGER NOT NULL,
    budget_remaining INTEGER NOT NULL,
    PRIMARY KEY (game_id, player_role)
);

CREATE TABLE IF NOT EXISTS selections (
    game_id INTEGER NOT NULL REFERENCES evaluation_games(id),
    auction_number INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    item_name TEXT NOT NULL,
    rating REAL NOT NULL,
    rating_band TEXT NOT NULL,
    final_bid INTEGER NOT NULL,
    winner_role TEXT NOT NULL,
    winner_name TEXT NOT NULL,
    learner_final_offer INTEGER NOT NULL,
    opponent_final_offer INTEGER NOT NULL,
    PRIMARY KEY (game_id, auction_number)
);

CREATE TABLE IF NOT EXISTS learner_actions (
    game_id INTEGER NOT NULL REFERENCES evaluation_games(id),
    decision_number INTEGER NOT NULL,
    auction_number INTEGER NOT NULL,
    rating REAL NOT NULL,
    current_bid INTEGER NOT NULL,
    own_budget INTEGER NOT NULL,
    opponent_budget INTEGER NOT NULL,
    action_index INTEGER NOT NULL,
    action_name TEXT NOT NULL,
    target_bid INTEGER,
    PRIMARY KEY (game_id, decision_number)
);

CREATE TABLE IF NOT EXISTS training_exposure (
    run_id TEXT NOT NULL REFERENCES training_runs(id),
    opponent_name TEXT NOT NULL,
    episodes INTEGER NOT NULL,
    PRIMARY KEY (run_id, opponent_name)
);

CREATE INDEX IF NOT EXISTS idx_games_run_checkpoint
    ON evaluation_games(run_id, checkpoint_steps);
CREATE INDEX IF NOT EXISTS idx_games_opponent
    ON evaluation_games(opponent_name);
CREATE INDEX IF NOT EXISTS idx_selections_game
    ON selections(game_id);
CREATE INDEX IF NOT EXISTS idx_actions_game
    ON learner_actions(game_id);
"""


def rating_band(rating: float) -> str:
    """Return a stable ten-point label, including the rating of 100."""

    value = max(1, min(100, int(rating)))
    if value == 100:
        return "100"
    lower = (value // 10) * 10
    if value < 10:
        return "1-9"
    return f"{lower}-{lower + 9}"


class TrainingHistory:
    """SQLite-backed evaluation history for training checkpoints."""

    def __init__(self, path: str | Path = "data/training_history.sqlite3") -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(training_runs)"
                ).fetchall()
            }
            if "learner_family" not in columns:
                connection.execute(
                    """
                    ALTER TABLE training_runs
                    ADD COLUMN learner_family TEXT NOT NULL DEFAULT 'iterated'
                    """
                )
            if "selected_checkpoint_steps" not in columns:
                connection.execute(
                    """
                    ALTER TABLE training_runs
                    ADD COLUMN selected_checkpoint_steps INTEGER
                    """
                )
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create_run(
        self,
        *,
        algorithm: str,
        total_timesteps: int,
        seed: int,
        config: dict[str, Any],
        learner_family: str = "iterated",
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO training_runs (
                    id, created_at, status, algorithm, learner_family,
                    total_timesteps, seed, config_json
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(UTC).isoformat(),
                    algorithm,
                    learner_family,
                    total_timesteps,
                    seed,
                    json.dumps(config, sort_keys=True),
                ),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        model_path: str | None = None,
        selected_checkpoint_steps: int | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE training_runs
                SET completed_at = ?, status = ?, model_path = ?,
                    selected_checkpoint_steps = COALESCE(
                        ?, selected_checkpoint_steps
                    )
                WHERE id = ?
                """,
                (
                    datetime.now(UTC).isoformat(),
                    status,
                    model_path,
                    selected_checkpoint_steps,
                    run_id,
                ),
            )

    def select_checkpoint(
        self,
        run_id: str,
        checkpoint_steps: int,
        *,
        model_path: str,
    ) -> None:
        """Mark the checkpoint promoted from a completed training run."""

        with self.connect() as connection:
            exists = connection.execute(
                """
                SELECT 1 FROM evaluation_games
                WHERE run_id = ? AND checkpoint_steps = ?
                """,
                (run_id, checkpoint_steps),
            ).fetchone()
            if not exists:
                raise ValueError(
                    f"Checkpoint {checkpoint_steps} is not recorded for {run_id}"
                )
            connection.execute(
                """
                UPDATE training_runs
                SET selected_checkpoint_steps = ?, model_path = ?
                WHERE id = ?
                """,
                (checkpoint_steps, model_path, run_id),
            )

    def record_training_exposure(
        self,
        run_id: str,
        episodes_by_opponent: dict[str, int],
    ) -> None:
        with self.connect() as connection:
            for opponent_name, episodes in episodes_by_opponent.items():
                connection.execute(
                    """
                    INSERT INTO training_exposure (
                        run_id, opponent_name, episodes
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(run_id, opponent_name)
                    DO UPDATE SET episodes = excluded.episodes
                    """,
                    (run_id, opponent_name, int(episodes)),
                )

    def training_exposure(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT opponent_name AS opponent,
                       episodes AS training_matches,
                       ROUND(
                           100.0 * episodes /
                           SUM(episodes) OVER (),
                           2
                       ) AS actual_share_pct
                FROM training_exposure
                WHERE run_id = ?
                ORDER BY episodes DESC, opponent_name
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def truncate_run(
        self,
        run_id: str,
        checkpoint_steps: int,
        *,
        model_path: str | None = None,
    ) -> int:
        """Remove evaluation history after a selected restored checkpoint.

        Whole-run training exposure is also removed because it cannot be
        accurately divided by checkpoint after training has completed.
        Returns the number of evaluation games removed.
        """

        with self.connect() as connection:
            exists = connection.execute(
                """
                SELECT 1
                FROM evaluation_games
                WHERE run_id = ? AND checkpoint_steps = ?
                """,
                (run_id, checkpoint_steps),
            ).fetchone()
            if not exists:
                raise ValueError(
                    f"Checkpoint {checkpoint_steps} is not recorded for {run_id}"
                )
            game_filter = """
                SELECT id FROM evaluation_games
                WHERE run_id = ? AND checkpoint_steps > ?
            """
            removed = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM ({game_filter})",
                    (run_id, checkpoint_steps),
                ).fetchone()[0]
            )
            parameters = (run_id, checkpoint_steps)
            for table in ("learner_actions", "selections", "player_results"):
                connection.execute(
                    f"DELETE FROM {table} WHERE game_id IN ({game_filter})",
                    parameters,
                )
            connection.execute(
                """
                DELETE FROM evaluation_games
                WHERE run_id = ? AND checkpoint_steps > ?
                """,
                parameters,
            )
            connection.execute(
                "DELETE FROM training_exposure WHERE run_id = ?",
                (run_id,),
            )
            connection.execute(
                """
                UPDATE training_runs
                SET total_timesteps = ?,
                    status = 'truncated',
                    model_path = COALESCE(?, model_path),
                    selected_checkpoint_steps = ?
                WHERE id = ?
                """,
                (checkpoint_steps, model_path, checkpoint_steps, run_id),
            )
        return removed

    def record_game(
        self,
        *,
        run_id: str,
        checkpoint_steps: int,
        episode: int,
        seed: int,
        opponent_name: str,
        learner_agent: str,
        engine: AuctionEngine,
        learner_actions: list[dict[str, Any]] | None = None,
    ) -> int:
        if not engine.done:
            raise ValueError("Only completed games can be recorded")
        opponent_agent = engine.other(learner_agent)
        if engine.winner == learner_agent:
            winner_role = "learner"
        elif engine.winner == opponent_agent:
            winner_role = "opponent"
        else:
            winner_role = "tie"

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO evaluation_games (
                    run_id, checkpoint_steps, episode, seed, opponent_name,
                    learner_agent, winner_role
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    checkpoint_steps,
                    episode,
                    seed,
                    opponent_name,
                    learner_agent,
                    winner_role,
                ),
            )
            game_id = int(cursor.lastrowid)
            for role, agent, name in (
                ("learner", learner_agent, "RL Bot"),
                ("opponent", opponent_agent, opponent_name),
            ):
                remaining = engine.budgets[agent]
                connection.execute(
                    """
                    INSERT INTO player_results (
                        game_id, player_role, player_name, score,
                        items_drafted, budget_start, budget_used,
                        budget_remaining
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        game_id,
                        role,
                        name,
                        engine.scores[agent],
                        len(engine.rosters[agent]),
                        engine.budget_start,
                        engine.budget_start - remaining,
                        remaining,
                    ),
                )

            for auction_number, auction in enumerate(engine.history, start=1):
                if auction.winner == learner_agent:
                    selection_winner_role = "learner"
                    selection_winner_name = "RL Bot"
                elif auction.winner == opponent_agent:
                    selection_winner_role = "opponent"
                    selection_winner_name = opponent_name
                else:
                    selection_winner_role = "unsold"
                    selection_winner_name = "Unsold"
                connection.execute(
                    """
                    INSERT INTO selections (
                        game_id, auction_number, item_id, item_name, rating,
                        rating_band, final_bid, winner_role, winner_name,
                        learner_final_offer, opponent_final_offer
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        game_id,
                        auction_number,
                        auction.item.id,
                        auction.item.name,
                        auction.item.value,
                        rating_band(auction.item.value),
                        auction.price,
                        selection_winner_role,
                        selection_winner_name,
                        auction.bids[learner_agent],
                        auction.bids[opponent_agent],
                    ),
                )
            for decision_number, action in enumerate(
                learner_actions or [], start=1
            ):
                connection.execute(
                    """
                    INSERT INTO learner_actions (
                        game_id, decision_number, auction_number, rating,
                        current_bid, own_budget, opponent_budget, action_index,
                        action_name, target_bid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        game_id,
                        decision_number,
                        action["auction_number"],
                        action["rating"],
                        action["current_bid"],
                        action["own_budget"],
                        action["opponent_budget"],
                        action["action_index"],
                        action["action_name"],
                        action["target_bid"],
                    ),
                )
        return game_id

    def runs(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*,
                       COUNT(DISTINCT g.id) AS evaluation_games,
                       MAX(g.checkpoint_steps) AS latest_checkpoint
                FROM training_runs r
                LEFT JOIN evaluation_games g ON g.run_id = r.id
                GROUP BY r.id
                ORDER BY r.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def checkpoints(self, run_id: str) -> list[int]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT checkpoint_steps
                FROM evaluation_games
                WHERE run_id = ?
                ORDER BY checkpoint_steps
                """,
                (run_id,),
            ).fetchall()
        return [int(row["checkpoint_steps"]) for row in rows]

    def player_summary(
        self, run_id: str, checkpoint_steps: int
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT g.opponent_name AS opponent,
                       p.player_role AS role,
                       p.player_name AS player,
                       COUNT(*) AS games,
                       ROUND(AVG(p.score), 2) AS avg_score,
                       ROUND(AVG(p.items_drafted), 2) AS avg_items,
                       ROUND(AVG(p.budget_used), 2) AS avg_budget_used,
                       ROUND(AVG(p.budget_remaining), 2) AS avg_budget_remaining,
                       ROUND(100.0 * AVG(
                           CASE
                               WHEN g.winner_role = p.player_role THEN 1.0
                               WHEN g.winner_role = 'tie' THEN 0.5
                               ELSE 0.0
                           END
                       ), 1) AS win_credit_pct
                FROM evaluation_games g
                JOIN player_results p ON p.game_id = g.id
                WHERE g.run_id = ? AND g.checkpoint_steps = ?
                GROUP BY g.opponent_name, p.player_role, p.player_name
                ORDER BY g.opponent_name, p.player_role
                """,
                (run_id, checkpoint_steps),
            ).fetchall()
        return [dict(row) for row in rows]

    def rating_summary(
        self, run_id: str, checkpoint_steps: int
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT g.opponent_name AS opponent,
                       s.rating_band,
                       COUNT(*) AS items,
                       ROUND(AVG(s.final_bid), 2) AS avg_final_bid,
                       SUM(s.winner_role = 'learner') AS learner_won,
                       SUM(s.winner_role = 'opponent') AS opponent_won,
                       SUM(s.winner_role = 'unsold') AS unsold
                FROM selections s
                JOIN evaluation_games g ON g.id = s.game_id
                WHERE g.run_id = ? AND g.checkpoint_steps = ?
                GROUP BY g.opponent_name, s.rating_band
                ORDER BY g.opponent_name,
                    CASE s.rating_band
                        WHEN '100' THEN 100
                        ELSE CAST(substr(s.rating_band, 1, instr(s.rating_band, '-') - 1) AS INTEGER)
                    END DESC
                """,
                (run_id, checkpoint_steps),
            ).fetchall()
        return [dict(row) for row in rows]

    def pricing_behavior(
        self,
        run_id: str,
        checkpoint_steps: int,
    ) -> list[dict[str, Any]]:
        """Aggregate auction prices and learner offers into ten-point bands."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT CASE
                           WHEN s.rating >= 90 THEN '90-100'
                           WHEN s.rating < 10 THEN '1-9'
                           ELSE printf(
                               '%d-%d',
                               CAST(s.rating / 10 AS INTEGER) * 10,
                               CAST(s.rating / 10 AS INTEGER) * 10 + 9
                           )
                       END AS rating_band,
                       CASE
                           WHEN s.rating >= 90 THEN 90
                           WHEN s.rating < 10 THEN 1
                           ELSE CAST(s.rating / 10 AS INTEGER) * 10
                       END AS band_start,
                       COUNT(*) AS items,
                       ROUND(AVG(s.final_bid), 2) AS avg_final_price,
                       ROUND(AVG(s.learner_final_offer), 2)
                           AS avg_learner_offer,
                       ROUND(AVG(
                           CASE WHEN s.winner_role = 'learner'
                                THEN s.final_bid END
                       ), 2) AS avg_price_when_won,
                       ROUND(100.0 * AVG(
                           CASE WHEN s.winner_role = 'learner'
                                THEN 1.0 ELSE 0.0 END
                       ), 1) AS learner_win_pct
                FROM selections s
                JOIN evaluation_games g ON g.id = s.game_id
                WHERE g.run_id = ? AND g.checkpoint_steps = ?
                GROUP BY rating_band, band_start
                ORDER BY band_start
                """,
                (run_id, checkpoint_steps),
            ).fetchall()
        return [dict(row) for row in rows]

    def selections(
        self,
        run_id: str,
        checkpoint_steps: int,
        *,
        opponent: str | None = None,
        limit: int = 5_000,
    ) -> list[dict[str, Any]]:
        where = """
            WHERE g.run_id = ? AND g.checkpoint_steps = ?
        """
        parameters: list[Any] = [run_id, checkpoint_steps]
        if opponent:
            where += " AND g.opponent_name = ?"
            parameters.append(opponent)
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT g.episode, g.seed, g.opponent_name AS opponent,
                       s.auction_number, s.item_name, s.rating, s.rating_band,
                       s.final_bid, s.winner_name AS winner,
                       s.learner_final_offer, s.opponent_final_offer
                FROM selections s
                JOIN evaluation_games g ON g.id = s.game_id
                {where}
                ORDER BY g.opponent_name, g.episode, s.auction_number
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def progress(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT g.checkpoint_steps, g.opponent_name AS opponent,
                       ROUND(AVG(CASE
                           WHEN g.winner_role = 'learner' THEN 100.0
                           WHEN g.winner_role = 'tie' THEN 50.0
                           ELSE 0.0
                       END), 1) AS win_credit_pct,
                       ROUND(AVG(p.score), 2) AS avg_score
                       ,ROUND(AVG(p.items_drafted), 2) AS avg_items
                       ,ROUND(AVG(p.budget_used), 2) AS avg_budget_used
                FROM evaluation_games g
                JOIN player_results p
                  ON p.game_id = g.id AND p.player_role = 'learner'
                WHERE g.run_id = ?
                GROUP BY g.checkpoint_steps, g.opponent_name
                ORDER BY g.checkpoint_steps, g.opponent_name
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def action_progress(self, run_id: str) -> list[dict[str, Any]]:
        """Return the learner's action mix at each evaluation checkpoint."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                WITH counts AS (
                    SELECT g.checkpoint_steps, a.action_name, COUNT(*) AS uses
                    FROM learner_actions a
                    JOIN evaluation_games g ON g.id = a.game_id
                    WHERE g.run_id = ?
                    GROUP BY g.checkpoint_steps, a.action_name
                )
                SELECT checkpoint_steps, action_name, uses,
                       ROUND(
                           100.0 * uses /
                           SUM(uses) OVER (PARTITION BY checkpoint_steps),
                           2
                       ) AS action_pct
                FROM counts
                ORDER BY checkpoint_steps, action_name
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def bid_curve(
        self,
        run_id: str,
        *,
        opponent: str | None = None,
    ) -> list[dict[str, Any]]:
        """Observed learner offers by exact rating and checkpoint.

        ``avg_stop_offer`` only uses auctions won by the opponent, which means
        the learner eventually chose to pass. It is therefore the closest
        observable proxy for a maximum willingness to pay.
        """

        opponent_clause = ""
        parameters: list[Any] = [run_id]
        if opponent:
            opponent_clause = "AND g.opponent_name = ?"
            parameters.append(opponent)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT g.checkpoint_steps,
                       CAST(s.rating AS INTEGER) AS rating,
                       COUNT(*) AS items,
                       ROUND(AVG(s.learner_final_offer), 2)
                           AS avg_final_offer,
                       ROUND(AVG(
                           CASE WHEN s.winner_role = 'opponent'
                                THEN s.learner_final_offer END
                       ), 2) AS avg_stop_offer,
                       SUM(s.winner_role = 'opponent') AS stop_samples,
                       ROUND(100.0 * AVG(
                           CASE WHEN s.winner_role = 'learner'
                                THEN 1.0 ELSE 0.0 END
                       ), 1) AS learner_win_pct
                FROM selections s
                JOIN evaluation_games g ON g.id = s.game_id
                WHERE g.run_id = ?
                  {opponent_clause}
                GROUP BY g.checkpoint_steps, CAST(s.rating AS INTEGER)
                ORDER BY g.checkpoint_steps, rating
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]
