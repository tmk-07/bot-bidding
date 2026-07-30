from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from item_auction import AscendingAuctionDuel  # noqa: E402
from item_auction.rl import TrainingHistory  # noqa: E402


BUDGET = 500
POOL_SIZE = 20
HUMAN = "You"
BOT = "Random Bot"
GAME_VERSION = 4
TRAINING_HISTORY_PATH = ROOT / "data" / "training_history.sqlite3"

st.set_page_config(
    page_title="Auction",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1320px; padding-top: 2.25rem; }
      h1 { letter-spacing: -.035em; }
      [data-testid="stSidebar"] {
        border-right: 1px solid #d9e0e8;
      }
      .lot-card, .roster-card {
        background: #ffffff;
        border: 1px solid #d9e0e8;
        border-radius: 12px;
      }
      .lot-card { padding: 1.5rem; min-height: 238px; }
      .lot-index {
        color: #64748b; font-size: .77rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
      }
      .lot-name {
        color: #172033; font-size: 1.55rem; font-weight: 700;
        margin-top: .4rem;
      }
      .lot-value {
        color: #0f766e; font-size: 4.4rem; line-height: 1;
        font-weight: 750; margin: .45rem 0 .25rem;
      }
      .subtle { color: #64748b; font-size: .88rem; }
      .bid-status {
        background: #edf6f5; border: 1px solid #cce4e1;
        border-radius: 10px; padding: .9rem 1rem; margin-bottom: 1rem;
      }
      .bid-price { color:#0f766e; font-size:1.8rem; font-weight:750; }
      .notice {
        border-radius: 9px; padding: .72rem .85rem; margin-top: .75rem;
        border: 1px solid #d9e0e8; background: #ffffff; color: #334155;
      }
      .notice.win { border-color:#a7d8cf; background:#f0f9f7; color:#115e59; }
      .notice.loss { border-color:#e3d2bd; background:#fdf8f2; color:#854d0e; }
      .roster-card { padding: 1rem 1.1rem; min-height: 215px; }
      .roster-card.leader { border-color: #78b9ae; }
      .roster-head { display:flex; align-items:flex-start; justify-content:space-between; }
      .roster-name { color:#172033; font-weight:700; }
      .cash { color:#0f766e; font-size:1.45rem; font-weight:750; }
      .roster-row {
        display:flex; justify-content:space-between; padding:.3rem 0;
        border-bottom:1px solid #edf0f3; font-size:.86rem;
      }
      div[data-testid="stMetric"] {
        background:#ffffff; border:1px solid #d9e0e8;
        border-radius:10px; padding:.75rem .9rem;
      }
      .stButton > button { font-weight:650; }
    </style>
    """,
    unsafe_allow_html=True,
)


def start_draft() -> None:
    try:
        duel = AscendingAuctionDuel(
            HUMAN,
            BOT,
            budget=BUDGET,
            pool_size=POOL_SIZE,
        )
        duel.reset(secrets.randbits(63))
        st.session_state.duel = duel
        st.session_state.last_response = None
        st.session_state.game_result = None
        st.session_state.ui_error = None
        st.session_state.game_version = GAME_VERSION
    except (ValueError, TypeError) as exc:
        st.session_state.ui_error = str(exc)


def raise_bid(widget_key: str) -> None:
    try:
        duel = st.session_state.duel
        response = duel.human_raise(st.session_state[widget_key])
        st.session_state.last_response = response
        st.session_state.ui_error = None
        if duel.done:
            st.session_state.game_result = duel.result()
    except (ValueError, RuntimeError) as exc:
        st.session_state.ui_error = str(exc)


def pass_bid() -> None:
    try:
        duel = st.session_state.duel
        response = duel.human_pass()
        st.session_state.last_response = response
        st.session_state.ui_error = None
        if duel.done:
            st.session_state.game_result = duel.result()
    except (ValueError, RuntimeError) as exc:
        st.session_state.ui_error = str(exc)


def roster_card(name: str, duel: AscendingAuctionDuel, leader: str) -> str:
    roster = duel.engine.rosters[name]
    score = duel.engine.scores[name]
    rows = "".join(
        (
            '<div class="roster-row">'
            f"<span>{entry.item.name} · {entry.item.value:.0f}</span>"
            f"<strong>${entry.price}</strong></div>"
        )
        for entry in reversed(roster)
    )
    rows = rows or '<div class="subtle" style="margin-top:.8rem">No items yet</div>'
    leader_class = " leader" if name == leader else ""
    return (
        f'<div class="roster-card{leader_class}">'
        '<div class="roster-head">'
        f'<div><div class="roster-name">{name}</div>'
        f'<div class="subtle">{len(roster)} items · {score:.0f} points</div></div>'
        f'<div class="cash">${duel.engine.budgets[name]}</div></div>'
        f'<div style="max-height:145px;overflow:auto;margin-top:.65rem">{rows}</div></div>'
    )


def training_line_chart(wide_frame: pd.DataFrame, y_label: str) -> None:
    """Render a checkpoint chart whose training-step axis begins exactly at 0."""

    frame = (
        wide_frame.reset_index()
        .melt(
            id_vars="checkpoint_steps",
            var_name="Series",
            value_name=y_label,
        )
        .dropna()
    )
    max_steps = max(1, int(frame["checkpoint_steps"].max()))
    chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "checkpoint_steps:Q",
                title="Training steps",
                scale=alt.Scale(domain=[0, max_steps], nice=False),
            ),
            y=alt.Y(f"{y_label}:Q", title=y_label),
            color=alt.Color("Series:N", title=None),
            tooltip=[
                alt.Tooltip("Series:N"),
                alt.Tooltip("checkpoint_steps:Q", title="Training steps", format=","),
                alt.Tooltip(f"{y_label}:Q", format=".2f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def family_evolution_rows(
    history: TrainingHistory,
    runs: list[dict],
) -> list[dict]:
    """Combine completed sessions into one chronological learner lineage."""

    usable_runs = sorted(
        (
            run
            for run in runs
            if run["status"] in {"completed", "truncated"}
        ),
        key=lambda run: run["created_at"],
    )
    rows: list[dict] = []
    step_offset = 0
    sequence = 0
    for run_number, run in enumerate(usable_runs, start=1):
        config = json.loads(run["config_json"])
        phase = config.get("training_phase", "context").replace("-", " ").title()
        progress = history.progress(run["id"])
        checkpoints = sorted({row["checkpoint_steps"] for row in progress})
        for checkpoint in checkpoints:
            checkpoint_rows = [
                row
                for row in progress
                if row["checkpoint_steps"] == checkpoint
            ]
            if not checkpoint_rows:
                continue
            sequence += 1
            rows.append(
                {
                    "family_steps": step_offset + checkpoint,
                    "sequence": sequence,
                    "session": f"{run_number}. {phase}",
                    "run_id": run["id"],
                    "checkpoint_steps": checkpoint,
                    "win_credit_pct": sum(
                        row["win_credit_pct"] for row in checkpoint_rows
                    )
                    / len(checkpoint_rows),
                    "avg_score": sum(
                        row["avg_score"] for row in checkpoint_rows
                    )
                    / len(checkpoint_rows),
                    "avg_items": sum(
                        row["avg_items"] for row in checkpoint_rows
                    )
                    / len(checkpoint_rows),
                    "avg_budget_used": sum(
                        row["avg_budget_used"] for row in checkpoint_rows
                    )
                    / len(checkpoint_rows),
                }
            )
        if checkpoints:
            step_offset += checkpoints[-1]
    return rows


def family_pricing_rows(
    history: TrainingHistory,
    runs: list[dict],
) -> list[dict]:
    """Return final-checkpoint pricing behavior for every usable session."""

    usable_runs = sorted(
        (
            run
            for run in runs
            if run["status"] in {"completed", "truncated"}
        ),
        key=lambda run: run["created_at"],
    )
    rows: list[dict] = []
    for run_number, run in enumerate(usable_runs, start=1):
        checkpoints = history.checkpoints(run["id"])
        if not checkpoints:
            continue
        checkpoint = (
            run.get("selected_checkpoint_steps")
            if run.get("selected_checkpoint_steps") in checkpoints
            else checkpoints[-1]
        )
        config = json.loads(run["config_json"])
        phase = config.get("training_phase", "context").replace("-", " ").title()
        for row in history.pricing_behavior(run["id"], checkpoint):
            rows.append(
                {
                    **row,
                    "session": f"{run_number}. {phase}",
                    "run_id": run["id"],
                    "checkpoint_steps": checkpoint,
                }
            )
    return rows


with st.sidebar:
    st.markdown("### Game setup")
    st.markdown(
        """
        **You vs Random Bot**

        20 items · ratings 1–100
        $500 each · no roster limit
        """
    )
    st.button(
        "New draft",
        type="primary",
        use_container_width=True,
        on_click=start_draft,
    )
    st.divider()
    st.caption(
        "The bot sets a private random spending limit for each item. "
        "It raises by $1 until that limit is reached."
    )

if st.session_state.get("game_version") != GAME_VERSION:
    start_draft()

st.title("Auction")
st.caption(
    "Twenty items. Two bidders. Raise the current price or pass. "
    "Upcoming items stay hidden."
)

if st.session_state.get("ui_error"):
    st.error(st.session_state.ui_error)

duel: AscendingAuctionDuel = st.session_state.duel
env = duel.env
result = st.session_state.game_result
completed = len(env.history)

play_tab, history_tab, training_tab, rules_tab = st.tabs(
    ["Draft", "Game history", "Training", "Rules"]
)

with play_tab:
    metrics = st.columns(4)
    metrics[0].metric("Auction", f"{min(completed + 1, POOL_SIZE)} / {POOL_SIZE}")
    metrics[1].metric("Your cash", f"${duel.human_budget}")
    metrics[2].metric("Bot cash", f"${duel.bot_budget}")
    metrics[3].metric("Items left", POOL_SIZE - completed)
    st.progress(completed / POOL_SIZE)

    lot_col, action_col = st.columns([1.05, 1.6])
    with lot_col:
        if not duel.done and duel.current_item:
            item = duel.current_item
            st.markdown(
                f"""
                <div class="lot-card">
                  <div class="lot-index">Item {completed + 1} of {POOL_SIZE}</div>
                  <div class="lot-name">{item.name}</div>
                  <div class="lot-value">{item.visible_value:.0f}</div>
                  <div class="subtle">rating · {item.category}</div>
                  <div class="subtle" style="margin-top:1.1rem">
                    Upcoming items are hidden from both bidders.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            winner = "Tie"
            if result and len(result.winners) == 1:
                winner = "You win" if result.winners[0] == HUMAN else "Bot wins"
            st.markdown(
                f"""
                <div class="lot-card">
                  <div class="lot-index">Draft complete</div>
                  <div class="lot-name" style="margin-top:1rem">{winner}</div>
                  <div class="subtle" style="margin-top:.5rem">
                    All {POOL_SIZE} items have been auctioned.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with action_col:
        if not duel.done:
            leader_text = duel.leader or "No bids yet"
            st.markdown(
                f"""
                <div class="bid-status">
                  <div class="subtle">Current bid</div>
                  <div class="bid-price">${duel.current_price}</div>
                  <div class="subtle">Leader: {leader_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            can_raise = duel.minimum_raise <= duel.human_budget
            if can_raise:
                bid_key = f"raise_{env.item_index}_{duel.current_price}"
                st.number_input(
                    "Your next bid",
                    min_value=duel.minimum_raise,
                    max_value=duel.human_budget,
                    value=duel.minimum_raise,
                    step=1,
                    key=bid_key,
                )
                button_cols = st.columns(2)
                button_cols[0].button(
                    "Raise",
                    type="primary",
                    use_container_width=True,
                    on_click=raise_bid,
                    args=(bid_key,),
                )
                pass_label = "Pass" if duel.leader else "Pass item"
                button_cols[1].button(
                    pass_label,
                    use_container_width=True,
                    on_click=pass_bid,
                )
            else:
                st.warning("You do not have enough cash to raise.")
                st.button(
                    "Pass",
                    type="primary",
                    use_container_width=True,
                    on_click=pass_bid,
                )
            st.caption("The bot's private limit is revealed only by its decisions.")
        else:
            final_cols = st.columns(2)
            final_cols[0].metric(
                "Your score",
                f"{result.scores[HUMAN]:.0f}",
                f"{len(result.rosters[HUMAN])} items",
            )
            final_cols[1].metric(
                "Bot score",
                f"{result.scores[BOT]:.0f}",
                f"{len(result.rosters[BOT])} items",
            )
            st.button(
                "Play again",
                type="primary",
                use_container_width=True,
                on_click=start_draft,
            )

        response = st.session_state.last_response
        if response:
            css = "notice"
            if response.resolved and response.auction:
                if response.auction.winner == HUMAN:
                    css += " win"
                elif response.auction.winner == BOT:
                    css += " loss"
            st.markdown(
                f'<div class="{css}">{response.message}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Rosters")
    live_scores = dict(env.scores)
    score_leader = max(live_scores, key=live_scores.get)
    roster_cols = st.columns(2)
    roster_cols[0].markdown(
        roster_card(HUMAN, duel, score_leader), unsafe_allow_html=True
    )
    roster_cols[1].markdown(
        roster_card(BOT, duel, score_leader), unsafe_allow_html=True
    )

with history_tab:
    rows = [
        {
            "#": index + 1,
            "Item": auction.item.name,
            "Rating": auction.item.value,
            "Winner": auction.winner or "Unsold",
            "Price": auction.price,
            "Your final offer": auction.bids[HUMAN],
            "Bot final offer": auction.bids[BOT],
        }
        for index, auction in enumerate(env.history)
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True, height=520)
    else:
        st.info("No completed auctions yet.")

with training_tab:
    st.markdown("#### Baseline training history")
    st.caption(
        "Checkpoint evaluations against the opponents selected for each run. "
        "The pure-random bot is excluded."
    )
    training_history = TrainingHistory(TRAINING_HISTORY_PATH)
    runs = training_history.runs()
    if not runs:
        st.info(
            "No training runs yet. Start one with "
            "`item-auction-train-baseline --timesteps 100000`."
        )
    else:
        family_names = {
            "iterated": "Iterated RL Bot",
            "deal-value": "Deal-Value RL Bot",
        }
        available_families = list(
            dict.fromkeys(
                run.get("learner_family", "iterated") for run in runs
            )
        )
        selected_family = st.selectbox(
            "Learner",
            available_families,
            format_func=lambda value: family_names.get(
                value, value.replace("-", " ").title()
            ),
        )
        family_runs = [
            run
            for run in runs
            if run.get("learner_family", "iterated") == selected_family
        ]
        evolution_rows = family_evolution_rows(training_history, family_runs)
        if evolution_rows:
            st.markdown("##### Learner evolution across sessions")
            st.caption(
                "Each color is one training session. The x-axis accumulates "
                "learner decisions across completed sessions. Because curricula "
                "can change, performance metrics reflect each session's own "
                "evaluation opponents."
            )
            evolution_metrics = {
                "Win credit": ("win_credit_pct", "Win credit %"),
                "Average score": ("avg_score", "Average score"),
                "Items drafted": ("avg_items", "Average items"),
                "Budget used": ("avg_budget_used", "Average dollars"),
            }
            evolution_choice = st.selectbox(
                "Evolution measure",
                list(evolution_metrics),
                key=f"evolution-{selected_family}",
            )
            evolution_field, evolution_label = evolution_metrics[
                evolution_choice
            ]
            evolution_frame = pd.DataFrame(evolution_rows)
            evolution_chart = (
                alt.Chart(evolution_frame)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "family_steps:Q",
                        title="Cumulative learner decisions",
                        scale=alt.Scale(
                            domain=[
                                0,
                                max(
                                    1,
                                    int(evolution_frame["family_steps"].max()),
                                ),
                            ],
                            nice=False,
                        ),
                    ),
                    y=alt.Y(f"{evolution_field}:Q", title=evolution_label),
                    color=alt.Color("session:N", title="Session"),
                    order=alt.Order("sequence:Q"),
                    tooltip=[
                        alt.Tooltip("session:N", title="Session"),
                        alt.Tooltip("run_id:N", title="Run"),
                        alt.Tooltip(
                            "family_steps:Q",
                            title="Cumulative decisions",
                            format=",",
                        ),
                        alt.Tooltip(
                            "checkpoint_steps:Q",
                            title="Session checkpoint",
                            format=",",
                        ),
                        alt.Tooltip(
                            f"{evolution_field}:Q",
                            title=evolution_label,
                            format=".2f",
                        ),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(evolution_chart, use_container_width=True)
        pricing_rows = family_pricing_rows(training_history, family_runs)
        if pricing_rows:
            st.markdown("##### Pricing behavior across sessions")
            st.caption(
                "Compares the final checkpoint from each completed session. "
                "Final auction price is the clearing price; learner final offer "
                "is the highest amount the RL bot actually bid before the "
                "auction ended."
            )
            pricing_measures = {
                "Final auction price": (
                    "avg_final_price",
                    "Average final price",
                ),
                "Learner final offer": (
                    "avg_learner_offer",
                    "Average learner offer",
                ),
                "Price when learner won": (
                    "avg_price_when_won",
                    "Average price paid",
                ),
            }
            pricing_choice = st.selectbox(
                "Pricing measure",
                list(pricing_measures),
                key=f"pricing-{selected_family}",
            )
            pricing_field, pricing_label = pricing_measures[pricing_choice]
            pricing_frame = pd.DataFrame(pricing_rows)
            pricing_chart = (
                alt.Chart(pricing_frame)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "rating_band:N",
                        title="Item rating",
                        sort=[
                            "1-9",
                            "10-19",
                            "20-29",
                            "30-39",
                            "40-49",
                            "50-59",
                            "60-69",
                            "70-79",
                            "80-89",
                            "90-100",
                        ],
                    ),
                    y=alt.Y(f"{pricing_field}:Q", title=pricing_label),
                    color=alt.Color("session:N", title="Session"),
                    tooltip=[
                        alt.Tooltip("session:N", title="Session"),
                        alt.Tooltip("run_id:N", title="Run"),
                        alt.Tooltip(
                            "checkpoint_steps:Q",
                            title="Final checkpoint",
                            format=",",
                        ),
                        alt.Tooltip("rating_band:N", title="Rating"),
                        alt.Tooltip(
                            f"{pricing_field}:Q",
                            title=pricing_label,
                            format=".2f",
                        ),
                        alt.Tooltip("items:Q", title="Items"),
                        alt.Tooltip(
                            "learner_win_pct:Q",
                            title="RL won %",
                            format=".1f",
                        ),
                    ],
                )
                .properties(height=340)
            )
            st.altair_chart(pricing_chart, use_container_width=True)
        run_labels = {
            (
                f"{run['created_at'][:19].replace('T', ' ')} · "
                f"{json.loads(run['config_json']).get('training_phase', 'context').replace('-', ' ').title()} · "
                f"{run['algorithm']} · {run['status']} · {run['id']}"
            ): run["id"]
            for run in family_runs
        }
        selected_run_label = st.selectbox("Training run", list(run_labels))
        selected_run = run_labels[selected_run_label]
        selected_run_record = next(
            run for run in family_runs if run["id"] == selected_run
        )
        checkpoints = training_history.checkpoints(selected_run)
        if not checkpoints:
            st.warning("This run has not completed its first evaluation yet.")
        else:
            promoted_checkpoint = selected_run_record.get(
                "selected_checkpoint_steps"
            )
            default_checkpoint = (
                promoted_checkpoint
                if promoted_checkpoint in checkpoints
                else checkpoints[-1]
            )
            selected_checkpoint = st.select_slider(
                "Checkpoint steps",
                options=checkpoints,
                value=default_checkpoint,
                format_func=lambda value: f"{value:,}",
            )
            if promoted_checkpoint in checkpoints:
                st.caption(
                    f"Promoted checkpoint: {promoted_checkpoint:,} steps. "
                    "The slider defaults to the model selected for play."
                )
            summary = training_history.player_summary(
                selected_run, selected_checkpoint
            )
            learner_rows = [row for row in summary if row["role"] == "learner"]
            total_games = sum(row["games"] for row in learner_rows)
            if learner_rows:
                weighted = lambda field: sum(
                    row[field] * row["games"] for row in learner_rows
                ) / max(1, total_games)
                summary_metrics = st.columns(4)
                summary_metrics[0].metric("Evaluation games", f"{total_games:,}")
                summary_metrics[1].metric(
                    "RL win credit", f"{weighted('win_credit_pct'):.1f}%"
                )
                summary_metrics[2].metric(
                    "RL average score", f"{weighted('avg_score'):.1f}"
                )
                summary_metrics[3].metric(
                    "RL average budget used",
                    f"${weighted('avg_budget_used'):.1f}",
                )

            st.markdown("##### Player results by opponent")
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "opponent": "Opponent",
                    "role": "Role",
                    "player": "Player",
                    "games": "Games",
                    "avg_score": "Avg score",
                    "avg_items": "Avg items drafted",
                    "avg_budget_used": "Avg budget used",
                    "avg_budget_remaining": "Avg budget left",
                    "win_credit_pct": "Win credit %",
                },
            )

            exposure_rows = training_history.training_exposure(selected_run)
            if exposure_rows:
                st.markdown("##### Training matches sampled")
                st.caption(
                    "Actual completed training-game exposure, separate from "
                    "the held-out evaluation games shown above."
                )
                st.dataframe(
                    exposure_rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "opponent": "Opponent",
                        "training_matches": "Training matches",
                        "actual_share_pct": "Actual share %",
                    },
                )

            progress = training_history.progress(selected_run)
            if progress:
                progress_frame = pd.DataFrame(progress)
                win_progress = progress_frame.pivot(
                    index="checkpoint_steps",
                    columns="opponent",
                    values="win_credit_pct",
                )
                score_progress = progress_frame.pivot(
                    index="checkpoint_steps",
                    columns="opponent",
                    values="avg_score",
                )
                item_progress = progress_frame.pivot(
                    index="checkpoint_steps",
                    columns="opponent",
                    values="avg_items",
                )
                budget_progress = progress_frame.pivot(
                    index="checkpoint_steps",
                    columns="opponent",
                    values="avg_budget_used",
                )
                chart_columns = st.columns(2)
                with chart_columns[0]:
                    st.markdown("##### Win credit over training")
                    training_line_chart(
                        win_progress,
                        "Win credit %",
                    )
                with chart_columns[1]:
                    st.markdown("##### Average score over training")
                    training_line_chart(
                        score_progress,
                        "Average score",
                    )
                behavior_columns = st.columns(2)
                with behavior_columns[0]:
                    st.markdown("##### Items drafted over training")
                    training_line_chart(
                        item_progress,
                        "Average items",
                    )
                with behavior_columns[1]:
                    st.markdown("##### Budget used over training")
                    training_line_chart(
                        budget_progress,
                        "Average dollars",
                    )

            action_progress = training_history.action_progress(selected_run)
            if action_progress:
                action_frame = pd.DataFrame(action_progress)
                action_mix = action_frame.pivot(
                    index="checkpoint_steps",
                    columns="action_name",
                    values="action_pct",
                ).fillna(0)
                st.markdown("##### RL action mix over training")
                st.caption(
                    "Share of the RL bot's evaluation decisions assigned to "
                    "each bid action at every checkpoint."
                )
                training_line_chart(
                    action_mix,
                    "Share of decisions %",
                )

            st.markdown("##### RL willingness to pay over training")
            st.caption(
                "The bot has no fixed maximum-bid formula. This estimates its "
                "learned stopping behavior from completed evaluation auctions."
            )
            curve_opponents = sorted(
                {row["opponent"] for row in summary}
            )
            curve_controls = st.columns(2)
            with curve_controls[0]:
                curve_opponent = st.selectbox(
                    "Curve opponent",
                    ["All opponents", *curve_opponents],
                )
            with curve_controls[1]:
                curve_measure = st.selectbox(
                    "Curve measure",
                    ["Offer before passing", "Observed final offer"],
                )
            bid_curve_rows = training_history.bid_curve(
                selected_run,
                opponent=(
                    None
                    if curve_opponent == "All opponents"
                    else curve_opponent
                ),
            )
            if bid_curve_rows:
                bid_curve_frame = pd.DataFrame(bid_curve_rows)
                measure_field = (
                    "avg_stop_offer"
                    if curve_measure == "Offer before passing"
                    else "avg_final_offer"
                )
                measure_label = (
                    "Average offer before passing"
                    if curve_measure == "Offer before passing"
                    else "Average observed final offer"
                )
                curve_chart = (
                    alt.Chart(bid_curve_frame.dropna(subset=[measure_field]))
                    .mark_line(point=False)
                    .encode(
                        x=alt.X(
                            "rating:Q",
                            title="Item rating",
                            scale=alt.Scale(
                                domain=[0, 100],
                                nice=False,
                            ),
                        ),
                        y=alt.Y(
                            f"{measure_field}:Q",
                            title=measure_label,
                            scale=alt.Scale(zero=True),
                        ),
                        color=alt.Color(
                            "checkpoint_steps:O",
                            title="Checkpoint",
                            sort="ascending",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "checkpoint_steps:Q",
                                title="Training steps",
                                format=",",
                            ),
                            alt.Tooltip("rating:Q", title="Rating"),
                            alt.Tooltip(
                                f"{measure_field}:Q",
                                title=measure_label,
                                format=".2f",
                            ),
                            alt.Tooltip("items:Q", title="Items evaluated"),
                            alt.Tooltip(
                                "stop_samples:Q",
                                title="Pass samples",
                            ),
                            alt.Tooltip(
                                "learner_win_pct:Q",
                                title="RL won %",
                                format=".1f",
                            ),
                        ],
                    )
                    .properties(height=340)
                )
                st.altair_chart(curve_chart, use_container_width=True)
                if curve_measure == "Offer before passing":
                    st.caption(
                        "This uses items the opponent won, meaning the RL bot "
                        "eventually passed. Winning items end before revealing "
                        "the bot's true stopping point."
                    )

            st.markdown("##### Price and ownership by rating range")
            rating_rows = training_history.rating_summary(
                selected_run, selected_checkpoint
            )
            st.dataframe(
                rating_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "opponent": "Opponent",
                    "rating_band": "Rating range",
                    "items": "Items shown",
                    "avg_final_bid": "Avg final bid",
                    "learner_won": "RL Bot won",
                    "opponent_won": "Opponent won",
                    "unsold": "Unsold",
                },
            )

            st.markdown("##### Individual selections")
            opponents = sorted({row["opponent"] for row in summary})
            opponent_filter = st.selectbox(
                "Opponent filter",
                ["All opponents", *opponents],
            )
            selection_rows = training_history.selections(
                selected_run,
                selected_checkpoint,
                opponent=(
                    None
                    if opponent_filter == "All opponents"
                    else opponent_filter
                ),
            )
            st.dataframe(
                selection_rows,
                use_container_width=True,
                hide_index=True,
                height=520,
                column_config={
                    "episode": "Game",
                    "seed": "Seed",
                    "opponent": "Opponent",
                    "auction_number": "Auction",
                    "item_name": "Item",
                    "rating": "Rating",
                    "rating_band": "Range",
                    "final_bid": "Final bid",
                    "winner": "Winner",
                    "learner_final_offer": "RL final offer",
                    "opponent_final_offer": "Opponent final offer",
                },
            )
            if len(selection_rows) == 5_000:
                st.caption(
                    "Showing the first 5,000 selections for this filter. "
                    "All rows remain stored in the SQLite history database."
                )

with rules_tab:
    st.markdown(
        """
        #### Bidding

        You open the bidding or pass. After every bid, the bot either raises by
        $1 or holds. If it raises, you can raise again or pass. The current
        leader pays the visible current bid.

        The bot samples a private random percentage of its remaining budget as
        its limit for each new item. That limit stays fixed during the item's
        auction.

        #### Draft

        - Exactly 20 items, each rated from 1–100
        - $500 starting budget per player
        - No required roster split
        - Upcoming items are unavailable to both bidders
        - Highest total item rating after all 20 auctions wins
        """
    )
