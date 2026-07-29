from __future__ import annotations

import secrets
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from item_auction import AscendingAuctionDuel  # noqa: E402


BUDGET = 500
POOL_SIZE = 20
HUMAN = "You"
BOT = "Random Bot"
GAME_VERSION = 3

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

play_tab, history_tab, rules_tab = st.tabs(["Draft", "History", "Rules"])

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
