"""Streamlit UI for the spam-cleaner agent.

Run with:  streamlit run app.py
"""
# Use the Windows certificate store (needed behind filtered / inspected networks).
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph import RECURSION_LIMIT, build_graph

DB_FILE = Path(__file__).with_name("cleaner.sqlite")
THREAD_ID = "cleaner-main"
ACTIONS = {"inbox": "📥 לתיבה הראשית", "trash": "🗑 לאשפה", "skip": "⏭ דלג"}


# ---------- 5.1 get_graph ----------

@st.cache_resource
def get_graph():
    """Build the graph once per Streamlit server, with a persistent SQLite checkpointer."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return build_graph(SqliteSaver(conn))


# ---------- 5.2 render_review ----------

def render_review(payload: dict) -> dict | None:
    """Draw the approval screen for one batch. Returns the resume dict only after a button click."""
    batch = payload.get("batch") or []
    important = [m for m in batch if m.get("verdict") == "maybe_important"]
    junk = [m for m in batch if m.get("verdict") != "maybe_important"]
    batch_key = batch[0]["id"] if batch else "empty"

    st.subheader(f"{len(batch)} הודעות לבדיקה")
    if payload.get("has_more"):
        st.caption("אחרי האישור תוצג האצווה הבאה.")

    # ⭐ Possibly important – one decision per message, default: move to inbox.
    choices: dict[str, str] = {}
    if important:
        st.markdown(f"### ⭐ ייתכן שחשוב ({len(important)})")
        for m in important:
            st.markdown(f"**{m.get('subject', '')}** · {m.get('sender', '')}  \n"
                        f"_{m.get('reason', '')}_")
            choices[m["id"]] = st.radio(
                "פעולה", list(ACTIONS), format_func=ACTIONS.get, horizontal=True,
                key=f"act-{batch_key}-{m['id']}", label_visibility="collapsed")
            st.divider()

    # Junk – collapsed table, all checked for trash by default.
    edited = pd.DataFrame()
    if junk:
        with st.expander(f"🗑 פרסומות וספאם ({len(junk)})"):
            select_all = st.checkbox("בחר הכול", value=True, key=f"all-{batch_key}")
            table = pd.DataFrame([{
                "trash": select_all,
                "subject": m.get("subject", ""),
                "sender": m.get("sender", ""),
                "label": m.get("label", ""),
                "reason": m.get("reason", ""),
                "id": m["id"],
            } for m in junk])
            edited = st.data_editor(
                table, hide_index=True, use_container_width=True,
                key=f"junk-{batch_key}-{select_all}",
                disabled=["subject", "sender", "label", "reason"],
                column_config={
                    "trash": st.column_config.CheckboxColumn("לאשפה"),
                    "subject": "נושא", "sender": "שולח", "label": "תווית", "reason": "סיבה",
                    "id": None,   # hidden
                })

    col_go, col_stop = st.columns(2)
    go = col_go.button("✅ אשר והמשך", type="primary", key=f"go-{batch_key}")
    stop = col_stop.button("⏹ אשר ועצור", key=f"stop-{batch_key}")
    if not (go or stop):
        return None

    decisions = {"trash": [], "inbox": [], "skip": [], "stop": stop}
    for msg_id, action in choices.items():
        decisions[action].append(msg_id)
    for row in edited.to_dict("records"):
        decisions["trash" if row["trash"] else "skip"].append(row["id"])
    return decisions


# ---------- 5.3 main ----------

def main() -> None:
    """Start a run, resume a paused one, and show totals + the action log."""
    st.set_page_config(page_title="ניקוי פרסומות וספאם", page_icon="🧹")
    st.markdown("<style>.stApp {direction: rtl; text-align: right;}</style>",
                unsafe_allow_html=True)
    st.title("🧹 ניקוי פרסומות וספאם")

    graph = get_graph()
    cfg = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": RECURSION_LIMIT}
    snapshot = graph.get_state(cfg)

    # Paused at review (also after a browser reload) -> show the approval screen.
    if "review" in (snapshot.next or ()):
        payload = next((i.value for t in snapshot.tasks for i in t.interrupts), {})
        decisions = render_review(payload)
        if decisions is not None:
            with st.spinner("מבצעת את הפעולות ומביאה את האצווה הבאה…"):
                graph.invoke(Command(resume=decisions), cfg)
            st.rerun()
        return

    if st.button("🚀 הפעל ניקוי", type="primary"):
        with st.spinner("מביאה הודעות ומסווגת…"):
            graph.invoke({"batch": [], "has_more": False, "stop": False, "decisions": {}}, cfg)
        st.rerun()

    values = snapshot.values or {}
    stats = values.get("stats") or {}
    if stats:
        st.markdown("### סיכום (מצטבר מכל הריצות)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("לאשפה", stats.get("trashed", 0))
        c2.metric("לתיבה הראשית", stats.get("inbox", 0))
        c3.metric("דולגו", stats.get("skipped", 0))
        c4.metric("שגיאות", stats.get("errors", 0))
    log = values.get("log") or []
    if log:
        with st.expander("יומן פעולות (50 האחרונות)"):
            st.text("\n".join(log[-50:]))


if __name__ == "__main__":
    main()
