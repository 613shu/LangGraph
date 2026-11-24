"""Graph nodes and routing functions for the spam-cleaner agent."""
import json
from functools import lru_cache
from typing import Literal

from googleapiclient.errors import HttpError
from langchain.chat_models import init_chat_model
from langsmith import traceable
from langgraph.types import interrupt

from gmail_tools import fetch_messages, move_to_inbox, trash_message
from state import CleanerState, VerdictList

BATCH_SIZE = 30   # messages shown to the user per review round
LLM_CHUNK = 10    # messages sent to the LLM per call
MODEL = "google_genai:gemini-3.8-flash"


# ---------- 3.1 fetch ----------

def fetch(state: CleanerState) -> dict:
    """Fetch the next batch of not-yet-reviewed messages from Promotions + Spam."""
    seen_ids = set(state.get("seen_ids") or [])
    batch, has_more = fetch_messages(seen_ids, BATCH_SIZE)
    return {"batch": batch, "has_more": has_more}


# ---------- 3.2 classify ----------

SYSTEM_PROMPT = """You classify Gmail messages from the Promotions and Spam folders.
For each message return one verdict:
- "junk": a regular ad, a marketing newsletter, or spam.
- "maybe_important": a personal message, a mailing-list / Google Groups message,
  a receipt, an order confirmation, or an official / bank notice.
Copy each id exactly as given. Write a short reason in Hebrew for every verdict."""

_FIELDS = ("id", "label", "sender", "subject", "snippet", "headers")
_FALLBACK_REASON = "לא סווג על ידי המודל – מוצג לבדיקה ליתר ביטחון."


@lru_cache(maxsize=1)
def _get_classifier():
    """Build the LLM once, bound to the VerdictList schema."""
    return init_chat_model(MODEL, temperature=0).with_structured_output(VerdictList)


@traceable(name="classify")
def classify(state: CleanerState) -> dict:
    """Classify the batch in chunks of LLM_CHUNK and attach verdict + reason to each message."""
    batch = [dict(m) for m in state.get("batch") or []]
    classifier = _get_classifier()

    for start in range(0, len(batch), LLM_CHUNK):
        chunk = batch[start:start + LLM_CHUNK]
        lines = [json.dumps({k: m.get(k) for k in _FIELDS}, ensure_ascii=False) for m in chunk]
        result = classifier.invoke([("system", SYSTEM_PROMPT), ("human", "\n".join(lines))])
        by_id = {v.id: v for v in (result.verdicts if result else [])}

        for m in chunk:
            v = by_id.get(m["id"])
            if v is None:                      # missing / wrong id -> safe default
                m["verdict"], m["reason"] = "maybe_important", _FALLBACK_REASON
            else:
                m["verdict"], m["reason"] = v.verdict, v.reason

    return {"batch": batch}


# ---------- 3.3 review ----------

def review(state: CleanerState) -> dict:
    """Pause for human approval. No side effects here: the node re-runs from the top on resume."""
    batch = state.get("batch") or []
    answer = interrupt({"batch": batch, "has_more": state.get("has_more", False)}) or {}

    batch_ids = [m["id"] for m in batch]
    valid = set(batch_ids)
    inbox = [i for i in dict.fromkeys(answer.get("inbox") or []) if i in valid]
    trash = [i for i in dict.fromkeys(answer.get("trash") or []) if i in valid and i not in inbox]
    decided = set(inbox) | set(trash)
    skip = [i for i in batch_ids if i not in decided]   # unchecked / explicit skip -> skip

    return {"decisions": {"trash": trash, "inbox": inbox, "skip": skip,
                          "stop": bool(answer.get("stop", False))}}


# ---------- 3.4 execute ----------

def execute(state: CleanerState) -> dict:
    """Apply the approved decisions in Gmail. Failed actions are logged and NOT marked as seen."""
    decisions = state.get("decisions") or {}
    by_id = {m["id"]: m for m in state.get("batch") or []}
    seen, log = [], []
    stats = {"trashed": 0, "inbox": 0, "skipped": 0, "errors": 0}

    def describe(msg_id: str) -> str:
        m = by_id.get(msg_id, {})
        return f"{m.get('subject', '')} ({m.get('sender', '')})"

    # Safe action first: move to the primary inbox.
    for msg_id in decisions.get("inbox", []):
        try:
            move_to_inbox(msg_id, by_id[msg_id]["label"])
            seen.append(msg_id)
            stats["inbox"] += 1
            log.append(f"📥 inbox: {describe(msg_id)}")
        except (HttpError, ValueError, KeyError) as e:
            stats["errors"] += 1
            log.append(f"⚠️ inbox failed: {describe(msg_id)} – {e}")

    for msg_id in decisions.get("trash", []):
        try:
            trash_message(msg_id)
            seen.append(msg_id)
            stats["trashed"] += 1
            log.append(f"🗑 trash: {describe(msg_id)}")
        except HttpError as e:
            stats["errors"] += 1
            log.append(f"⚠️ trash failed: {describe(msg_id)} – {e}")

    for msg_id in decisions.get("skip", []):
        seen.append(msg_id)
        stats["skipped"] += 1

    log.append(f"סבב הסתיים: {stats['trashed']} לאשפה, {stats['inbox']} לתיבה הראשית, "
               f"{stats['skipped']} דולגו, {stats['errors']} שגיאות.")
    return {"seen_ids": seen, "stats": stats, "log": log,
            "stop": bool(decisions.get("stop", False))}


# ---------- 3.5 route_empty ----------

def route_empty(state: CleanerState) -> Literal["empty", "classify"]:
    """Router after fetch: end the run when there is nothing left to review."""
    return "classify" if state.get("batch") else "empty"


# ---------- 3.6 route_more ----------

def route_more(state: CleanerState) -> Literal["more", "done"]:
    """Router after execute: loop back to fetch, unless the user stopped or nothing is left."""
    if state.get("stop"):
        return "done"
    return "more" if state.get("has_more") else "done"
