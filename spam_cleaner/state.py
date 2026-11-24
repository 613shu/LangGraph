"""State and schemas for the spam-cleaner agent."""
import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------- Reducers ----------

def merge_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer for seen_ids: merge two ID lists, drop duplicates, keep order."""
    left = left or []
    right = right or []
    seen = set(left)
    merged = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def sum_counts(left: dict | None, right: dict | None) -> dict:
    """Reducer for stats: add counters key by key (e.g. trashed/inbox/skipped)."""
    result = dict(left or {})
    for key, value in (right or {}).items():
        result[key] = result.get(key, 0) + value
    return result


# ---------- LLM structured output ----------

class Verdict(BaseModel):
    """Classification of a single message."""
    id: str = Field(description="The message id, copied exactly from the input.")
    verdict: Literal["junk", "maybe_important"] = Field(
        description="'junk' for a regular ad or spam; 'maybe_important' for a personal "
                    "message, a mailing-list / Google Groups message, a receipt, an order "
                    "confirmation, or an official / bank notice."
    )
    reason: str = Field(
        description="A short reason for the verdict. Required when verdict is 'maybe_important'."
    )


class VerdictList(BaseModel):
    """Verdicts for one chunk of messages (one LLM call)."""
    verdicts: list[Verdict] = Field(description="One verdict per input message.")


# ---------- Graph state ----------

class CleanerState(TypedDict):
    batch: list[dict]                               # current batch, up to 30 messages (overwrite)
    has_more: bool                                  # more messages exist beyond this batch
    stop: bool                                      # user clicked "Approve & stop"
    decisions: dict                                 # {"trash","inbox","skip": [ids], "stop": bool} (overwrite)
    seen_ids: Annotated[list[str], merge_unique]    # already-reviewed message ids
    stats: Annotated[dict, sum_counts]              # {"trashed": n, "inbox": n, "skipped": n}
    log: Annotated[list[str], operator.add]         # action log
