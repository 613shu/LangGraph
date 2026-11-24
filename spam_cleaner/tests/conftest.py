"""Shared test helpers: a fake LLM, a fake Gmail and a graph wired to both.

No test here talks to Gmail, Gemini or LangSmith.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent          # spam_cleaner/
sys.path.insert(0, str(ROOT))                          # from state / nodes / graph import ...
sys.path.insert(0, str(ROOT / "eval"))                 # from dataset / run_eval import ...
os.environ["LANGSMITH_TRACING"] = "false"              # tests never send traces
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from state import Verdict, VerdictList  # noqa: E402


# ---------- 7.1 FakeClassifier ----------

class FakeClassifier:
    """Fake LLM that mimics `init_chat_model(...).with_structured_output(VerdictList)`."""

    def __init__(self, answers: dict[str, str], drop: set[str] | None = None):
        self.answers = answers          # id -> "junk" / "maybe_important"
        self.drop = set(drop or [])     # ids the "model" forgets to return
        self.calls: list[int] = []      # number of messages in each call

    def invoke(self, messages: list[tuple[str, str]]) -> VerdictList:
        human = next(text for role, text in messages if role == "human")
        ids = [json.loads(line)["id"] for line in human.splitlines() if line.strip()]
        self.calls.append(len(ids))
        return VerdictList(verdicts=[
            Verdict(id=i, verdict=self.answers.get(i, "junk"), reason=f"fake reason for {i}")
            for i in ids if i not in self.drop
        ])


# ---------- 7.2 FakeGmail ----------

class FakeGmail:
    """In-memory Promotions + Spam mailbox with the same API as gmail_tools."""

    def __init__(self, messages: list[dict]):
        self.boxes: dict[str, dict] = {m["id"]: dict(m) for m in messages}  # keeps order
        self.trashed: list[str] = []
        self.moved: list[tuple[str, str]] = []
        self.fetch_calls = 0

    def fetch_messages(self, seen_ids: set[str], limit: int = 30) -> tuple[list[dict], bool]:
        self.fetch_calls += 1
        unseen = [dict(m) for i, m in self.boxes.items() if i not in seen_ids]
        return unseen[:limit], len(unseen) > limit

    def trash_message(self, msg_id: str) -> None:
        self.boxes.pop(msg_id)
        self.trashed.append(msg_id)

    def move_to_inbox(self, msg_id: str, label: str) -> None:
        if label not in ("CATEGORY_PROMOTIONS", "SPAM"):
            raise ValueError(f"Unknown label: {label!r}")
        self.boxes.pop(msg_id)
        self.moved.append((msg_id, label))


# ---------- 7.3 fake_env ----------

START_INPUT = {"batch": [], "has_more": False, "stop": False, "decisions": {}}


@pytest.fixture
def fake_env(monkeypatch):
    """The real graph (InMemorySaver) with nodes wired to FakeClassifier + FakeGmail.

    The mailbox holds the 30 messages of eval/dataset.EXAMPLES, and the fake LLM
    answers each one with its expected verdict.
    """
    import nodes
    from dataset import EXAMPLES
    from graph import RECURSION_LIMIT, build_graph
    from langgraph.checkpoint.memory import InMemorySaver

    expected = {m["id"]: verdict for m, verdict, _ in EXAMPLES}
    gmail = FakeGmail([m for m, _, _ in EXAMPLES])
    llm = FakeClassifier(expected)

    monkeypatch.setattr(nodes, "_get_classifier", lambda: llm)
    monkeypatch.setattr(nodes, "fetch_messages", gmail.fetch_messages)
    monkeypatch.setattr(nodes, "trash_message", gmail.trash_message)
    monkeypatch.setattr(nodes, "move_to_inbox", gmail.move_to_inbox)

    return SimpleNamespace(
        graph=build_graph(InMemorySaver()),
        cfg={"configurable": {"thread_id": "test"}, "recursion_limit": RECURSION_LIMIT},
        gmail=gmail, llm=llm, expected=expected, start=dict(START_INPUT),
    )
