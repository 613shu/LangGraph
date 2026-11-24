"""End-to-end tests of the compiled graph: fake LLM + fake Gmail + InMemorySaver.

Flow under test: fetch -> classify -> review (interrupt) -> Command(resume=...) -> execute -> loop / END.
"""
from langgraph.types import Command

import nodes


def _default_decisions(batch: list[dict], stop: bool = False) -> dict:
    """What the review screen sends by default: junk -> trash, maybe_important -> inbox."""
    return {"trash": [m["id"] for m in batch if m["verdict"] == "junk"],
            "inbox": [m["id"] for m in batch if m["verdict"] == "maybe_important"],
            "stop": stop}


# ---------- 7.7 test_full_run_loops_until_empty ----------

def test_full_run_loops_until_empty(fake_env, monkeypatch):
    monkeypatch.setattr(nodes, "BATCH_SIZE", 20)        # 30 messages -> 2 rounds (20 + 10)
    g, cfg, gmail = fake_env.graph, fake_env.cfg, fake_env.gmail

    out = g.invoke(fake_env.start, cfg)
    batches = []
    while "__interrupt__" in out:
        value = out["__interrupt__"][0].value
        batch = value["batch"]
        batches.append(batch)

        # The screen gets every message with the LLM's verdict and reason.
        for m in batch:
            assert m["verdict"] == fake_env.expected[m["id"]]
            assert m["reason"]

        decisions = _default_decisions(batch)
        if len(batches) == 1:                            # leave one junk message unchecked
            decisions["trash"].remove("ex09")
            assert value["has_more"] is True
        out = g.invoke(Command(resume=decisions), cfg)

    assert [len(b) for b in batches] == [20, 10]
    assert all(m["id"] != "ex09" for m in batches[1])   # skipped message is not shown again
    assert fake_env.llm.calls == [10, 10, 10]            # LLM_CHUNK = 10

    assert len(gmail.moved) == 8 and len(gmail.trashed) == 21
    assert list(gmail.boxes) == ["ex09"]                 # unchecked message stays where it is

    state = g.get_state(cfg)
    assert state.next == ()                              # the run reached END
    assert state.values["stats"] == {"trashed": 21, "inbox": 8, "skipped": 1, "errors": 0}
    assert sorted(state.values["seen_ids"]) == sorted(fake_env.expected)
    assert len(state.values["log"]) == 21 + 8 + 2        # one line per action + one summary per round


# ---------- 7.8 test_approve_and_stop ----------

def test_approve_and_stop(fake_env, monkeypatch):
    monkeypatch.setattr(nodes, "BATCH_SIZE", 20)
    fake_env.llm.drop = {"ex20"}                         # the "model" forgets one junk message
    g, cfg, gmail = fake_env.graph, fake_env.cfg, fake_env.gmail

    out = g.invoke(fake_env.start, cfg)
    batch = out["__interrupt__"][0].value["batch"]

    # A message missing from the LLM answer falls back to the safe verdict.
    ex20 = next(m for m in batch if m["id"] == "ex20")
    assert ex20["verdict"] == "maybe_important"
    assert ex20["reason"] == nodes._FALLBACK_REASON

    out = g.invoke(Command(resume=_default_decisions(batch, stop=True)), cfg)

    assert "__interrupt__" not in out                    # no second review screen
    state = g.get_state(cfg)
    assert state.next == ()
    assert state.values["stop"] is True
    assert state.values["has_more"] is True              # messages remain, but the user stopped
    assert gmail.fetch_calls == 1
    assert len(gmail.boxes) == 10                        # the second batch was never touched
    assert state.values["stats"] == {"trashed": 11, "inbox": 9, "skipped": 0, "errors": 0}


# ---------- 7.9 test_empty_mailbox_ends_without_review ----------

def test_empty_mailbox_ends_without_review(fake_env):
    fake_env.gmail.boxes.clear()
    g, cfg = fake_env.graph, fake_env.cfg

    out = g.invoke(fake_env.start, cfg)

    assert "__interrupt__" not in out
    assert g.get_state(cfg).next == ()
    assert out["batch"] == []
    assert fake_env.llm.calls == []                      # classify never ran
    assert fake_env.gmail.trashed == [] and fake_env.gmail.moved == []
