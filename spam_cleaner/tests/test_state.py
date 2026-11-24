"""Unit tests for the custom reducers in state.py."""
from state import merge_unique, sum_counts


# ---------- 7.4 TestReducers ----------

class TestReducers:
    def test_merge_unique_keeps_order_and_drops_duplicates(self):
        assert merge_unique(["a", "b"], ["b", "c", "a", "d"]) == ["a", "b", "c", "d"]

    def test_merge_unique_handles_none_and_empty(self):
        assert merge_unique(None, ["x"]) == ["x"]
        assert merge_unique(["x"], None) == ["x"]
        assert merge_unique(None, None) == []

    def test_merge_unique_does_not_mutate_input(self):
        left = ["a"]
        merge_unique(left, ["b"])
        assert left == ["a"]

    def test_sum_counts_adds_by_key(self):
        assert sum_counts({"trashed": 2, "inbox": 1}, {"trashed": 3, "skipped": 4}) == \
            {"trashed": 5, "inbox": 1, "skipped": 4}

    def test_sum_counts_handles_none(self):
        assert sum_counts(None, {"trashed": 1}) == {"trashed": 1}
        assert sum_counts({"inbox": 2}, None) == {"inbox": 2}

    def test_sum_counts_does_not_mutate_input(self):
        left = {"trashed": 1}
        sum_counts(left, {"trashed": 1})
        assert left == {"trashed": 1}
