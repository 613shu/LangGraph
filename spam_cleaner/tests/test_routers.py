"""Unit tests for the two conditional-edge routers in nodes.py."""
from nodes import route_empty, route_more


# ---------- 7.5 TestRouters ----------

class TestRouters:
    def test_route_empty_with_messages(self):
        assert route_empty({"batch": [{"id": "1"}]}) == "classify"

    def test_route_empty_without_messages(self):
        assert route_empty({"batch": []}) == "empty"
        assert route_empty({}) == "empty"

    def test_route_more_loops_when_messages_remain(self):
        assert route_more({"has_more": True, "stop": False}) == "more"

    def test_route_more_done_when_nothing_left(self):
        assert route_more({"has_more": False, "stop": False}) == "done"
        assert route_more({}) == "done"

    def test_route_more_stop_wins_even_with_more(self):
        assert route_more({"has_more": True, "stop": True}) == "done"
