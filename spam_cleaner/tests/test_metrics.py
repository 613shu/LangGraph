"""Unit tests for the summary evaluators in eval/run_eval.py."""
import pytest

from run_eval import false_alarm_rate, important_recall

IMP, JUNK = {"verdict": "maybe_important"}, {"verdict": "junk"}


# ---------- 7.6 TestMetrics ----------

class TestMetrics:
    def test_perfect_classifier(self):
        refs = [IMP, IMP, JUNK, JUNK]
        assert important_recall(refs, refs) == {"key": "important_recall", "score": 1.0}
        assert false_alarm_rate(refs, refs) == {"key": "false_alarm_rate", "score": 0.0}

    def test_mixed_results(self):
        refs = [IMP, IMP, IMP, IMP, JUNK, JUNK]
        outs = [IMP, IMP, IMP, JUNK, IMP, JUNK]        # 1 important missed, 1 false alarm
        assert important_recall(outs, refs)["score"] == pytest.approx(0.75)
        assert false_alarm_rate(outs, refs)["score"] == pytest.approx(0.5)

    def test_flag_everything(self):
        refs = [IMP, JUNK, JUNK]
        outs = [IMP, IMP, IMP]
        assert important_recall(outs, refs)["score"] == 1.0
        assert false_alarm_rate(outs, refs)["score"] == 1.0

    def test_missing_output_counts_as_not_flagged(self):
        refs = [IMP, JUNK]
        outs = [None, None]                            # the target crashed on both
        assert important_recall(outs, refs)["score"] == 0.0
        assert false_alarm_rate(outs, refs)["score"] == 0.0

    def test_no_examples_of_a_class(self):
        assert important_recall([JUNK], [JUNK])["score"] == 1.0
        assert false_alarm_rate([IMP], [IMP])["score"] == 0.0
