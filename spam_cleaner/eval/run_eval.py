"""LangSmith evaluation: v1 vs v2 classifier prompts on the spam-cleaner dataset.

Run from spam_cleaner/:  python eval/run_eval.py
(create the dataset first with: python eval/dataset.py)
"""
# Use the Windows certificate store (needed behind filtered / inspected networks).
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

from langchain.chat_models import init_chat_model
from langsmith import evaluate
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `import nodes` work

import nodes  # noqa: E402

SYSTEM_PROMPT_V1 = nodes.SYSTEM_PROMPT  # the current prompt, kept as is

SYSTEM_PROMPT_V2 = """You classify Gmail messages from the Promotions and Spam folders.
Each input line is a JSON object with: id, label, sender, subject, snippet, headers.
For each message return one verdict:

- "maybe_important" – anything the user would not want to lose:
  * a mailing-list / Google Groups message: the headers contain "List-Id"
    (this means the user is a member of that list, even if it looks like bulk mail);
  * a receipt, invoice, order confirmation or shipping confirmation from a real shop;
  * a personal message written by a real person to the user (often lands in Spam);
  * an official notice from a bank, utility, government office or school.
- "junk" – everything else:
  * marketing newsletters and sales: "List-Unsubscribe" and/or "Precedence: bulk"
    WITHOUT a "List-Id" header;
  * spam, phishing, prize / crypto / fake-delivery scams;
  * ads that pretend to be personal ("Hi, just wanted to update you…") but are
    selling something – they are still junk.

Rules: copy each id exactly as given. Write a short reason in Hebrew for every verdict,
naming the signal you used (for example the List-Id header, or "an ad disguised as personal").

Examples:
{"id": "a1", "sender": "Chess Club <chess-club@googlegroups.com>", "subject": "Next meeting", "snippet": "Hi all, we meet Sunday at 18:00...", "headers": {"List-Id": "<chess-club.googlegroups.com>", "Precedence": "list"}}
-> maybe_important: "הודעה מקבוצת Google שאת חברה בה (כותרת List-Id)."
{"id": "a2", "sender": "Mega Sale <sale@megasale.com>", "subject": "Only today 60% off", "snippet": "Huge discounts on everything...", "headers": {"List-Unsubscribe": "<mailto:u@megasale.com>", "Precedence": "bulk"}}
-> junk: "דיוור שיווקי עם List-Unsubscribe וללא List-Id."
{"id": "a3", "sender": "Ronit <ronit@cheap-loans.net>", "subject": "Hi, quick update", "snippet": "Hi! A special loan was approved for you, click here...", "headers": {}}
-> junk: "פרסומת שמתחפשת להודעה אישית ומציעה הלוואה."
"""


# ---------- 6.2 make_target ----------

def make_target(prompt: str) -> Callable[[dict], dict]:
    """Return a LangSmith target that classifies one message with the given system prompt."""
    def target(inputs: dict) -> dict:
        nodes.SYSTEM_PROMPT = prompt  # classify() reads the module global at call time
        msg = nodes.classify({"batch": [inputs["message"]]})["batch"][0]
        return {"verdict": msg["verdict"], "reason": msg["reason"]}

    target.__name__ = "classify_v1" if prompt == SYSTEM_PROMPT_V1 else "classify_v2"
    return target


# ---------- 6.3 important_recall ----------

def important_recall(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """Share of truly important messages that the classifier flagged as maybe_important."""
    pairs = [(o, r) for o, r in zip(outputs, reference_outputs) if r["verdict"] == "maybe_important"]
    hits = sum(1 for o, _ in pairs if (o or {}).get("verdict") == "maybe_important")
    score = hits / len(pairs) if pairs else 1.0
    return {"key": "important_recall", "score": score}


# ---------- 6.4 false_alarm_rate ----------

def false_alarm_rate(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """Share of junk messages that the classifier wrongly flagged as maybe_important (lower is better)."""
    pairs = [(o, r) for o, r in zip(outputs, reference_outputs) if r["verdict"] == "junk"]
    alarms = sum(1 for o, _ in pairs if (o or {}).get("verdict") == "maybe_important")
    score = alarms / len(pairs) if pairs else 0.0
    return {"key": "false_alarm_rate", "score": score}


# ---------- 6.5 reason_quality (LLM-as-judge) ----------

class ReasonGrade(BaseModel):
    score: Literal[0, 1] = Field(description="1 = the reason is relevant, accurate and supports the correct verdict")
    comment: str = Field(description="One short sentence explaining the grade")


JUDGE_PROMPT = """You grade the reason that an email classifier gave for its verdict.
You get the message (sender, subject, snippet, headers), the correct verdict and message kind,
and the classifier's verdict and reason (the reason is in Hebrew).
Give score 1 only if ALL of these hold:
- the classifier's verdict equals the correct verdict;
- the reason refers to details that really appear in the message (no invented facts);
- the reason is specific and actually supports the verdict (not a generic phrase).
Otherwise give score 0. Write the comment in English, one short sentence."""


@lru_cache(maxsize=1)
def _get_judge():
    """Build the judge LLM once, bound to the ReasonGrade schema."""
    return init_chat_model(nodes.MODEL, temperature=0).with_structured_output(ReasonGrade)


def reason_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: is the classifier's reason relevant, accurate and consistent with the message?"""
    msg = inputs["message"]
    case = {
        "message": {k: msg.get(k) for k in ("sender", "subject", "snippet", "headers")},
        "correct_verdict": reference_outputs["verdict"],
        "message_kind": reference_outputs.get("kind"),
        "classifier_verdict": (outputs or {}).get("verdict"),
        "classifier_reason": (outputs or {}).get("reason"),
    }
    grade = _get_judge().invoke([("system", JUDGE_PROMPT),
                                 ("human", json.dumps(case, ensure_ascii=False))])
    if grade is None:
        return {"key": "reason_quality", "score": 0, "comment": "judge returned no grade"}
    return {"key": "reason_quality", "score": grade.score, "comment": grade.comment}


# ---------- 6.6 main ----------

EXPERIMENTS = [("v1-baseline", SYSTEM_PROMPT_V1),
               ("v2-headers-fewshot", SYSTEM_PROMPT_V2)]


def main() -> None:
    """Run the v1 and v2 experiments on the dataset and print where to compare them."""
    from dataset import DATASET_NAME  # sibling file in eval/

    # Sequential on purpose: make_target swaps a module global in `nodes`.
    for prefix, prompt in EXPERIMENTS:
        results = evaluate(
            make_target(prompt),
            data=DATASET_NAME,
            evaluators=[reason_quality],
            summary_evaluators=[important_recall, false_alarm_rate],
            experiment_prefix=prefix,
            metadata={"prompt_version": prefix, "model": nodes.MODEL},
            max_concurrency=2,
        )

        rows = list(results)
        outs = [(r["run"].outputs or {}) for r in rows]
        refs = [r["example"].outputs for r in rows]
        recall = important_recall(outs, refs)["score"]
        alarms = false_alarm_rate(outs, refs)["score"]
        print(f"{results.experiment_name}: important_recall={recall:.2f}, "
              f"false_alarm_rate={alarms:.2f}")

    print(f"Compare in LangSmith: Datasets -> {DATASET_NAME} -> Compare experiments")


if __name__ == "__main__":
    main()
