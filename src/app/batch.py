"""The context-window batch run (specification section 4.5).

    python run.py batch                         strategy on
    python run.py batch --no-context-strategy   strategy off (whole transcript kept)
    python run.py batch --compare               both runs, and write docs/metrics.md

The specification describes "10 invoices processed sequentially in one thread", carried over
from another assignment. For SilverFile the 10 items are 10 tax returns, run one after another
through the full graph (alternating eval case 1 and eval case 2), as if one model-calling agent
worked through a preparer's queue in a single conversation.

For each item the batch records the input context that agent would receive at the item's last
turn, built by ``memory.build_context``:

* **strategy on (selective recall):** the current return's structured answers, a one-line
  summary of every earlier decision in the batch, open escalations, and only the last
  ``DEFAULT_RECENT_TURNS`` turns verbatim;
* **strategy off:** the same, but with every turn of every return so far kept verbatim.

Token counts use ``memory.estimate_tokens`` (about four characters per token): no agent in this
system sends the transcript to a model, so there is no provider token count to read. The
conversation itself is real: each question SilverFile asked and each answer given.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from app import session as sessions
from app.documents import load_forms_file, register_path
from memory import DEFAULT_RECENT_TURNS, build_context, estimate_tokens

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "demo"
ITEMS = 10
CASES = [("eval_case_1", "eval_case_1_answers.txt"), ("eval_case_2", "eval_case_2_answers.txt")]


def _run_return(graph, session_id: str, answers: list[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Run one return through the full graph. Returns its transcript and final status."""
    documents = [register_path(session_id, DEMO / "w2_image_1.jpg", [])]
    entered, amounts = load_forms_file(json.loads((DEMO / "eval_case_1_forms.json").read_text()))
    status = sessions.start(graph, session_id, documents=documents, entered_documents=entered,
                            entered_amounts=amounts)
    transcript: list[dict[str, str]] = []
    queue = list(answers)
    while status["state"] == "question" and queue:
        q = status["question"]
        text = queue.pop(0)
        transcript.append({"role": "assistant", "content": q["question"]})
        transcript.append({"role": "user", "content": text})
        status = sessions.answer(graph, session_id, q["question_id"], text)
    return transcript, status


def _decision(item: int, case: str, status: dict[str, Any]) -> str:
    values = status.get("values") or {}
    lines = ((values.get("draft_return") or {}).get("lines") or {})
    refund = (lines.get("34") or {}).get("value")
    if status["state"] == "preparer":
        issues = "; ".join(i["title"] for i in status["packet"]["issues"])
        return f"Return {item} ({case}): paused for a preparer ({issues})"
    return f"Return {item} ({case}): {values.get('run_status')}, refund ${refund:,}" if refund is not None \
        else f"Return {item} ({case}): {values.get('run_status') or status['state']}"


def run(strategy: bool, items: int = ITEMS, db: str | None = None) -> list[dict[str, Any]]:
    """Run the batch and return one row per item."""
    db = db or str(Path(tempfile.mkdtemp()) / "batch.sqlite")
    graph = sessions.open_graph(db, offline=True)
    transcript: list[dict[str, str]] = []
    decisions: list[str] = []
    rows = []
    for item in range(1, items + 1):
        case, answers_file = CASES[(item - 1) % len(CASES)]
        answers = (DEMO / answers_file).read_text().splitlines()
        turns, status = _run_return(graph, f"batch-{'on' if strategy else 'off'}-{item:02d}", answers)
        transcript += turns
        values = status.get("values") or {}
        facts = {
            "current_return": status.get("intake_answers") or values.get("intake_answers") or {},
            "session_decisions_so_far": list(decisions),
        }
        escalations = values.get("intake_escalations") or []
        context = build_context(facts, transcript, escalations,
                                max_recent_turns=DEFAULT_RECENT_TURNS if strategy else None)
        rows.append({"item": item, "case": case, "turns_so_far": len(transcript),
                     "input_tokens": estimate_tokens(context)})
        decisions.append(_decision(item, case, status))
    return rows


def metrics_markdown(on: list[dict[str, Any]], off: list[dict[str, Any]]) -> str:
    table = "\n".join(
        f"| {a['item']} | {a['case'].replace('_', ' ')} | {b['turns_so_far']} | {b['input_tokens']:,} | {a['input_tokens']:,} |"
        for a, b in zip(on, off))
    first_off, last_off = off[0]["input_tokens"], off[-1]["input_tokens"]
    first_on, last_on = on[0]["input_tokens"], on[-1]["input_tokens"]
    return f"""# Context-window strategy metrics

Specification section 4.5. Produced by `python run.py batch --compare`.

**What the 10 items are.** The specification's "10 invoices" are carried over from another
assignment. For SilverFile they are 10 tax returns run one after another through the full graph,
alternating eval case 1 (clean run) and eval case 2 (paused for a preparer), as if one
model-calling agent worked through a preparer's queue in a single conversation.

**Strategy: selective recall** (`memory.build_context`). The agent receives the current return's
structured answers, a one-line summary of each earlier decision in the batch, open escalations,
and only the last {DEFAULT_RECENT_TURNS} turns verbatim. With `--no-context-strategy` every turn of every return
is kept.

**How tokens are counted.** `memory.estimate_tokens` (about four characters per token) on the
exact context that would be sent. No agent in SilverFile currently sends the conversation to a
model, so there is no provider count to read; the conversation itself is real (each question
SilverFile asked and each answer given).

| Item | Return | Turns so far | Input tokens, strategy off | Input tokens, strategy on |
|---|---|---|---|---|
{table}

**Result.** Without the strategy, input grows from {first_off:,} to {last_off:,} tokens
({last_off / first_off:.1f}x) as every turn is kept. With it, input goes from {first_on:,} to {last_on:,} tokens:
the only growth is the one-line summary of each earlier decision.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the context-window batch (specification 4.5).")
    parser.add_argument("--no-context-strategy", action="store_true",
                        help="keep the whole transcript instead of compacting it")
    parser.add_argument("--compare", action="store_true",
                        help="run with the strategy off and on, and write docs/metrics.md")
    parser.add_argument("--items", type=int, default=ITEMS)
    args = parser.parse_args(argv)

    if args.compare:
        off = run(strategy=False, items=args.items)
        on = run(strategy=True, items=args.items)
        out = REPO_ROOT / "docs" / "metrics.md"
        out.write_text(metrics_markdown(on, off))
        for a, b in zip(on, off):
            print(f"item {a['item']:2d}  off {b['input_tokens']:6,}  on {a['input_tokens']:6,}")
        print(f"Wrote {out.relative_to(REPO_ROOT)}")
        return
    rows = run(strategy=not args.no_context_strategy, items=args.items)
    label = "off" if args.no_context_strategy else "on"
    for r in rows:
        print(f"item {r['item']:2d}  {r['case']}  turns {r['turns_so_far']:4d}  input tokens ({label}) {r['input_tokens']:,}")


if __name__ == "__main__":
    main()
