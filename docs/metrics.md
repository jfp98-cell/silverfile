# Context-window strategy metrics

Specification section 4.5. Produced by `python run.py batch --compare`.

**What the 10 items are.** The specification's "10 invoices" are carried over from another
assignment. For SilverFile they are 10 tax returns run one after another through the full graph,
alternating eval case 1 (clean run) and eval case 2 (paused for a preparer), as if one
model-calling agent worked through a preparer's queue in a single conversation.

**Strategy: selective recall** (`memory.build_context`). The agent receives the current return's
structured answers, a one-line summary of each earlier decision in the batch, open escalations,
and only the last 6 turns verbatim. With `--no-context-strategy` every turn of every return
is kept.

**How tokens are counted.** `memory.estimate_tokens` (about four characters per token) on the
exact context that would be sent. No agent in SilverFile currently sends the conversation to a
model, so there is no provider count to read; the conversation itself is real (each question
SilverFile asked and each answer given).

| Item | Return | Turns so far | Input tokens, strategy off | Input tokens, strategy on |
|---|---|---|---|---|
| 1 | eval case 1 | 38 | 1,199 | 457 |
| 2 | eval case 2 | 76 | 2,210 | 469 |
| 3 | eval case 1 | 114 | 3,226 | 487 |
| 4 | eval case 2 | 152 | 4,237 | 500 |
| 5 | eval case 1 | 190 | 5,253 | 518 |
| 6 | eval case 2 | 228 | 6,264 | 531 |
| 7 | eval case 1 | 266 | 7,280 | 549 |
| 8 | eval case 2 | 304 | 8,291 | 562 |
| 9 | eval case 1 | 342 | 9,307 | 580 |
| 10 | eval case 2 | 380 | 10,318 | 593 |

**Result.** Without the strategy, input grows from 1,199 to 10,318 tokens
(8.6x) as every turn is kept. With it, input goes from 457 to 593 tokens:
the only growth is the one-line summary of each earlier decision.
