"""The context-window batch (specification 4.5): without the strategy the context grows with
every turn; with it the context stays nearly flat."""

from app.batch import metrics_markdown, run


def test_strategy_keeps_context_flat_while_off_grows():
    off = run(strategy=False, items=4)
    on = run(strategy=True, items=4)
    off_growth = off[-1]["input_tokens"] - off[0]["input_tokens"]
    on_growth = on[-1]["input_tokens"] - on[0]["input_tokens"]
    assert off_growth > 2 * off[0]["input_tokens"]  # roughly linear in the number of turns
    assert on_growth < off_growth / 10  # materially flatter
    assert all(a["input_tokens"] < b["input_tokens"] for a, b in zip(on, off))
    assert "| 4 |" in metrics_markdown(on, off)
