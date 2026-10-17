import json

from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from core.llm import chat, strip_think


def _pct(x) -> str:
    return f"{x * 100:.2f}%"


def report_header(state: AgentState) -> str:
    """Hard numbers for the handoff report, generated in code so they can't be hallucinated."""
    m = state.get("backtest_metrics") or {}
    tr, te = m.get("train", {}), m.get("test", {})
    bh = m.get("benchmark", {})
    rows = [
        ("Sharpe", f"{tr.get('sharpe', 0):.2f}", f"{te.get('sharpe', 0):.2f}", f"{bh.get('test', {}).get('sharpe', 0):.2f}"),
        ("Max drawdown", _pct(tr.get("max_drawdown", 0)), _pct(te.get("max_drawdown", 0)), _pct(bh.get("test", {}).get("max_drawdown", 0))),
        ("Total return", _pct(tr.get("total_return", 0)), _pct(te.get("total_return", 0)), _pct(bh.get("test", {}).get("total_return", 0))),
        ("Win rate (active days)", _pct(tr.get("win_rate", 0)), _pct(te.get("win_rate", 0)), "-"),
        ("Exposure", _pct(tr.get("exposure", 0)), _pct(te.get("exposure", 0)), "100.00%"),
        ("Trades", str(tr.get("trades", 0)), str(te.get("trades", 0)), "-"),
    ]
    table = "| Metric | Train | Test (OOS) | Buy & hold (OOS) |\n|---|---|---|---|\n"
    table += "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
    gates = state.get("gates") or {}
    gate_md = "\n".join(f"- {'✅' if g['passed'] else '❌'} {g['label']}" for g in gates.values())
    return (
        f"# Strategy Handoff: {state.get('idea_title', 'Untitled')}\n\n"
        f"**Ticker:** {state.get('ticker', '')} · **OOS split from:** {m.get('split_date')} · "
        f"**Costs:** {m.get('cost_bps', 1)}bp per unit turnover · "
        f"**Idea {state.get('idea_count', 1)}, iteration {state.get('iteration_count', 1)}**\n\n"
        f"## Validation Gates\n{gate_md}\n\n## Backtest Metrics\n{table}\n\n"
        f"## Market Theory\n{state.get('market_theory', '')}\n\n---\n\n"
    )


def packager_agent(state: AgentState) -> dict:

    system_prompt = """You are the Lead Execution Engineer. Your job is to take a validated, profitable Python prototype and format it for the C++ high-frequency trading team.
Take the validated `current_code` and generate a comprehensive Markdown report containing:
1. Production Code: The finalized Python script with thorough algorithmic docstrings.
2. Complexity Analysis: Big-O time and space complexity of the signal generation.
3. Execution Edge Cases: Explicit warnings on slippage assumptions, missing tick data handling, and corporate action impacts.
4. C++ Translation Architecture: Directives for translating the specific pandas vectorization into optimized C++ (e.g., Eigen or Armadillo library recommendations).
The metrics table and gate results are already in the report header; do not restate or invent numbers. Start directly with section 1."""

    human_content = (
        f"Validated Python Code:\n```python\n{state.get('current_code')}\n```\n\n"
        f"Backtest Metrics:\n{json.dumps(state.get('backtest_metrics', {}), default=str)}\n\n"
        "Please generate the final Markdown report."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]

    body = strip_think(chat(messages, node="packager"))

    return {
        "final_report": report_header(state) + body,
        "status": "packaged"
    }
