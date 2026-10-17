import json

from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from core.llm import chat, strip_think
from config import MAX_ITERATIONS, get_thresholds

GATE_ORDER = ("sharpe", "max_drawdown", "lookahead", "out_of_sample")


def evaluate_gates(state: AgentState) -> dict:
    """Deterministic pass/fail: no errors AND OOS Sharpe > limit AND OOS max DD < limit AND lookahead test passes."""
    th = state.get("thresholds") or get_thresholds()
    test = (state.get("backtest_metrics") or {}).get("test") or {}
    error = state.get("error_logs") or ""
    ran = not error and bool(test)
    lookahead = state.get("lookahead") or {}

    gates = {
        "sharpe": {
            "label": f"OOS Sharpe > {th['sharpe']:.2f}",
            "passed": ran and test["sharpe"] > th["sharpe"],
            "value": test.get("sharpe") if ran else None,
        },
        "max_drawdown": {
            "label": f"OOS max drawdown < {th['max_drawdown']:.0%}",
            "passed": ran and test["max_drawdown"] < th["max_drawdown"],
            "value": test.get("max_drawdown") if ran else None,
        },
        "lookahead": {
            "label": "Lookahead test (truncated-history replay + shift(-n) scan)",
            "passed": bool(lookahead.get("passed")),
            "value": lookahead.get("message", "not run"),
        },
        "out_of_sample": {
            "label": "Out-of-sample run (held-out split, no errors)",
            "passed": ran,
            "value": f"{test.get('days', 0)} held-out days" if ran else (error.strip().splitlines() or ["no metrics"])[-1][:200],
        },
    }
    return {"passed": all(g["passed"] for g in gates.values()), "gates": gates, "thresholds": th}


def auditor_agent(state: AgentState) -> dict:
    verdict = evaluate_gates(state)
    th = verdict["thresholds"]
    max_iter = state.get("max_iterations") or MAX_ITERATIONS

    if verdict["passed"]:
        status = "passed"
    elif state.get("iteration_count", 0) >= max_iter:
        status = "failed"
    else:
        status = "coding"

    system_prompt = f"""You are the Head of Risk Management and Quantitative Auditing. You serve as the absolute gatekeeper against lookahead bias, curve-fitting, and excessive drawdown exposure.
The pass/fail verdict has ALREADY been computed deterministically by the risk engine (gates: out-of-sample Sharpe > {th['sharpe']}, out-of-sample max drawdown < {th['max_drawdown']*100:.0f}%, lookahead test, clean execution). Do not override or re-litigate it; your job is the written critique.
Your Mandate:
1. Traceback Analysis: If execution errors exist, diagnose the exact matrix shape mismatch, indexing error, or syntax fault, and provide a precise, vectorized fix.
2. Lookahead: If the lookahead test failed, identify the exact line that uses future information and give the causal replacement.
3. Performance Audit: Compare train vs. out-of-sample metrics and the buy-and-hold benchmark. Flag overfitting (large train/test gap), excessive turnover, or tail risk.
4. Anti-Overfitting: DO NOT suggest tweaking arbitrary parameters (e.g., changing a rolling window from 20 to 18). Diagnose the structural flaw (e.g., over-trading in chop, extreme tail risk) and demand regime filters, volatility-based position sizing, or volume thresholds.
5. Output: a concise, highly technical critique (bullet points, under 250 words). If the verdict is PASSED, list the residual risks the execution team must monitor."""

    gate_lines = "\n".join(
        f"- {g['label']}: {'PASS' if g['passed'] else 'FAIL'} ({g['value'] if not isinstance(g['value'], float) else round(g['value'], 4)})"
        for g in (verdict["gates"][k] for k in GATE_ORDER)
    )
    human_content = f"Risk-engine verdict: {'PASSED' if verdict['passed'] else 'FAILED'}\n{gate_lines}\n\n"
    human_content += f"Current Code:\n```python\n{state.get('current_code', 'None')}\n```\n\n"
    if state.get('error_logs'):
        human_content += f"Execution Error Logs:\n{state.get('error_logs')}\n\n"
    if state.get('backtest_metrics'):
        human_content += f"Backtest Metrics (1bp/turnover costs, train/test split):\n{json.dumps(state['backtest_metrics'], indent=1, default=str)}\n\n"
    human_content += "Please write the audit critique."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]
    feedback = strip_think(chat(messages, node="auditor"))

    return {
        "auditor_feedback": feedback,
        "gates": verdict["gates"],
        "status": status,
    }
