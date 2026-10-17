from langgraph.graph import StateGraph, END
import pandas as pd
from core.state import AgentState
from core.sandbox import execute_strategy
from agents.ideator import ideator_agent
from agents.coder import coder_agent
from agents.auditor import auditor_agent
from agents.packager import packager_agent
from config import MAX_IDEAS


def route_after_auditor(state: AgentState) -> str:
    """passed -> packager; still iterating -> coder; idea exhausted -> new idea, or END once MAX_IDEAS is reached."""
    status = state.get("status")
    if status == "passed":
        return "packager"
    if status == "failed":
        if state.get("idea_count", 0) >= (state.get("max_ideas") or MAX_IDEAS):
            return END
        return "ideator"
    return "coder"


def build_graph(market_df: pd.DataFrame):

    def sandbox_node(state: AgentState) -> dict:
        code_str = state.get("current_code", "")
        if not code_str:
            return {"error_logs": "No code provided to the sandbox.", "backtest_metrics": {},
                    "equity_curve": {}, "lookahead": {}}

        result = execute_strategy(code_str, market_df)
        return {
            "backtest_metrics": result["metrics"] or {},
            "equity_curve": result["equity"] or {},
            "lookahead": result["lookahead"],
            "error_logs": result["error"] or "",
        }

    workflow = StateGraph(AgentState)

    workflow.add_node("ideator", ideator_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("sandbox", sandbox_node)
    workflow.add_node("auditor", auditor_agent)
    workflow.add_node("packager", packager_agent)

    workflow.set_entry_point("ideator")

    workflow.add_edge("ideator", "coder")
    workflow.add_edge("coder", "sandbox")
    workflow.add_edge("sandbox", "auditor")

    workflow.add_conditional_edges(
        "auditor",
        route_after_auditor,
        {
            "packager": "packager",
            "ideator": "ideator",
            "coder": "coder",
            END: END,
        }
    )

    workflow.add_edge("packager", END)

    return workflow.compile()
