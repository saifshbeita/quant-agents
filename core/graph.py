from langgraph.graph import StateGraph, END
import pandas as pd
from core.state import AgentState
from core.sandbox import execute_strategy
from agents.ideator import ideator_agent
from agents.coder import coder_agent
from agents.auditor import auditor_agent
from agents.packager import packager_agent
from config import MAX_ITERATIONS

def build_graph(market_df: pd.DataFrame):
    
    def sandbox_node(state: AgentState) -> dict:
        code_str = state.get("current_code", "")
        if not code_str:
            return {"error_logs": "No code provided to the sandbox.", "backtest_metrics": {}}
            
        result = execute_strategy(code_str, market_df)
        
        if result["success"]:
            return {
                "backtest_metrics": result["metrics"],
                "error_logs": ""
            }
        else:
            return {
                "backtest_metrics": {},
                "error_logs": result["error"]
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
    
    def auditor_router(state: AgentState):
        status = state.get("status")
        if status == "passed":
            return "packager"
        elif state.get("iteration_count", 0) >= MAX_ITERATIONS:
            return "ideator"
        else:
            return "coder"
            
    workflow.add_conditional_edges(
        "auditor",
        auditor_router,
        {
            "packager": "packager",
            "ideator": "ideator",
            "coder": "coder"
        }
    )
    
    workflow.add_edge("packager", END)
    
    app = workflow.compile()
    return app
