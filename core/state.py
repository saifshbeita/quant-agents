from typing import TypedDict

class AgentState(TypedDict):
    market_theory: str
    current_code: str
    backtest_metrics: dict
    error_logs: str
    auditor_feedback: str
    iteration_count: int
    status: str # "coding", "auditing", "packaged", "failed"
    final_report: str
