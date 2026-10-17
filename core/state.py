from typing import TypedDict


class AgentState(TypedDict, total=False):
    # run parameters
    ticker: str
    max_ideas: int
    max_iterations: int
    thresholds: dict  # {"sharpe", "max_drawdown", "demo_mode"}

    # idea loop
    idea_count: int
    idea_title: str
    tried_ideas: list
    market_theory: str

    # code loop
    current_code: str
    iteration_count: int
    backtest_metrics: dict  # {"train", "test", "full", "benchmark", "split_date", ...}
    equity_curve: dict
    lookahead: dict
    error_logs: str
    auditor_feedback: str
    gates: dict

    status: str  # "pending", "coding", "passed", "failed", "packaged"
    final_report: str
