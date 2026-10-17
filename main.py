import os
import pandas as pd
from core.graph import build_graph
from data.fetcher import get_historical_data
from config import TARGET_TICKER, MAX_IDEAS, MAX_ITERATIONS, RECURSION_LIMIT, get_thresholds

def main():
    print(f"Loading historical data for {TARGET_TICKER}...")
    market_df = get_historical_data(TARGET_TICKER)
    print(f"Loaded {len(market_df)} trading days of data.\n")
    
    initial_state = {
        "market_theory": "",
        "current_code": "",
        "backtest_metrics": {},
        "error_logs": "",
        "auditor_feedback": "",
        "iteration_count": 0,
        "idea_count": 0,
        "tried_ideas": [],
        "ticker": TARGET_TICKER,
        "max_ideas": MAX_IDEAS,
        "max_iterations": MAX_ITERATIONS,
        "thresholds": get_thresholds(),
        "status": "pending",
        "final_report": ""
    }
    
    app = build_graph(market_df)
    
    print("Starting Execution Graph...\n" + "="*40)
    
    for output in app.stream(initial_state, config={"recursion_limit": RECURSION_LIMIT}):
        for node_name, state_update in output.items():
            print(f"--- Node Executed: {node_name.upper()} ---")
            
            if "iteration_count" in state_update:
                print(f"Iteration Count: {state_update['iteration_count']}")
                
            if "status" in state_update:
                print(f"Status: {state_update['status']}")
                
            if "backtest_metrics" in state_update and state_update["backtest_metrics"]:
                metrics = state_update["backtest_metrics"]["test"]
                print(f"OOS Metrics -> Sharpe: {metrics.get('sharpe', 0):.2f} | Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}% | Return: {metrics.get('total_return', 0)*100:.2f}%")
                
            if "error_logs" in state_update and state_update["error_logs"]:
                print(f"Errors detected! Snippet: {str(state_update['error_logs'])[:300]}...")
                
            print("-" * 40)
            
            if node_name == "packager" and "final_report" in state_update:
                final_report = state_update["final_report"]
                
                print("\n=== FINAL STRATEGY REPORT ===")
                print(final_report)
                print("=============================")
                
                # Write to handoff file
                file_path = os.path.join(os.path.dirname(__file__), "strategy_handoff.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(final_report)
                print(f"\nReport successfully written to {file_path}")

if __name__ == "__main__":
    main()
