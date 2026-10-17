import argparse
import os
import sys

from core.runner import run_pipeline
from config import TARGET_TICKER, MAX_IDEAS, DEMO_MODE


def print_event(ev: dict) -> None:
    node, update = ev["node"], ev["payload"]
    tok = ev["tokens"]
    if node == "__start__":
        d = update["data"]
        print(f"Run {update['run_id']} | model {update['model']} @ {update['endpoint']}")
        print(f"Loaded {d['rows']} trading days of {update['ticker']} ({d['start']} -> {d['end']}, {d['source']})")
        print(f"Gates: OOS Sharpe > {update['thresholds']['sharpe']}, OOS max DD < {update['thresholds']['max_drawdown']:.0%}"
              f"{' [DEMO_MODE]' if update['demo_mode'] else ''}\n" + "=" * 40)
        return
    if node == "__end__":
        print(f"\nRun finished: status={update['status']} passed={update['passed']} ideas={update['ideas_tried']} "
              f"llm_calls={update['total_llm_calls']} tokens={update['total_total']} "
              f"({update['tokens_per_s']} tok/s) in {update['wall_time_s']}s")
        return
    if node == "__error__":
        print(f"!!! Run aborted: {update['error']}")
        return

    print(f"--- Node Executed: {node.upper()} ({ev['latency_s']:.1f}s, {tok['total']} tokens) ---")
    if node == "ideator":
        print(f"Idea #{update.get('idea_count')}: {update.get('idea_title')}")
    if "iteration_count" in update and node == "coder":
        print(f"Iteration Count: {update['iteration_count']}")
    if "status" in update:
        print(f"Status: {update['status']}")
    if update.get("backtest_metrics"):
        for split in ("train", "test"):
            m = update["backtest_metrics"][split]
            print(f"{split.upper():5s} -> Sharpe: {m['sharpe']:.2f} | Max Drawdown: {m['max_drawdown']*100:.2f}% | "
                  f"Return: {m['total_return']*100:.2f}%")
    if update.get("lookahead"):
        print(f"Lookahead: {'PASS' if update['lookahead'].get('passed') else 'FAIL'} - {update['lookahead'].get('message', '')[:200]}")
    if update.get("error_logs"):
        print(f"Errors detected! Snippet: {str(update['error_logs'])[-300:]}")
    if update.get("gates"):
        print("Gates: " + ", ".join(f"{k}={'PASS' if g['passed'] else 'FAIL'}" for k, g in update["gates"].items()))
    print("-" * 40)

    if node == "packager" and update.get("final_report"):
        final_report = update["final_report"]
        print("\n=== FINAL STRATEGY REPORT ===")
        print(final_report)
        print("=============================")
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_handoff.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\nReport successfully written to {file_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # e.g. Windows cp125x consoles vs. emoji in the report
    parser = argparse.ArgumentParser(description="Run the quant-agents research pipeline.")
    parser.add_argument("--ticker", default=TARGET_TICKER)
    parser.add_argument("--start", help="YYYY-MM-DD (default: last 2 years)")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--max-ideas", type=int, default=MAX_IDEAS)
    parser.add_argument("--demo", action="store_true", default=DEMO_MODE, help="use looser DEMO_MODE gates")
    args = parser.parse_args()

    params = {"ticker": args.ticker, "start": args.start, "end": args.end,
              "max_ideas": args.max_ideas, "demo_mode": args.demo}
    for ev in run_pipeline(params):
        print_event(ev)


if __name__ == "__main__":
    main()
