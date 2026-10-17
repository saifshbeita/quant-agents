"""
Streaming pipeline runner shared by the CLI (main.py) and the Streamlit app.

run_pipeline(params) yields one event per graph node:
    {run_id, seq, node, payload, ts, latency_s, tokens}
plus bookend events "__start__" (run metadata) and "__end__" (summary), or "__error__".
Every event is appended to runs/<run_id>.jsonl so a run can be replayed later.
"""
import datetime as dt
import glob
import json
import os
import time
import uuid
from typing import Iterator

import numpy as np
import pandas as pd

import config
from core import llm


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return obj.isoformat()
    return str(obj)


def _tokens(calls: list[dict]) -> dict:
    prompt = sum(c["prompt_tokens"] for c in calls)
    completion = sum(c["completion_tokens"] for c in calls)
    return {
        "prompt": prompt,
        "completion": completion,
        "total": prompt + completion,
        "llm_calls": len(calls),
        "llm_latency_s": round(sum(c["latency_s"] for c in calls), 3),
    }


def new_run_id(ticker: str) -> str:
    return f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{ticker.upper()}-{uuid.uuid4().hex[:6]}"


def run_pipeline(params: dict | None = None) -> Iterator[dict]:
    """
    params (all optional): ticker, start, end, period, max_ideas, max_iterations, demo_mode,
    run_id, runs_dir, market_df (pre-loaded DataFrame, mainly for tests).
    """
    from core.graph import build_graph  # deferred: pulls in langgraph and the agents
    from data.fetcher import get_historical_data

    params = dict(params or {})
    ticker = (params.get("ticker") or config.TARGET_TICKER).upper()
    max_ideas = int(params.get("max_ideas") or config.MAX_IDEAS)
    max_iterations = int(params.get("max_iterations") or config.MAX_ITERATIONS)
    demo_mode = bool(params.get("demo_mode", config.DEMO_MODE))
    thresholds = config.get_thresholds(demo_mode)
    run_id = params.get("run_id") or new_run_id(ticker)
    runs_dir = params.get("runs_dir") or config.RUNS_DIR
    os.makedirs(runs_dir, exist_ok=True)
    log_path = os.path.join(runs_dir, f"{run_id}.jsonl")

    seq = 0
    all_calls: list[dict] = []
    t_run = time.perf_counter()
    t_last = t_run

    def event(node: str, payload: dict, calls: list[dict] | None = None) -> dict:
        nonlocal seq, t_last
        now = time.perf_counter()
        ev = {
            "run_id": run_id,
            "seq": seq,
            "node": node,
            "payload": payload,
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
            "latency_s": round(now - t_last, 3),
            "tokens": _tokens(calls or []),
        }
        if calls:
            ev["tokens"]["calls"] = calls
        seq += 1
        t_last = now
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=_jsonable) + "\n")
        return ev

    def summary(status: str, final: dict, error: str | None = None) -> dict:
        tok = _tokens(all_calls)
        gen_time = sum(c["latency_s"] for c in all_calls)
        return {
            "status": status,
            "passed": final.get("status") == "packaged",
            "ideas_tried": final.get("idea_count", 0),
            "wall_time_s": round(time.perf_counter() - t_run, 2),
            "tokens_per_s": round(tok["completion"] / gen_time, 1) if gen_time else 0.0,
            "avg_llm_latency_s": round(gen_time / len(all_calls), 2) if all_calls else 0.0,
            **{f"total_{k}": v for k, v in tok.items()},
            "error": error,
        }

    llm.drain_calls()  # discard anything recorded outside this run
    final: dict = {}
    try:
        df = params.get("market_df")
        if df is None:
            df = get_historical_data(ticker, start=params.get("start"), end=params.get("end"),
                                     period=params.get("period", "2y"))
        yield event("__start__", {
            "run_id": run_id,
            "ticker": ticker,
            "max_ideas": max_ideas,
            "max_iterations": max_iterations,
            "demo_mode": demo_mode,
            "thresholds": thresholds,
            "model": llm.model_label(),
            "endpoint": llm.endpoint_label(),
            "data": {
                "source": df.attrs.get("source", "provided"),
                "rows": len(df),
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end": df.index[-1].strftime("%Y-%m-%d"),
            },
            "cost_bps": config.TRANSACTION_COST_BPS,
            "train_fraction": config.TRAIN_FRACTION,
        })

        initial_state = {
            "ticker": ticker,
            "max_ideas": max_ideas,
            "max_iterations": max_iterations,
            "thresholds": thresholds,
            "idea_count": 0,
            "tried_ideas": [],
            "iteration_count": 0,
            "market_theory": "",
            "current_code": "",
            "backtest_metrics": {},
            "error_logs": "",
            "auditor_feedback": "",
            "status": "pending",
            "final_report": "",
        }
        # ideator + (coder, sandbox, auditor) per iteration, per idea, + packager, with headroom
        needed = max_ideas * (1 + 3 * max_iterations) + 5
        graph = build_graph(df)
        for update in graph.stream(initial_state, config={"recursion_limit": max(config.RECURSION_LIMIT, needed)}):
            for node, payload in update.items():
                payload = payload or {}
                final.update(payload)
                calls = llm.drain_calls()
                all_calls.extend(calls)
                yield event(node, payload, calls)
    except Exception as e:
        calls = llm.drain_calls()
        all_calls.extend(calls)
        err = f"{type(e).__name__}: {e}"
        yield event("__error__", {"error": err}, calls)
        yield event("__end__", summary("error", final, err))
        return

    yield event("__end__", summary(final.get("status", "unknown"), final))


def load_run(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_runs(runs_dir: str | None = None) -> list[dict]:
    """Recorded runs, newest first: [{path, run_id, model, ticker, status, passed}]."""
    runs = []
    for path in glob.glob(os.path.join(runs_dir or config.RUNS_DIR, "*.jsonl")):
        try:
            events = load_run(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not events:
            continue
        start = next((e["payload"] for e in events if e["node"] == "__start__"), {})
        end = next((e["payload"] for e in events if e["node"] == "__end__"), {})
        runs.append({
            "path": path,
            "run_id": events[0].get("run_id", os.path.basename(path)),
            "model": start.get("model", "?"),
            "ticker": start.get("ticker", "?"),
            "status": end.get("status", "incomplete"),
            "passed": end.get("passed", False),
            "mtime": os.path.getmtime(path),
        })
    return sorted(runs, key=lambda r: r["mtime"], reverse=True)
