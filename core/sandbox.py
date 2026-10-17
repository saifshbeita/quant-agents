"""
Backtest harness for AI-generated strategies.

The untrusted code runs in a separate Python process (core/sandbox_worker.py) with a hard
timeout; this is process isolation for robustness, not a security boundary. Everything that
decides pass/fail (positions, costs, metrics, the lookahead test) is computed here.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import config

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_worker.py")
TRADING_DAYS = 252
N_LOOKAHEAD_CHECKS = 8
NEG_SHIFT_RE = re.compile(r"\.shift\(\s*(?:periods\s*=\s*)?-")


def static_lookahead_flags(code: str) -> list[str]:
    """Regex scan for negative shifts, which pull future bars into the current row."""
    flags = []
    for lineno, line in enumerate(code.splitlines(), 1):
        if NEG_SHIFT_RE.search(line.split("#", 1)[0]):
            flags.append(f"line {lineno}: `{line.strip()}` uses shift(-n), which reads future bars")
    return flags


def lookahead_points(n: int, split_idx: int) -> list[int]:
    """Prefix lengths k at which generate_signals(df.iloc[:k]) is re-run."""
    if n < 3:
        return []
    lo = min(max(30, int(n * 0.3)), n - 1)
    ks = set(np.linspace(lo, n - 1, N_LOOKAHEAD_CHECKS).astype(int).tolist())
    if lo <= split_idx <= n - 1:
        ks.add(split_idx)
    return sorted(ks)


def _metrics(r: pd.Series, pos: pd.Series, turnover: pd.Series) -> dict:
    n = len(r)
    if n == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "ann_return": 0.0,
                "win_rate": 0.0, "exposure": 0.0, "trades": 0, "ann_turnover": 0.0, "days": 0}
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1)
    std = r.std()
    sharpe = float(r.mean() / std * np.sqrt(TRADING_DAYS)) if np.isfinite(std) and std > 0 else 0.0
    peak = equity.cummax().clip(lower=1.0)  # starting capital counts as a peak
    max_dd = float(abs((equity / peak - 1).min()))
    active = pos != 0
    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total,
        "ann_return": float((1 + total) ** (TRADING_DAYS / n) - 1) if total > -1 else -1.0,
        "win_rate": float((r[active] > 0).mean()) if active.any() else 0.0,
        "exposure": float(active.mean()),
        "trades": int((turnover > 0).sum()),
        "ann_turnover": float(turnover.sum() / n * TRADING_DAYS),
        "days": int(n),
    }


def compute_backtest(signals: pd.Series, df: pd.DataFrame, cost_bps: float | None = None,
                     train_fraction: float | None = None) -> tuple[dict, dict]:
    """Turn raw signals into positions/returns; return (metrics, equity_curve)."""
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    train_fraction = config.TRAIN_FRACTION if train_fraction is None else train_fraction

    target = signals.reindex(df.index).fillna(0.0).clip(-1.0, 1.0)
    position = target.shift(1).fillna(0.0)  # signal at close t is traded over bar t+1
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    mkt = df["Market_Return"]
    strat = position * mkt - turnover * cost_bps / 1e4

    split_idx = int(len(df) * train_fraction)
    tr, te = slice(0, split_idx), slice(split_idx, len(df))
    ones = pd.Series(1.0, index=df.index)
    no_turn = pd.Series(0.0, index=df.index)
    metrics = {
        "train": _metrics(strat.iloc[tr], position.iloc[tr], turnover.iloc[tr]),
        "test": _metrics(strat.iloc[te], position.iloc[te], turnover.iloc[te]),
        "full": _metrics(strat, position, turnover),
        "benchmark": {
            "train": _metrics(mkt.iloc[tr], ones.iloc[tr], no_turn.iloc[tr]),
            "test": _metrics(mkt.iloc[te], ones.iloc[te], no_turn.iloc[te]),
        },
        "split_date": df.index[split_idx].strftime("%Y-%m-%d") if split_idx < len(df) else None,
        "train_fraction": train_fraction,
        "cost_bps": cost_bps,
    }
    equity = {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "strategy": np.round((1 + strat).cumprod().values, 6).tolist(),
        "buy_hold": np.round((1 + mkt).cumprod().values, 6).tolist(),
        "position": np.round(position.values, 4).tolist(),
        "split_date": metrics["split_date"],
    }
    return metrics, equity


def check_lookahead(signals: pd.Series, prefix: list[dict], df: pd.DataFrame, flags: list[str]) -> dict:
    """Compare the last signal of each prefix run with the full-data signal at the same bar."""
    full = signals.fillna(0.0)
    mismatches = []
    for p in prefix:
        k = p["k"]
        date = df.index[k - 1].strftime("%Y-%m-%d")
        if "error" in p:
            mismatches.append({"k": k, "date": date, "error": p["error"]})
            continue
        prefix_val = 0.0 if p["value"] is None else p["value"]
        full_val = float(full.iloc[k - 1])
        if not np.isclose(prefix_val, full_val, rtol=1e-6, atol=1e-8):
            mismatches.append({"k": k, "date": date, "prefix_value": prefix_val, "full_value": full_val})

    lines = []
    if mismatches:
        m = mismatches[0]
        if "error" in m:
            lines.append(f"generate_signals crashed when given only the first {m['k']} rows "
                         f"(data up to {m['date']}): {m['error']}")
        else:
            lines.append(
                f"Signal on {m['date']} (bar {m['k'] - 1}) is {m['full_value']:.6g} when computed on the full "
                f"history but {m['prefix_value']:.6g} when computed only on data up to {m['date']}. "
                f"The signal depends on future rows ({len(mismatches)}/{len(prefix)} checkpoints differ). "
                "Typical causes: shift(-n), center=True rolling windows, full-sample statistics "
                "(df.mean(), df.std(), max/min over the whole series), bfill, or fitting on all data."
            )
    lines += [f"Static check: {f}" for f in flags]
    return {
        "passed": not mismatches and not flags,
        "checkpoints": [p["k"] for p in prefix],
        "mismatches": mismatches,
        "static_flags": flags,
        "message": " ".join(lines) if lines else f"Signals identical on {len(prefix)} truncated-history checkpoints.",
    }


def execute_strategy(code_str: str, df: pd.DataFrame, timeout_s: float | None = None) -> dict:
    """
    Run generate_signals in a subprocess and evaluate it.

    Returns {success, error, metrics, lookahead, equity}. `success`/`error` describe whether the
    code ran; the lookahead verdict is reported separately in `lookahead`.
    """
    timeout_s = config.SANDBOX_TIMEOUT_S if timeout_s is None else timeout_s
    flags = static_lookahead_flags(code_str)
    split_idx = int(len(df) * config.TRAIN_FRACTION)
    ks = lookahead_points(len(df), split_idx)
    not_run = {"passed": False, "checkpoints": [], "mismatches": [], "static_flags": flags,
               "message": "Lookahead test not run because the code failed to execute."}

    def fail(msg: str) -> dict:
        return {"success": False, "error": msg, "metrics": None, "lookahead": not_run, "equity": None}

    with tempfile.TemporaryDirectory(prefix="qa_sandbox_") as work:
        with open(os.path.join(work, "code.py"), "w", encoding="utf-8") as f:
            f.write(code_str)
        with open(os.path.join(work, "prefix.json"), "w", encoding="utf-8") as f:
            json.dump(ks, f)
        df.to_pickle(os.path.join(work, "df.pkl"))
        try:
            proc = subprocess.run([sys.executable, "-I", WORKER, work], cwd=work, capture_output=True,
                                  text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return fail(f"TimeoutError: strategy did not finish within {timeout_s:.0f}s "
                        "(likely a Python loop over rows; vectorize with pandas/numpy).")
        result_path = os.path.join(work, "result.json")
        if not os.path.exists(result_path):
            return fail(f"Sandbox process exited with code {proc.returncode} without a result.\n"
                        f"{(proc.stderr or '')[-2000:]}")
        with open(result_path, encoding="utf-8") as f:
            res = json.load(f)

    if not res.get("ok"):
        return fail(res.get("error") or "Unknown sandbox error")

    signals = pd.Series([np.nan if v is None else v for v in res["signals"]], index=df.index, dtype=float)
    try:
        metrics, equity = compute_backtest(signals, df)
    except Exception as e:
        return fail(f"{type(e).__name__} while scoring signals: {e}")
    return {
        "success": True,
        "error": None,
        "metrics": metrics,
        "lookahead": check_lookahead(signals, res["prefix"], df, flags),
        "equity": equity,
    }
