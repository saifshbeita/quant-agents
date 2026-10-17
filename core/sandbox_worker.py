"""
Child process for core.sandbox: runs untrusted generate_signals code and nothing else.

Usage: python -I sandbox_worker.py <input_dir>
  <input_dir>/code.py      strategy source
  <input_dir>/df.pkl       market data (written by the trusted parent)
  <input_dir>/prefix.json  list of k values for the lookahead test
Writes <input_dir>/result.json. All metric computation happens in the parent.
"""
import contextlib
import io
import json
import linecache
import os
import sys
import traceback

import numpy as np
import pandas as pd


def _to_series(out, df: pd.DataFrame) -> pd.Series:
    if isinstance(out, pd.DataFrame) and out.shape[1] == 1:
        out = out.iloc[:, 0]
    if not isinstance(out, pd.Series):
        raise TypeError(f"generate_signals must return a pandas Series, got {type(out).__name__}")
    out = out.reindex(df.index)
    return pd.to_numeric(out.astype(float) if out.dtype == bool else out, errors="raise").astype(float)


def _encode(values) -> list:
    return [None if not np.isfinite(v) else float(v) for v in values]


def main(work_dir: str) -> None:
    with open(os.path.join(work_dir, "code.py"), encoding="utf-8") as f:
        code = f.read()
    with open(os.path.join(work_dir, "prefix.json"), encoding="utf-8") as f:
        prefix_ks = json.load(f)
    df = pd.read_pickle(os.path.join(work_dir, "df.pkl"))

    result = {"ok": False, "stage": "exec", "error": None, "signals": None, "prefix": [], "stdout": ""}
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            # Register the source so tracebacks show the offending line to the Coder.
            linecache.cache["<strategy>"] = (len(code), None, code.splitlines(True), "<strategy>")
            ns = {"pd": pd, "np": np, "__name__": "strategy"}
            exec(compile(code, "<strategy>", "exec"), ns)
            fn = ns.get("generate_signals")
            if not callable(fn):
                raise ValueError("The code must define generate_signals(df: pd.DataFrame) -> pd.Series.")

            result["stage"] = "full_run"
            signals = _to_series(fn(df.copy()), df)
            result["signals"] = _encode(signals.values)

            result["stage"] = "lookahead_test"
            for k in prefix_ks:
                sub = df.iloc[:k].copy()
                try:
                    s = _to_series(fn(sub), sub)
                    v = s.iloc[-1]
                    result["prefix"].append({"k": k, "value": None if not np.isfinite(v) else float(v)})
                except Exception as e:
                    result["prefix"].append({"k": k, "error": f"{type(e).__name__}: {e}"})
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    result["stdout"] = captured.getvalue()[-2000:]

    with open(os.path.join(work_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(sys.argv[1])
