"""
Offline end-to-end tests with a fake LLM: the graph terminates, gates are enforced in code,
endpoint failures are reported, and the Streamlit app loads (replay + live run).

Run: python test_e2e_mock.py   (or: pytest test_e2e_mock.py)
"""
import json
import os
import sys
import tempfile

import config
from core import llm
from core.mock_llm import MockLLM
from core.runner import list_runs, load_run, run_pipeline
from data.fetcher import _load_csv

TMP = tempfile.mkdtemp(prefix="qa_e2e_")


def market_df():
    df = _load_csv("SPY")
    df["Market_Return"] = df["Close"].pct_change()
    return df.dropna()


def run(fake, **params) -> list[dict]:
    llm.set_llm(fake)
    try:
        return list(run_pipeline({"market_df": market_df(), "runs_dir": TMP, **params}))
    finally:
        llm.set_llm(None)


def nodes(events):
    return [e["node"] for e in events]


def test_graph_terminates_when_nothing_passes():
    events = run(MockLLM(latency_s=0, always_buggy=True), max_ideas=3, max_iterations=5, run_id="never-passes")
    seq = nodes(events)
    assert seq[0] == "__start__" and seq[-1] == "__end__"
    assert seq.count("ideator") == 3, seq
    assert seq.count("coder") == 15, seq
    assert "packager" not in seq
    end = events[-1]["payload"]
    assert end["status"] == "failed" and not end["passed"]
    # every audit failed on the lookahead gate, decided in code
    audits = [e["payload"] for e in events if e["node"] == "auditor"]
    assert all(not a["gates"]["lookahead"]["passed"] for a in audits)
    # ideator resets the per-idea state
    for e in events:
        if e["node"] == "ideator":
            p = e["payload"]
            assert p["current_code"] == "" and p["auditor_feedback"] == "" and p["error_logs"] == ""
    # JSONL log mirrors the stream
    logged = load_run(os.path.join(TMP, "never-passes.jsonl"))
    assert nodes(logged) == seq
    assert all({"node", "payload", "ts", "latency_s", "tokens"} <= set(e) for e in logged)


def test_demo_mode_passes_and_packages():
    events = run(MockLLM(latency_s=0), max_ideas=3, demo_mode=True, run_id="demo-pass")
    seq = nodes(events)
    assert seq[-2:] == ["packager", "__end__"], seq
    final_audit = [e for e in events if e["node"] == "auditor"][-1]["payload"]
    assert final_audit["status"] == "passed"
    assert all(g["passed"] for g in final_audit["gates"].values())
    report = [e for e in events if e["node"] == "packager"][0]["payload"]["final_report"]
    assert "## Validation Gates" in report and "Test (OOS)" in report
    end = events[-1]["payload"]
    assert end["passed"] and end["total_llm_calls"] == sum(e["tokens"]["llm_calls"] for e in events)
    start = events[0]["payload"]
    assert start["thresholds"]["demo_mode"] is True


class _DownLLM:
    def invoke(self, messages):
        raise ConnectionError("endpoint down")


def test_endpoint_failure_is_reported():
    events = run(_DownLLM(), run_id="endpoint-down")
    assert nodes(events) == ["__start__", "__error__", "__end__"]
    assert "endpoint down" in events[1]["payload"]["error"]
    assert events[-1]["payload"]["status"] == "error"
    assert [r["run_id"] for r in list_runs(TMP) if r["status"] == "error"] == ["endpoint-down"]


def test_ui_loads_replay_and_live_run():
    from streamlit.testing.v1 import AppTest

    old = config.RUNS_DIR, config.LLM_MOCK
    config.RUNS_DIR, config.LLM_MOCK = TMP, True
    llm.set_llm(MockLLM(latency_s=0))
    try:
        at = AppTest.from_file("app.py", default_timeout=300).run()
        assert not at.exception, [e.value for e in at.exception]
        run_btn = next(b for b in at.button if "Run research loop" in b.label)
        assert not run_btn.disabled

        # replay a recorded run
        replay_btn = next(b for b in at.button if "Replay" in b.label)
        at.toggle[1].set_value(False)  # no animation
        replay_btn.click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any('class="verdict' in m.value for m in at.markdown)
        assert len(at.metric) == 10

        # live run with the fake LLM (DEMO_MODE on -> passes)
        at.toggle[0].set_value(True)
        next(b for b in at.button if "Run research loop" in b.label).click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("PASSED" in m.value for m in at.markdown if 'class="verdict' in m.value)
        assert len(at.get("download_button")) == 1
    finally:
        config.RUNS_DIR, config.LLM_MOCK = old
        llm.set_llm(None)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"[OK]   {name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{'All tests passed' if not failed else f'{failed} test(s) failed'} (run logs in {TMP})")
    sys.exit(1 if failed else 0)
