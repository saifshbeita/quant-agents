"""Streamlit front-end: live agent timeline, validation gates, equity curve and handoff report."""
import datetime as dt
import difflib
import html
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from core import llm
from core.runner import list_runs, load_run, run_pipeline

st.set_page_config(page_title="Quant Agents", page_icon="📈", layout="wide")

# Dark-surface chart colors (categorical slots 1-2, stepped for dark backgrounds).
SERIES_STRATEGY = "#3987e5"
SERIES_BENCH = "#d95926"
INK, INK_MUTED, GRID = "#e8e8e6", "#a3a29b", "rgba(255,255,255,0.07)"

NODE_META = {
    "__start__": ("🚀", "Run started"),
    "ideator": ("💡", "Ideator"),
    "coder": ("🧑‍💻", "Coder"),
    "sandbox": ("🧪", "Sandbox"),
    "auditor": ("🛡️", "Auditor"),
    "packager": ("📦", "Packager"),
    "__end__": ("🏁", "Run finished"),
    "__error__": ("⚠️", "Run aborted"),
}
RUNNING_LABEL = {
    "ideator": "Ideator is formulating a market theory…",
    "coder": "Coder is writing generate_signals…",
    "sandbox": "Sandbox is backtesting + replaying truncated history…",
    "auditor": "Auditor is scoring the gates…",
    "packager": "Packager is writing the handoff report…",
}

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1500px;}
.amd-strip {display:flex; flex-wrap:wrap; gap:6px 28px; align-items:center; padding:10px 18px;
  border:1px solid rgba(237,28,36,0.45); border-left:4px solid #ED1C24; border-radius:10px;
  background:linear-gradient(90deg, rgba(237,28,36,0.10), rgba(255,255,255,0.02)); margin-bottom:14px;}
.amd-strip .brand {font-weight:700; letter-spacing:0.02em; color:#fff;}
.amd-strip .kv {display:flex; flex-direction:column; line-height:1.15;}
.amd-strip .kv span {font-size:0.70rem; text-transform:uppercase; letter-spacing:0.06em; color:#a3a29b;}
.amd-strip .kv b {font-size:0.95rem; color:#e8e8e6; font-weight:600; font-variant-numeric: tabular-nums;}
.verdict {padding:12px 16px; border-radius:10px; margin:4px 0 14px 0; font-weight:600;}
.verdict.ok {background:rgba(25,158,112,0.14); border:1px solid rgba(25,158,112,0.55);}
.verdict.bad {background:rgba(230,103,103,0.12); border:1px solid rgba(230,103,103,0.5);}
.verdict.run {background:rgba(57,135,229,0.12); border:1px solid rgba(57,135,229,0.5);}
.verdict small {display:block; font-weight:400; color:#a3a29b; margin-top:2px;}
.gates {width:100%; border-collapse:collapse; font-size:0.92rem;}
.gates td {padding:7px 8px; border-bottom:1px solid rgba(255,255,255,0.07); vertical-align:top;}
.gates td.icon {width:28px;} .gates td.val {color:#a3a29b; font-variant-numeric: tabular-nums;}
.gates .tag {font-size:0.72rem; font-weight:700; padding:1px 6px; border-radius:4px; margin-left:6px;}
.gates .tag.pass {color:#7fdcb5; border:1px solid #199e70;} .gates .tag.fail {color:#f0a0a0; border:1px solid #e66767;}
div[data-testid="stMetric"] {background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
  border-radius:10px; padding:10px 14px;}
div[data-testid="stMetricValue"] {font-size:1.45rem; font-variant-numeric: tabular-nums;}
.pipeline {color:#a3a29b; font-size:0.95rem; margin-top:-8px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- run state folding

def new_agg() -> dict:
    return {"meta": {}, "end": None, "error": None, "ideas": [], "report": "", "events": [],
            "prompt": 0, "completion": 0, "calls": 0, "llm_latency": 0.0, "call_log": []}


def apply_event(agg: dict, ev: dict) -> None:
    node, p = ev["node"], ev.get("payload") or {}
    agg["events"].append(ev)
    tok = ev.get("tokens") or {}
    agg["prompt"] += tok.get("prompt", 0)
    agg["completion"] += tok.get("completion", 0)
    agg["calls"] += tok.get("llm_calls", 0)
    agg["llm_latency"] += tok.get("llm_latency_s", 0.0)
    agg["call_log"].extend(tok.get("calls", []))
    if node == "__start__":
        agg["meta"] = p
    elif node == "__end__":
        agg["end"] = p
    elif node == "__error__":
        agg["error"] = p.get("error")
    elif node == "ideator":
        agg["ideas"].append({"no": p.get("idea_count", len(agg["ideas"]) + 1), "title": p.get("idea_title", "Idea"),
                             "theory": p.get("market_theory", ""), "iters": []})
    elif node == "coder":
        if not agg["ideas"]:
            agg["ideas"].append({"no": 1, "title": "Idea", "theory": "", "iters": []})
        agg["ideas"][-1]["iters"].append({"no": p.get("iteration_count"), "code": p.get("current_code", "")})
    elif node in ("sandbox", "auditor") and agg["ideas"] and agg["ideas"][-1]["iters"]:
        it = agg["ideas"][-1]["iters"][-1]
        if node == "sandbox":
            it.update(metrics=p.get("backtest_metrics") or {}, equity=p.get("equity_curve") or {},
                      lookahead=p.get("lookahead") or {}, error=p.get("error_logs") or "")
        else:
            it.update(gates=p.get("gates") or {}, critique=p.get("auditor_feedback", ""), status=p.get("status"))
    elif node == "packager":
        agg["report"] = p.get("final_report", "")


def all_iters(agg: dict) -> list[tuple[dict, dict]]:
    return [(idea, it) for idea in agg["ideas"] for it in idea["iters"]]


def focus_iter(agg: dict):
    """The iteration to headline: the packaged one, else the latest one with results."""
    pairs = all_iters(agg)
    for key in ("gates", "metrics", "code"):
        for idea, it in reversed(pairs):
            if it.get(key):
                return idea, it
    return None, None


def next_node(agg: dict) -> str | None:
    if not agg["events"]:
        return "ideator"
    last = agg["events"][-1]
    node, p = last["node"], last.get("payload") or {}
    if node == "__start__":
        return "ideator"
    if node in ("ideator", "coder", "sandbox"):
        return {"ideator": "coder", "coder": "sandbox", "sandbox": "auditor"}[node]
    if node == "auditor":
        status = p.get("status")
        if status == "passed":
            return "packager"
        if status == "coding":
            return "coder"
        return "ideator" if len(agg["ideas"]) < agg["meta"].get("max_ideas", config.MAX_IDEAS) else None
    return None


# ---------------------------------------------------------------- formatting helpers

def pct(x) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def num(x, d=2) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def strip_html(agg: dict, live_defaults: bool = False) -> str:
    meta = agg["meta"]
    model = meta.get("model") or (llm.model_label() if live_defaults else "–")
    endpoint = meta.get("endpoint") or (llm.endpoint_label() if live_defaults else "–")
    tps = f"{agg['completion'] / agg['llm_latency']:.1f}" if agg["llm_latency"] and agg["completion"] else "–"
    lat = f"{agg['llm_latency'] / agg['calls']:.2f}s" if agg["calls"] else "–"
    items = [("Model", model), ("Endpoint", endpoint), ("Output tokens/s", tps), ("Latency / call", lat),
             ("LLM calls", str(agg["calls"])), ("Tokens", f"{agg['prompt'] + agg['completion']:,}")]
    kv = "".join(f'<div class="kv"><span>{k}</span><b>{html.escape(str(v))}</b></div>' for k, v in items)
    return f'<div class="amd-strip"><div class="brand">⚡ Powered by AMD</div>{kv}</div>'


def verdict_html(agg: dict, source: str) -> str:
    meta = agg["meta"]
    data = meta.get("data", {})
    th = meta.get("thresholds", {})
    sub = (f"{source} · run {meta.get('run_id', '–')} · {meta.get('ticker', '')} {data.get('start', '')} → "
           f"{data.get('end', '')} ({data.get('rows', 0)} bars, {data.get('source', '')}) · gates: OOS Sharpe > "
           f"{th.get('sharpe', '–')}, OOS max DD < {pct(th.get('max_drawdown'))}"
           f"{' · DEMO_MODE' if meta.get('demo_mode') else ''}")
    end = agg["end"]
    if agg["error"]:
        cls, text = "bad", f"⚠️ Run aborted: {agg['error']}"
    elif end is None:
        idea = agg["ideas"][-1] if agg["ideas"] else None
        where = f"idea {idea['no']}/{meta.get('max_ideas', '?')}, iteration {len(idea['iters'])}" if idea else "starting"
        cls, text = "run", f"⏳ Research loop running ({where})"
    elif end.get("passed"):
        idea, it = focus_iter(agg)
        cls, text = "ok", f"✅ PASSED: “{idea['title']}” cleared every gate (idea {idea['no']}, iteration {it['no']}). Handoff report ready."
    else:
        cls, text = "bad", f"❌ No strategy cleared the gates after {end.get('ideas_tried', 0)} idea(s)."
    return f'<div class="verdict {cls}">{html.escape(text)}<small>{html.escape(sub)}</small></div>'


def gates_html(gates: dict) -> str:
    def fmt(key, g):
        v = g.get("value")
        if key == "sharpe":
            return num(v)
        if key == "max_drawdown":
            return pct(v)
        s = str(v or "")
        return s if len(s) < 180 else s[:177] + "…"
    rows = []
    for key in ("sharpe", "max_drawdown", "lookahead", "out_of_sample"):
        g = gates.get(key)
        if not g:
            continue
        ok = g.get("passed")
        rows.append(
            f'<tr><td class="icon">{"✅" if ok else "❌"}</td><td>{html.escape(g["label"])}'
            f'<span class="tag {"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</span></td>'
            f'<td class="val">{html.escape(fmt(key, g))}</td></tr>')
    return f'<table class="gates">{"".join(rows)}</table>'


def equity_fig(eq: dict) -> go.Figure:
    dates = pd.to_datetime(eq["dates"])
    fig = go.Figure()
    for name, ys, color in (("Strategy", eq["strategy"], SERIES_STRATEGY), ("Buy & hold", eq["buy_hold"], SERIES_BENCH)):
        fig.add_trace(go.Scatter(x=dates, y=ys, name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>"))
        fig.add_annotation(x=dates[-1], y=ys[-1], text=f"{name} {ys[-1]:.2f}×", showarrow=False, xanchor="left",
                           xshift=6, font=dict(color=INK_MUTED, size=11))
    if eq.get("split_date"):
        fig.add_vrect(x0=pd.Timestamp(eq["split_date"]), x1=dates[-1], fillcolor="rgba(255,255,255,0.045)",
                      line_width=0, annotation_text="Out-of-sample", annotation_position="top left",
                      annotation_font=dict(color=INK_MUTED, size=11))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"))
    fig.update_layout(
        height=360, margin=dict(l=10, r=90, t=30, b=10), hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_MUTED)),
        yaxis=dict(title="Growth of $1", gridcolor=GRID, zeroline=False, tickformat=".2f"),
        xaxis=dict(range=[dates[0], dates[-1]], gridcolor=GRID, showspikes=True, spikecolor="rgba(255,255,255,0.3)", spikethickness=1),
    )
    return fig


def iter_label(idea: dict, it: dict) -> str:
    status = {"passed": "✅ passed", "failed": "❌ idea exhausted", "coding": "↻ revise"}.get(it.get("status"), "…")
    return f"Idea {idea['no']} · iteration {it['no']} · {idea['title'][:50]} ({status})"


def code_diff(agg: dict, idea: dict, it: dict) -> str | None:
    prev = [x for x in idea["iters"] if (x["no"] or 0) < (it["no"] or 0)]
    if not prev:
        return None
    diff = difflib.unified_diff(prev[-1]["code"].splitlines(), it["code"].splitlines(),
                                fromfile=f"iteration {prev[-1]['no']}", tofile=f"iteration {it['no']}", lineterm="")
    return "\n".join(diff) or "(no changes)"


# ---------------------------------------------------------------- dashboard

class Dashboard:
    def __init__(self, source: str, live: bool):
        self.source, self.live = source, live
        self.strip = st.empty()
        self.verdict = st.empty()
        left, right = st.columns([5, 7], gap="large")
        with left:
            st.markdown("##### Agent timeline")
            self.timeline = st.container(height=720, border=True, autoscroll=True if live else None)
            self.pending = None
        with right:
            st.markdown("##### Validation gates")
            self.gates_ph = st.empty()
            self.cards_ph = st.empty()
            self.chart_ph = st.empty()
        tabs = st.tabs(["📜 Theory", "🧑‍💻 Code", "🛡️ Auditor critique", "📦 Handoff report", "⚡ LLM calls"])
        self.tab_ph = [t.empty() for t in tabs]

    # -- timeline
    def add_event(self, ev: dict, agg: dict) -> None:
        if self.pending is not None:
            self.pending.empty()
        node, p = ev["node"], ev.get("payload") or {}
        icon, name = NODE_META.get(node, ("•", node))
        tok = ev.get("tokens") or {}
        bits = [f"{ev.get('latency_s', 0):.1f}s"]
        if tok.get("total"):
            bits.append(f"{tok['total']:,} tok")
        detail, state = "", "complete"
        if node == "ideator":
            detail = f"idea {p.get('idea_count')}: {p.get('idea_title', '')}"
        elif node == "coder":
            detail = f"iteration {p.get('iteration_count')}"
        elif node == "sandbox":
            test = (p.get("backtest_metrics") or {}).get("test")
            if p.get("error_logs"):
                detail, state = "execution error", "error"
            elif test:
                detail = f"OOS Sharpe {test['sharpe']:.2f} · DD {pct(test['max_drawdown'])}"
                if not (p.get("lookahead") or {}).get("passed"):
                    detail, state = detail + " · LOOKAHEAD", "error"
        elif node == "auditor":
            s = p.get("status")
            detail = {"passed": "all gates passed", "coding": "rejected → revise", "failed": "rejected → idea exhausted"}.get(s, s)
            state = "complete" if s == "passed" else "error"
        elif node == "__end__":
            detail = "passed" if p.get("passed") else p.get("status", "")
        elif node == "__error__":
            state = "error"
        label = f"{icon} {name}" + (f" — {detail}" if detail else "") + f"  ·  {' · '.join(bits)}"

        with self.timeline:
            with st.status(label, state=state, expanded=False):
                self._event_body(node, p)
            nxt = next_node(agg) if self.live and node not in ("__end__", "__error__") else None
            if nxt:
                self.pending = st.empty()
                self.pending.status(RUNNING_LABEL[nxt], state="running")
            else:
                self.pending = None

    @staticmethod
    def _event_body(node: str, p: dict) -> None:
        if node == "__start__":
            d = p.get("data", {})
            st.markdown(f"**{p.get('ticker')}** {d.get('start')} → {d.get('end')} · {d.get('rows')} bars from "
                        f"`{d.get('source')}` · model `{p.get('model')}` · max {p.get('max_ideas')} ideas × "
                        f"{p.get('max_iterations')} iterations · {p.get('cost_bps')}bp/turnover")
        elif node == "ideator":
            st.markdown(p.get("market_theory", ""))
        elif node == "coder":
            st.code(p.get("current_code", ""), language="python")
        elif node == "sandbox":
            if p.get("error_logs"):
                st.code(p["error_logs"][-2500:], language="text")
            m = p.get("backtest_metrics") or {}
            if m:
                st.markdown(" · ".join(f"**{s}** Sharpe {m[s]['sharpe']:.2f}, DD {pct(m[s]['max_drawdown'])}, "
                                       f"return {pct(m[s]['total_return'])}" for s in ("train", "test")))
            la = p.get("lookahead") or {}
            if la:
                st.markdown(f"{'✅' if la.get('passed') else '❌'} **Lookahead:** {la.get('message', '')}")
        elif node == "auditor":
            g = p.get("gates") or {}
            if g:
                st.markdown(gates_html(g), unsafe_allow_html=True)
            st.markdown(p.get("auditor_feedback", ""))
        elif node == "packager":
            st.markdown("Handoff report generated, see the **Handoff report** tab.")
        elif node == "__end__":
            st.markdown(f"{p.get('total_llm_calls', 0)} LLM calls · {p.get('total_total', 0):,} tokens · "
                        f"{p.get('tokens_per_s', 0)} tok/s · wall time {p.get('wall_time_s', 0)}s")
        elif node == "__error__":
            st.error(p.get("error", ""))

    # -- panels
    def render_panels(self, agg: dict, final: bool) -> None:
        key = len(agg["events"])
        self.strip.markdown(strip_html(agg), unsafe_allow_html=True)
        if agg["meta"]:
            self.verdict.markdown(verdict_html(agg, self.source), unsafe_allow_html=True)
        idea, it = focus_iter(agg)

        with self.gates_ph.container():
            if it and it.get("gates"):
                st.markdown(gates_html(it["gates"]), unsafe_allow_html=True)
                st.caption(f"Showing {iter_label(idea, it)}")
            else:
                st.caption("Gates appear after the first audit.")

        with self.cards_ph.container():
            m = (it or {}).get("metrics") or {}
            if m:
                bench = m.get("benchmark", {})
                for split, title in (("train", f"Train ({int(m.get('train_fraction', 0.7) * 100)}%)"),
                                     ("test", f"Test / out-of-sample (from {m.get('split_date')})")):
                    s, b = m[split], bench.get(split, {})
                    st.markdown(f"**{title}**")
                    c = st.columns(5)
                    vs = "Delta vs buy & hold over the same period"
                    c[0].metric("Sharpe", num(s["sharpe"]), delta=f"{s['sharpe'] - b.get('sharpe', 0):+.2f}", help=vs)
                    c[1].metric("Max DD", pct(s["max_drawdown"]), delta_color="inverse", help=vs,
                                delta=f"{(s['max_drawdown'] - b.get('max_drawdown', 0)) * 100:+.1f}pp")
                    c[2].metric("Return", pct(s["total_return"]), help=vs,
                                delta=f"{(s['total_return'] - b.get('total_return', 0)) * 100:+.1f}pp")
                    c[3].metric("Win rate", pct(s["win_rate"]), help="Share of in-market days with a positive return")
                    c[4].metric("Exposure", pct(s["exposure"]), help=f"{s['trades']} position changes")

        with self.chart_ph.container():
            eq = (it or {}).get("equity") or {}
            if eq.get("dates"):
                st.plotly_chart(equity_fig(eq), key=f"eq-{key}-{final}", config={"displayModeBar": False})

        self._render_tabs(agg, idea, it, final, key)

    def _render_tabs(self, agg, idea, it, final, key) -> None:
        theory_ph, code_ph, audit_ph, report_ph, calls_ph = self.tab_ph
        with theory_ph.container():
            if idea:
                st.markdown(idea["theory"])
                for other in agg["ideas"]:
                    if other is not idea:
                        with st.expander(f"Idea {other['no']}: {other['title']}"):
                            st.markdown(other["theory"])
            else:
                st.caption("No theory yet.")

        with code_ph.container():
            pairs = all_iters(agg)
            if not pairs:
                st.caption("No code yet.")
            else:
                sel_idea, sel_it = idea, it
                if final:
                    labels = [iter_label(i, x) for i, x in pairs]
                    default = next((n for n, (i, x) in enumerate(pairs) if x is it), len(pairs) - 1)
                    choice = st.selectbox("Iteration", range(len(pairs)), index=default, format_func=lambda n: labels[n])
                    sel_idea, sel_it = pairs[choice]
                code_tab, diff_tab = st.tabs(["Code", "Diff vs previous iteration"])
                with code_tab:
                    st.code(sel_it["code"], language="python", line_numbers=True)
                with diff_tab:
                    diff = code_diff(agg, sel_idea, sel_it)
                    if diff:
                        st.code(diff, language="diff")
                    else:
                        st.caption("First iteration of this idea, nothing to diff.")

        with audit_ph.container():
            if it and it.get("critique"):
                st.markdown(f"**{iter_label(idea, it)}**")
                st.markdown(it["critique"])
                for i, x in reversed(all_iters(agg)):
                    if x is not it and x.get("critique"):
                        with st.expander(iter_label(i, x)):
                            st.markdown(x["critique"])
            else:
                st.caption("No critique yet.")

        with report_ph.container():
            if agg["report"]:
                if final:
                    st.download_button("⬇️ Download handoff report (.md)", data=agg["report"],
                                       file_name=f"{agg['meta'].get('run_id', 'strategy')}_handoff.md",
                                       mime="text/markdown", on_click="ignore", type="primary")
                st.markdown(agg["report"])
            else:
                st.caption("The handoff report is written once a strategy passes every gate.")

        with calls_ph.container():
            if agg["call_log"]:
                df = pd.DataFrame(agg["call_log"])
                df["tokens_per_s"] = (df["completion_tokens"] / df["latency_s"].where(df["latency_s"] > 0)).round(1)
                st.dataframe(df, hide_index=True, key=f"calls-{key}-{final}")
            else:
                st.caption("No LLM calls yet.")


def render_stream(events, source: str, live: bool, delay: float = 0.0) -> list[dict]:
    """Draw events as they arrive (live run or animated replay) and return them."""
    dash = Dashboard(source, live=True)
    agg = new_agg()
    dash.strip.markdown(strip_html(agg, live_defaults=live), unsafe_allow_html=True)
    for ev in events:
        apply_event(agg, ev)
        dash.add_event(ev, agg)
        dash.render_panels(agg, final=False)
        if delay:
            time.sleep(delay)
    return agg["events"]


def render_static(events: list[dict], source: str) -> None:
    dash = Dashboard(source, live=False)
    agg = new_agg()
    for ev in events:
        apply_event(agg, ev)
        dash.add_event(ev, agg)
    dash.render_panels(agg, final=True)


# ---------------------------------------------------------------- page

@st.cache_data(ttl=20, show_spinner=False)
def endpoint_status(base_url: str, mock: bool) -> tuple[bool, str]:
    return llm.ping_endpoint()


def replayable(runs: list[dict]) -> list[dict]:
    return [r for r in runs if r["status"] not in ("error", "incomplete")]


st.title("📈 Quant Agents")
st.markdown('<div class="pipeline">Autonomous quant research desk · 💡 Ideator → 🧑‍💻 Coder → 🧪 Sandbox → '
            '🛡️ Auditor → 📦 Packager</div>', unsafe_allow_html=True)

reachable, reach_detail = endpoint_status(config.LLM_BASE_URL, config.LLM_MOCK)
runs = list_runs()
good_runs = replayable(runs)

with st.sidebar:
    st.header("Research run")
    ticker = st.text_input("Ticker", value=config.TARGET_TICKER).strip().upper()
    today = dt.date.today()
    dates = st.date_input("Date range", value=(today - dt.timedelta(days=730), today), max_value=today)
    max_ideas = st.slider("Max ideas", 1, 5, config.MAX_IDEAS,
                          help=f"Each idea gets up to {config.MAX_ITERATIONS} coder iterations before the Ideator moves on.")
    demo_mode = st.toggle("Demo mode (looser gates)", value=config.DEMO_MODE,
                          help=f"Strict: OOS Sharpe > {config.BENCHMARK_SHARPE}, DD < {config.MAX_DRAWDOWN_LIMIT:.0%}. "
                               f"Demo: Sharpe > {config.DEMO_BENCHMARK_SHARPE}, DD < {config.DEMO_MAX_DRAWDOWN_LIMIT:.0%}.")
    dates_ok = isinstance(dates, (tuple, list)) and len(dates) == 2 and (dates[1] - dates[0]).days > 120
    if not dates_ok:
        st.caption("Pick a start and end date at least ~4 months apart.")
    run_clicked = st.button("▶ Run research loop", type="primary", width="stretch",
                            disabled=not (reachable and dates_ok and ticker))
    st.caption(("🟢 " if reachable else "🔴 ") + f"`{llm.endpoint_label()}` · `{llm.model_label()}`")
    if not reachable and st.button("Retry connection", width="stretch"):
        endpoint_status.clear()
        st.rerun()

    st.divider()
    st.subheader("Replay recorded run")
    if runs:
        labels = {r["path"]: f"{r['run_id']} · {r['model']} · {'✅' if r['passed'] else r['status']}" for r in runs}
        default_path = (good_runs or runs)[0]["path"]
        paths = [r["path"] for r in runs]
        replay_path = st.selectbox("Recorded run", paths, index=paths.index(default_path),
                                   format_func=lambda p: labels[p], label_visibility="collapsed")
        animate = st.toggle("Animate replay", value=True)
        replay_clicked = st.button("⏯ Replay", width="stretch")
    else:
        st.caption(f"No recorded runs in `{os.path.relpath(config.RUNS_DIR)}` yet.")
        replay_clicked, replay_path, animate = False, None, False

if not reachable:
    st.error(f"LLM endpoint `{llm.endpoint_label()}` is unreachable ({reach_detail}). Live runs are disabled"
             + (", showing a recorded run instead." if good_runs else ". Record a run when the endpoint is up."))

if run_clicked:
    params = {"ticker": ticker, "start": dates[0].isoformat(), "end": (dates[1] + dt.timedelta(days=1)).isoformat(),
              "max_ideas": max_ideas, "demo_mode": demo_mode}
    events = render_stream(run_pipeline(params), source="live", live=True)
    st.session_state.update(events=events, source="live")
    calls = sum((e.get("tokens") or {}).get("llm_calls", 0) for e in events)
    if any(e["node"] == "__error__" for e in events) and calls == 0:
        endpoint_status.clear()  # the endpoint probably went away mid-click
    st.rerun()
elif replay_clicked and replay_path:
    events = load_run(replay_path)
    name = os.path.basename(replay_path)
    if animate:
        render_stream(events, source=f"replay of {name}", live=False, delay=0.35)
    st.session_state.update(events=events, source=f"replay of {name}")
    st.rerun()
elif st.session_state.get("events"):
    render_static(st.session_state["events"], st.session_state.get("source", ""))
elif not reachable and good_runs:
    render_static(load_run(good_runs[0]["path"]), f"replay of {os.path.basename(good_runs[0]['path'])} (endpoint offline)")
else:
    agg = new_agg()
    st.markdown(strip_html(agg, live_defaults=True), unsafe_allow_html=True)
    st.info("Configure a run in the sidebar and press **Run research loop**, or replay a recorded run. "
            "Each strategy is backtested out-of-sample with 1bp/turnover costs and must pass a deterministic "
            "lookahead test before the Packager writes a handoff report.")
