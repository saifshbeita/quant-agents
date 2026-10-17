from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from core.llm import chat, strip_think


def ideator_agent(state: AgentState) -> dict:
    ticker = state.get("ticker", "SPY")
    system_prompt = f"""You are an elite quantitative research theorist at a tier-one proprietary trading firm. Your objective is to formulate robust, mathematically rigorous trading hypotheses that exploit structural market inefficiencies, statistical arbitrage, or regime-dependent anomalies. Avoid generic retail technical indicators. You must define:
1. The Alpha Rationale: Why does this inefficiency exist (e.g., liquidity constraints, behavioral overreaction)?
2. Mathematical Formulation: Precise entry and exit logic using statistical thresholds (e.g., rolling Z-scores, volatility-adjusted momentum).
3. Regime Filters: Conditions under which the strategy should halt trading (e.g., realized-volatility spikes, trending vs. ranging environments).

Data constraints: the strategy trades a single instrument, {ticker}, on daily bars. The ONLY data available is the daily Open, High, Low, Close, Volume and Market_Return (close-to-close % change) of {ticker}. No VIX, no other tickers, no fundamentals, no intraday data. Every quantity must be computable from that history using information available at the bar's close.
Results are judged out-of-sample after 1bp-per-turnover transaction costs, so prefer low-turnover ideas.
Start with a one-line title prefixed by '# '. Keep the theory under 350 words."""

    human = "Please generate a new quantitative trading theory."
    tried = state.get("tried_ideas") or []
    if tried:
        human += "\n\nThese ideas were already tried and REJECTED by risk; propose something structurally different:\n"
        human += "\n".join(f"- {t}" for t in tried)

    theory = strip_think(chat([SystemMessage(content=system_prompt), HumanMessage(content=human)], node="ideator"))
    title = next((line.lstrip("# ").strip() for line in theory.splitlines() if line.strip()), "Untitled idea")

    return {
        "market_theory": theory,
        "idea_title": title[:120],
        "idea_count": state.get("idea_count", 0) + 1,
        "tried_ideas": tried + [title[:120]],
        "iteration_count": 0,
        "current_code": "",
        "auditor_feedback": "",
        "error_logs": "",
        "backtest_metrics": {},
        "lookahead": {},
        "gates": {},
        "status": "coding",
    }
