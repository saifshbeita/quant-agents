"""
Deterministic offline stand-in for the chat model (LLM_MOCK=1, tests, UI rehearsals).

Each idea plays out the same arc: the first attempt has a lookahead bug (full-sample z-score),
the next attempt is a causal version. Whether it then passes depends on the real gates.
"""
import os
import re
import time

from langchain_core.messages import AIMessage

IDEAS = [
    ("Volatility-Managed Trend", 50, 20),
    ("Calm-Regime Momentum Filter", 100, 30),
    ("Drawdown-Aware Trend Persistence", 30, 15),
]

THEORY = """# {title}

**Alpha rationale.** Institutional rebalancing and slow-moving capital make medium-term trends in {ticker} persist, while volatility clustering means risk-adjusted returns collapse in turbulent regimes.

**Formulation.** Go long when the close is above its {n}-day mean, scaled by the inverse of {v}-day realized volatility relative to its expanding median (capped at 1). Flat otherwise.

**Regime filter.** Stand aside when {v}-day realized volatility exceeds 2x its expanding median.
"""

BUGGY = """<think>
The theory needs a z-score of price. I'll normalize against the sample mean.
</think>
```python
def generate_signals(df: pd.DataFrame) -> pd.Series:
    z = (df["Close"] - df["Close"].mean()) / df["Close"].std()
    return (z > 0).astype(float)
```"""

FIXED = """```python
def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def generate_signals(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    trend = close > close.rolling({n}, min_periods={n}).mean()
    vol = realized_vol(df["Market_Return"], {v})
    calm = vol.expanding(min_periods={v}).median()
    scale = (calm / vol).clip(upper=1.0).fillna(0.0)
    halt = vol > 2 * calm
    signal = trend.astype(float) * scale
    return signal.where(~halt, 0.0).fillna(0.0)
```"""

PACKAGE = """## 1. Production Code
```python
{code}
```

## 2. Complexity Analysis
Rolling mean/std and the expanding median are O(n) per bar with O(w) state; the full pass is O(n) time, O(n) memory.

## 3. Execution Edge Cases
- Slippage: backtest charges 1bp per unit turnover; size-up days cluster after volatility drops, so expect wider spreads at those transitions.
- Missing bars: rolling windows require full history; a gap shortens the effective window and can flip the trend flag.
- Corporate actions: signals assume split/dividend-adjusted closes.

## 4. C++ Translation Architecture
Use ring buffers for the rolling windows (Welford for variance) and a two-heap running median for the expanding median; Eigen is unnecessary for a single instrument.
"""


class MockLLM:
    model_label = "mock-llm (offline)"

    def __init__(self, latency_s: float | None = None, always_buggy: bool = False):
        self.latency_s = float(os.getenv("MOCK_LLM_LATENCY_S", "0.3")) if latency_s is None else latency_s
        self.always_buggy = always_buggy

    def _idea(self, text: str):
        for title, n, v in IDEAS:
            if title in text:
                return title, n, v
        return IDEAS[0]

    def invoke(self, messages):
        system, human = messages[0].content, messages[-1].content
        if "research theorist" in system:
            tried = len(re.findall(r"^- ", human, re.MULTILINE))
            title, n, v = IDEAS[tried % len(IDEAS)]
            ticker = re.search(r"single instrument, (\S+?),", system)
            text = THEORY.format(title=title, n=n, v=v, ticker=ticker.group(1) if ticker else "the asset")
        elif "Quantitative Developer" in system:
            title, n, v = self._idea(human)
            first_attempt = "Previous Code:" not in human
            text = BUGGY if (first_attempt or self.always_buggy) else FIXED.format(n=n, v=v)
        elif "Risk Management" in system:
            failed = [line[2:].split(":")[0] for line in human.splitlines() if line.startswith("- ") and ": FAIL" in line]
            if not failed:
                text = ("- Verdict stands: all gates pass out-of-sample.\n- Residual risk: trend filters lag at regime turns; "
                        "monitor gap risk on Monday opens.\n- Volatility scaling caps exposure; check turnover stays under budget live.")
            else:
                text = "\n".join(f"- **{g}** failed." for g in failed)
                if any(g.startswith("Lookahead") for g in failed):
                    text += ("\n- The z-score uses `df['Close'].mean()` / `.std()` over the whole sample, so every bar "
                             "sees future prices. Replace with rolling or expanding statistics.")
                else:
                    text += "\n- Edge is too thin after costs; add a volatility regime filter instead of tuning windows."
        elif "Execution Engineer" in system:
            code = re.search(r"```python\n(.*?)```", human, re.DOTALL)
            text = PACKAGE.format(code=code.group(1).strip() if code else "")
        else:
            text = "OK"

        if self.latency_s:
            time.sleep(self.latency_s)
        prompt_tokens = sum(len(m.content) for m in messages) // 4
        completion_tokens = len(text) // 4
        return AIMessage(content=text, usage_metadata={
            "input_tokens": prompt_tokens, "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
