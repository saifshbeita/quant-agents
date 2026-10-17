import re

from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from core.llm import chat, strip_think

_PY_BLOCK_RE = re.compile(r"```[ \t]*(?:python|py)[ \t]*\r?\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK_RE = re.compile(r"```[^\n]*\r?\n(.*?)(?:```|$)", re.DOTALL)


def extract_code(text: str) -> str:
    """Strip <think> blocks, then take the first ```python block if present, else the raw text."""
    text = strip_think(text)
    for pattern in (_PY_BLOCK_RE, _ANY_BLOCK_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return text.strip()


def coder_agent(state: AgentState) -> dict:

    system_prompt = """You are a Senior Quantitative Developer. Your objective is to translate theoretical alpha models into zero-latency, heavily vectorized Python backtesting code.
Strict Directives:
1. Write executable Python code that defines the exact function signature: `def generate_signals(df: pd.DataFrame) -> pd.Series:`. Module-level helper functions are allowed.
2. Output only the code, in a single ```python block, with no explanatory text.
3. Inputs: `df` has a DatetimeIndex and float columns Open, High, Low, Close, Volume, Market_Return (close-to-close % change). No other data exists. `pd` and `np` are pre-imported; do not import anything else.
4. Output: a pd.Series aligned to df.index with the desired position in [-1, 1] (1 = fully long, -1 = fully short, 0 = flat). The harness trades the signal from bar t on bar t+1 and charges 1bp per unit of turnover, so do NOT shift the final signal yourself.
5. Performance: You are strictly forbidden from using `for` loops or `.iterrows()`. You must use pure `pandas` and `numpy` vectorized matrix operations. Execution is killed after 30 seconds.
6. Lookahead Bias Prevention: the value at bar t may only use rows <= t. Never use shift(-n), center=True, bfill, or full-sample statistics (e.g. df['Close'].mean()); use rolling/expanding/ewm windows instead. The harness re-runs your function on truncated history and rejects any signal that changes.
7. Error Handling: Gracefully handle `NaN` values, division by zero, and missing data gaps.
Address any feedback from the Auditor precisely."""

    human_content = f"Market Theory:\n{state.get('market_theory', '')}\n\n"

    if state.get('error_logs'):
        human_content += f"Execution Error Logs:\n{state.get('error_logs')}\n\n"

    lookahead = state.get('lookahead') or {}
    if lookahead and not lookahead.get('passed', True) and not state.get('error_logs'):
        human_content += f"LOOKAHEAD TEST FAILED:\n{lookahead.get('message')}\n\n"

    if state.get('auditor_feedback'):
        human_content += f"Auditor Feedback:\n{state.get('auditor_feedback')}\n\n"

    if state.get('current_code'):
        human_content += f"Previous Code:\n```python\n{state.get('current_code')}\n```\n\n"

    human_content += "Provide the complete Python code with the generate_signals function."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]

    code = extract_code(chat(messages, node="coder"))

    return {
        "current_code": code,
        "iteration_count": state.get('iteration_count', 0) + 1
    }
