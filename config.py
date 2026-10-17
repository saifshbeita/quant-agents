import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- LLM endpoint (own vLLM on AMD, AMD Token Factory, or any OpenAI-compatible API) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
# MODEL_NAME is accepted as a legacy alias for LLM_MODEL.
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or "Qwen/Qwen3.6-35B-A3B"
MODEL_NAME = LLM_MODEL
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "180"))
# Qwen3 models think by default; set LLM_DISABLE_THINKING=1 to send enable_thinking=False (vLLM chat template kwarg).
LLM_DISABLE_THINKING = _env_bool("LLM_DISABLE_THINKING", False)
# LLM_MOCK=1 swaps the endpoint for a deterministic offline fake (rehearsals / tests).
LLM_MOCK = _env_bool("LLM_MOCK", False)

# --- Research loop ---
TARGET_TICKER = os.getenv("TARGET_TICKER", "SPY")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))  # coder attempts per idea
MAX_IDEAS = int(os.getenv("MAX_IDEAS", "3"))  # ideas per run before giving up
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "100"))

# --- Sandbox ---
SANDBOX_TIMEOUT_S = float(os.getenv("SANDBOX_TIMEOUT_S", "30"))
TRANSACTION_COST_BPS = float(os.getenv("TRANSACTION_COST_BPS", "1.0"))  # per unit of turnover
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.7"))

# --- Audit gates (evaluated on the out-of-sample split) ---
BENCHMARK_SHARPE = float(os.getenv("BENCHMARK_SHARPE", "1.5"))
MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.15"))
DEMO_MODE = _env_bool("DEMO_MODE", False)
DEMO_BENCHMARK_SHARPE = float(os.getenv("DEMO_BENCHMARK_SHARPE", "0.5"))
DEMO_MAX_DRAWDOWN_LIMIT = float(os.getenv("DEMO_MAX_DRAWDOWN_LIMIT", "0.25"))

RUNS_DIR = os.getenv("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))


def get_thresholds(demo_mode: bool | None = None) -> dict:
    """Gate thresholds for a run. DEMO_MODE swaps in looser limits."""
    demo = DEMO_MODE if demo_mode is None else demo_mode
    if demo:
        return {"sharpe": DEMO_BENCHMARK_SHARPE, "max_drawdown": DEMO_MAX_DRAWDOWN_LIMIT, "demo_mode": True}
    return {"sharpe": BENCHMARK_SHARPE, "max_drawdown": MAX_DRAWDOWN_LIMIT, "demo_mode": False}
