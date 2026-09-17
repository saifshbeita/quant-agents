import os
from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
TARGET_TICKER = os.getenv("TARGET_TICKER", "SPY")
BENCHMARK_SHARPE = float(os.getenv("BENCHMARK_SHARPE", "1.5"))
MAX_DRAWDOWN_LIMIT = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.15"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
