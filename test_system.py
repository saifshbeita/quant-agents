import os
import pandas as pd
from dotenv import load_dotenv
from config import LLM_BASE_URL, TARGET_TICKER
from data.fetcher import get_historical_data
from core.sandbox import execute_strategy
from core.llm import ping_endpoint

def test_connectivity():
    print(f"Testing connectivity to LLM server at {LLM_BASE_URL}...")
    ok, detail = ping_endpoint()
    if ok:
        print("[OK] Successfully connected to LLM server (GET /models returned 200).")
    else:
        print(f"[FAIL] Could not connect to LLM server: {detail}")
        print("       (This is expected if the vLLM server is not currently running)")

def test_data_fetch():
    print(f"\nFetching 30 days (1mo) of data for {TARGET_TICKER}...")
    try:
        df = get_historical_data(ticker=TARGET_TICKER, period="1mo", interval="1d")
        if df is not None and not df.empty:
            print(f"[OK] Successfully downloaded {len(df)} rows of data.")
            return df
        else:
            print("[FAIL] DataFrame is empty.")
            return None
    except Exception as e:
        print(f"[FAIL] Exception during data fetch: {e}")
        return None

def test_sandbox(df):
    print("\nTesting core/sandbox.py vectorized metrics execution...")
    
    # A dummy strategy that goes long on up-days and short on down-days
    dummy_code = """
def generate_signals(df: pd.DataFrame) -> pd.Series:
    import numpy as np
    
    # Very basic momentum calculation:
    # 1 if today closed higher than yesterday, -1 if lower, else 0
    signals = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return signals
"""
    result = execute_strategy(dummy_code, df)
    if result["success"]:
        print("[OK] Sandbox execution successful.")
        for split in ("train", "test"):
            m = result["metrics"][split]
            print(f"     {split:5s} -> Sharpe: {m['sharpe']:.2f}, Max Drawdown: {m['max_drawdown']*100:.2f}%, "
                  f"Return: {m['total_return']*100:.2f}%, Trades: {m['trades']}")
        print(f"     Lookahead test: {'PASS' if result['lookahead']['passed'] else 'FAIL'}")
    else:
        print("[FAIL] Sandbox execution resulted in an error:")
        print(result["error"])

    print("\nTesting lookahead detection on a strategy that peeks at tomorrow...")
    cheat = """
def generate_signals(df):
    return np.sign(df['Close'].shift(-1) - df['Close'])
"""
    result = execute_strategy(cheat, df)
    if result["success"] and not result["lookahead"]["passed"]:
        print(f"[OK] Lookahead caught: {result['lookahead']['message'][:160]}...")
    else:
        print("[FAIL] Lookahead strategy was not flagged.")


def main():
    print("=== System Integration Test ===")
    load_dotenv()
    
    # 1. Test LLM Connection
    test_connectivity()
    
    # 2. Test Data Fetcher
    df = test_data_fetch()
    
    # 3. Test Sandbox Execution
    if df is not None:
        test_sandbox(df)
        
    print("\nTests complete.")

if __name__ == "__main__":
    main()
