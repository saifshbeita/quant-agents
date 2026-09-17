import os
import requests
import pandas as pd
from dotenv import load_dotenv
from config import LLM_BASE_URL, TARGET_TICKER
from data.fetcher import get_historical_data
from core.sandbox import execute_strategy

def test_connectivity():
    print(f"Testing connectivity to LLM server at {LLM_BASE_URL}...")
    try:
        # Check standard OpenAI models endpoint
        url = LLM_BASE_URL if not LLM_BASE_URL.endswith('/') else LLM_BASE_URL[:-1]
        response = requests.get(f"{url}/models", timeout=3)
        if response.status_code == 200:
            print("[OK] Successfully connected to LLM server.")
            print(f"     Available models endpoint responded with 200 OK.")
        else:
            print(f"[WARN] Connected, but returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Could not connect to LLM server.")
        print(f"       Details: {e}")
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
        metrics = result['metrics']
        print(f"     Computed Metrics -> Sharpe: {metrics.get('sharpe', 0):.2f}, "
              f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%, "
              f"Return: {metrics.get('total_return', 0)*100:.2f}%")
    else:
        print("[FAIL] Sandbox execution resulted in an error:")
        print(result["error"])

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
