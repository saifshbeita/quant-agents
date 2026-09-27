import yfinance as yf
import pandas as pd
import os

def get_historical_data(ticker: str = "SPY", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical price data and calculates daily market returns.
    Falls back to local CSV if yfinance fails.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError("No data fetched from yfinance.")
    except Exception as e:
        print(f"yfinance fetch failed: {e}. Falling back to local data.")
        csv_path = os.path.join(os.path.dirname(__file__), f"{ticker}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Local fallback data not found at {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Clean columns in case of yf multi-index saved to csv
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        elif len(df.columns) > 0 and 'Close' not in df.columns:
            # simple attempt to fix columns if saved with multi-index header
            df.columns = [c.split('.')[0] if '.' in c else c for c in df.columns]

    # Calculate 'Market_Return' as percentage change of 'Close'
    df['Market_Return'] = df['Close'].pct_change()
    
    # Ensure DatetimeIndex and drop NaN values
    df = df.dropna()
    
    return df
