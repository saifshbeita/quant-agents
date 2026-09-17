import yfinance as yf
import pandas as pd

def get_historical_data(ticker: str = "SPY", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical price data and calculates daily market returns.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # Calculate 'Market_Return' as percentage change of 'Close'
    df['Market_Return'] = df['Close'].pct_change()
    
    # Ensure DatetimeIndex and drop NaN values
    df = df.dropna()
    
    return df
