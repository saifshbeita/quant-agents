import os

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _download(ticker: str, period: str, interval: str, start=None, end=None) -> pd.DataFrame:
    kwargs = dict(interval=interval, progress=False, multi_level_index=False, auto_adjust=True)
    if start is not None:
        kwargs.update(start=str(start), end=str(end) if end is not None else None)
    else:
        kwargs["period"] = period
    return yf.download(ticker, **kwargs)


def _load_csv(ticker: str) -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, f"{ticker.upper()}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"yfinance unavailable and no offline data at {csv_path}")
    head = pd.read_csv(csv_path, nrows=2, header=None)
    if str(head.iloc[1, 0]).strip().lower() == "ticker":
        # Legacy yfinance multi-index export: Price/Ticker/Date header rows.
        df = pd.read_csv(csv_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def get_historical_data(ticker: str = "SPY", period: str = "2y", interval: str = "1d",
                        start=None, end=None) -> pd.DataFrame:
    """
    Fetch daily OHLCV and compute Market_Return. Falls back to data/<TICKER>.csv when
    yfinance fails or returns nothing. df.attrs["source"] records which one was used.
    """
    source = "yfinance"
    try:
        df = _download(ticker, period, interval, start, end)
        if df is None or df.empty:
            raise ValueError("yfinance returned no rows")
    except Exception as e:
        print(f"yfinance fetch failed ({e}); falling back to local CSV.")
        source = "local_csv"
        df = _load_csv(ticker)
        if start is not None:
            df = df.loc[pd.Timestamp(start):]
        if end is not None:
            df = df.loc[:pd.Timestamp(end)]
        elif start is None and period.endswith("mo"):
            df = df.iloc[-int(period[:-2]) * 21:]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLCV if c in df.columns]].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["Market_Return"] = df["Close"].pct_change()
    df = df.dropna()
    if df.empty:
        raise ValueError(f"No usable data for {ticker} in the requested range")
    df.attrs["source"] = source
    df.attrs["ticker"] = ticker.upper()
    return df


def refresh_fallback_csv(ticker: str = "SPY", period: str = "2y") -> str:
    """Re-download the offline fallback CSV (clean single-row header)."""
    df = _download(ticker, period, "1d")
    if df.empty:
        raise ValueError(f"yfinance returned no rows for {ticker}")
    path = os.path.join(DATA_DIR, f"{ticker.upper()}.csv")
    df[OHLCV].to_csv(path, index_label="Date")
    return path


if __name__ == "__main__":
    print(f"Wrote {refresh_fallback_csv()}")
