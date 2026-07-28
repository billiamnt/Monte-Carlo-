import yfinance as yf
import pandas as pd
import numpy as np


def fetch_historical(ticker: str, start: str = "2020-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df


def compute_log_returns(df: pd.DataFrame, column="close"):
    prices = df[column].squeeze().dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret, prices


def compute_drift_vol(log_returns: pd.Series, dt: float = 1 / 252):
    mu = log_returns.mean() / dt
    sigma = log_returns.std() / np.sqrt(dt)
    return mu, sigma


def get_company_info(ticker: str):
    info = yf.Ticker(ticker).info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
    }
