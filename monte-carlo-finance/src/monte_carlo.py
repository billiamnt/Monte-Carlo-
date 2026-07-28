import numpy as np
import pandas as pd


def gbm_simulation(
    last_price: float,
    mu: float,
    sigma: float,
    n_simulations: int = 2000,
    forecast_horizon: int = 252,
    dt: float = 1 / 252,
    seed: int = 42,
):
    np.random.seed(seed)
    paths = np.zeros((forecast_horizon + 1, n_simulations))
    paths[0] = last_price

    for t in range(1, forecast_horizon + 1):
        z = np.random.standard_normal(n_simulations)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        paths[t] = paths[t - 1] * np.exp(drift + diffusion)

    return paths


def simulate_with_returns(
    last_price: float,
    log_returns: pd.Series,
    n_simulations: int = 2000,
    forecast_horizon: int = 252,
    seed: int = 42,
):
    """Bootstrap from historical returns (non-parametric)."""
    np.random.seed(seed)
    paths = np.zeros((forecast_horizon + 1, n_simulations))
    paths[0] = last_price
    returns = log_returns.values

    for t in range(1, forecast_horizon + 1):
        sample = np.random.choice(returns, size=n_simulations)
        paths[t] = paths[t - 1] * np.exp(sample)

    return paths


def to_dataframe(paths: np.ndarray, price_dates: pd.DatetimeIndex):
    end = price_dates[-1]
    forecast_idx = pd.bdate_range(start=end, periods=paths.shape[0])
    return pd.DataFrame(paths, index=forecast_idx)


def compute_percentiles(paths: np.ndarray, pcts=(5, 25, 50, 75, 95)):
    return {f"p{p}": np.percentile(paths, p, axis=1) for p in pcts}
