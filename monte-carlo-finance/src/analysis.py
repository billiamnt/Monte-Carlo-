import numpy as np
import pandas as pd


def value_at_risk(paths: np.ndarray, confidence: float = 0.05):
    final_prices = paths[-1]
    returns = final_prices / paths[0, 0] - 1
    var = float(np.percentile(returns, confidence * 100))
    return var


def conditional_var(paths: np.ndarray, confidence: float = 0.05):
    final_prices = paths[-1]
    returns = final_prices / paths[0, 0] - 1
    threshold = np.percentile(returns, confidence * 100)
    cvar = float(returns[returns <= threshold].mean())
    return cvar


def probability_of_target(paths: np.ndarray, target_price: float):
    final_prices = paths[-1]
    return float((final_prices >= target_price).mean())


def expected_return(paths: np.ndarray):
    final_prices = paths[-1]
    return float(final_prices.mean() / paths[0, 0]) - 1


def max_drawdown(path: np.ndarray):
    peak = np.maximum.accumulate(path)
    dd = (path - peak) / peak
    return float(dd.min())


def average_max_drawdown(paths: np.ndarray):
    dds = np.array([max_drawdown(paths[:, i]) for i in range(paths.shape[1])])
    return float(dds.mean())


def summary_stats(paths: np.ndarray, confidence: float = 0.05):
    var = value_at_risk(paths, confidence)
    cvar = conditional_var(paths, confidence)
    exp_ret = expected_return(paths)
    prob_up = probability_of_target(paths, paths[0, 0])
    avg_dd = average_max_drawdown(paths)

    return {
        "VaR ({}%)".format(int(confidence * 100)): f"{var:.2%}",
        "CVaR ({}%)".format(int(confidence * 100)): f"{cvar:.2%}",
        "Expected Return": f"{exp_ret:.2%}",
        "P(Profit)": f"{prob_up:.2%}",
        "Avg Max Drawdown": f"{avg_dd:.2%}",
    }
