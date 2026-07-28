import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def simulation_traces(
    paths: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    sample_size: int = 200,
):
    traces = []
    n = min(sample_size, paths.shape[1])
    idx = np.random.choice(paths.shape[1], n, replace=False)
    for i in idx:
        trace = go.Scatter(
            x=forecast_dates,
            y=paths[:, i],
            mode="lines",
            line=dict(width=0.5, color="rgba(100, 149, 237, 0.15)"),
            showlegend=False,
            name="path",
            hovertemplate="%{y:.2f}<extra></extra>",
        )
        traces.append(trace)
    return traces


def percentile_bands(
    percentiles: dict,
    forecast_dates: pd.DatetimeIndex,
    last_price: float,
):
    bands = []
    fill_colors = {
        5: "rgba(255, 99, 71, 0.08)",
        25: "rgba(100, 149, 237, 0.08)",
        50: "rgba(100, 149, 237, 0.6)",
    }

    for p, vals in percentiles.items():
        if p == 50:
            bands.append(
                go.Scatter(
                    x=forecast_dates,
                    y=vals,
                    mode="lines",
                    name=f"{p}th percentile",
                    line=dict(color="navy", width=2),
                    hovertemplate="%{y:.2f}<extra></extra>",
                )
            )
        else:
            bands.append(
                go.Scatter(
                    x=forecast_dates,
                    y=vals,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate="%{y:.2f}<extra></extra>",
                )
            )

    lower_p = min(percentiles.keys())
    upper_p = max(percentiles.keys())
    bands.append(
        go.Scatter(
            x=forecast_dates,
            y=percentiles[upper_p],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            fill="tonexty",
            fillcolor=fill_colors.get(lower_p, "rgba(0,0,0,0.05)"),
            hovertemplate="%{y:.2f}<extra></extra>",
        )
    )

    return bands


def historical_prices_trace(df: pd.DataFrame):
    prices = df["close"].squeeze()
    return go.Scatter(
        x=prices.index,
        y=prices.values,
        mode="lines",
        name="Historical",
        line=dict(color="green", width=2),
        hovertemplate="%{y:.2f}<extra></extra>",
    )


def distribution_chart(paths: np.ndarray, last_price: float, confidence=0.05):
    final = paths[-1]
    returns = final / last_price - 1

    var = np.percentile(returns, confidence * 100)
    cvar = returns[returns <= var].mean()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=80,
            name="Returns distribution",
            marker_color="rgba(100, 149, 237, 0.6)",
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.add_vline(
        x=var,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR ({int(confidence*100)}%): {var:.2%}",
        annotation_position="top left",
    )

    fig.add_vline(
        x=0,
        line_color="green",
        line_width=1,
        annotation_text="Breakeven",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Simulated Return Distribution",
        xaxis_title="Return",
        yaxis_title="Frequency",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
    )

    return fig


def build_dashboard_figures(
    df: pd.DataFrame,
    paths: np.ndarray,
    percentiles: dict,
    forecast_dates: pd.DatetimeIndex,
    sample_size: int = 200,
    confidence: float = 0.05,
):
    price_panels = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=("Monte Carlo Forecast", "Return Distribution"),
    )

    hist_trace = historical_prices_trace(df)
    sim_traces = simulation_traces(paths, forecast_dates, sample_size)
    pct_traces = percentile_bands(percentiles, forecast_dates, float(df["close"].squeeze().iloc[-1]))

    price_panels.add_trace(hist_trace, row=1, col=1)
    for t in sim_traces:
        price_panels.add_trace(t, row=1, col=1)
    for t in pct_traces:
        price_panels.add_trace(t, row=1, col=1)

    price_panels.update_xaxes(title_text="Date", row=2, col=1)
    price_panels.update_yaxes(title_text="Price ($)", row=1, col=1)
    price_panels.update_layout(
        title="",
        height=700,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return price_panels
