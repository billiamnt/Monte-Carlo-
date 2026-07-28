import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_fetcher import fetch_historical, compute_log_returns, compute_drift_vol, get_company_info
from src.monte_carlo import gbm_simulation, simulate_with_returns, compute_percentiles
from src.analysis import summary_stats, probability_of_target, value_at_risk
from src.visualization import build_dashboard_figures, distribution_chart

st.set_page_config(page_title="Monte Carlo Finance", layout="wide")

st.title("Monte Carlo Finance Simulator")
st.markdown("Simulate thousands of possible price paths using real market data.")

with st.sidebar:
    st.header("Parameters")

    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
    n_simulations = st.slider("Simulations", 500, 5000, 2000, step=100)
    forecast_days = st.slider("Forecast Horizon (trading days)", 21, 756, 252, step=21)

    simulation_method = st.radio("Simulation Method", ["GBM (Parametric)", "Bootstrap (Non-parametric)"])

    confidence = st.select_slider("VaR Confidence Level", options=[0.01, 0.025, 0.05, 0.10], value=0.05)

    show_distribution = st.checkbox("Show Return Distribution", value=True)
    run_btn = st.button("Run Simulation", type="primary")

if run_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            df = fetch_historical(ticker, start=str(start_date))
            info = get_company_info(ticker)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    st.success(f"Loaded {len(df)} trading days of data.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Company", info["name"])
    col2.metric("Sector", info["sector"])
    col3.metric("Industry", info["industry"])
    last_close_val = float(df["close"].squeeze().iloc[-1])
    col4.metric("Last Close", f"${last_close_val:.2f}")

    log_returns, prices = compute_log_returns(df)
    mu, sigma = compute_drift_vol(log_returns)
    last_price = float(prices.iloc[-1])

    with st.spinner(f"Running {n_simulations:,} simulations..."):
        if simulation_method == "GBM (Parametric)":
            paths = gbm_simulation(
                last_price=last_price,
                mu=mu,
                sigma=sigma,
                n_simulations=n_simulations,
                forecast_horizon=forecast_days,
            )
        else:
            paths = simulate_with_returns(
                last_price=last_price,
                log_returns=log_returns,
                n_simulations=n_simulations,
                forecast_horizon=forecast_days,
            )

    percentiles = compute_percentiles(paths)
    stats = summary_stats(paths, confidence=confidence)

    last_close = df["close"].squeeze()
    last_dates = last_close.index
    forecast_dates = pd.bdate_range(start=last_dates[-1], periods=forecast_days + 1)

    st.subheader("Risk Metrics")
    metric_cols = st.columns(len(stats))
    for col, (label, value) in zip(metric_cols, stats.items()):
        col.metric(label, value)

    with st.expander("What-If Analysis"):
        target = st.number_input("Target Price ($)", value=round(last_price * 1.2, 2), step=1.0)
        prob = probability_of_target(paths, target)
        prob_var = value_at_risk(paths, confidence)

        st.metric(
            f"P(Price ≥ ${target:,.2f})",
            f"{prob:.1%}",
            delta=f"{prob - 0.5:.1%} vs 50%",
        )

        st.caption(
            f"Interpretation: there is a **{prob:.1%}** chance the price reaches "
            f"**${target:,.2f}** within {forecast_days} trading days."
        )

    st.subheader("Forecast Visualization")
    fig = build_dashboard_figures(
        df=df,
        paths=paths,
        percentiles=percentiles,
        forecast_dates=forecast_dates,
        sample_size=300,
        confidence=confidence,
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_distribution:
        st.subheader("Return Distribution at Horizon")
        dist_fig = distribution_chart(paths, last_price, confidence=confidence)
        st.plotly_chart(dist_fig, use_container_width=True)

    st.subheader("Raw Data Preview")
    preview = pd.DataFrame(
        {
            f"Sim {i+1}": paths[:, i]
            for i in range(min(10, paths.shape[1]))
        },
        index=forecast_dates,
    )
    preview.index.name = "Date"
    st.dataframe(preview.head(20), use_container_width=True)

    csv = preview.to_csv().encode("utf-8")
    st.download_button("Download Simulation Data", data=csv, file_name=f"{ticker}_mc_simulation.csv", mime="text/csv")

else:
    st.info("Enter parameters in the sidebar and click **Run Simulation** to start.")
