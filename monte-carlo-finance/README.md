# Monte Carlo Finance Simulator

A Monte Carlo simulation tool for financial asset price forecasting, risk analysis, and portfolio optimization — powered by real market data.

## Features

- **Real market data** via Yahoo Finance API (ticker symbol input)
- **Monte Carlo price paths** — thousands of simulated trajectories using geometric Brownian motion
- **Risk metrics** — Value at Risk (VaR), Conditional VaR, max drawdown
- **Confidence bands** — 5th, 25th, 50th, 75th, 95th percentiles over time
- **What-if analysis** — target price probability, stop-loss simulation
- **Interactive dashboard** built with Streamlit + Plotly

## Project Structure

```
monte-carlo-finance/
├── src/
│   ├── data_fetcher.py      # Fetch historical data from Yahoo Finance
│   ├── monte_carlo.py       # GBM simulation engine
│   ├── analysis.py          # Risk & return metrics computation
│   └── visualization.py     # Plotly chart builders
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── notebooks/
│   └── demo.ipynb           # Jupyter walkthrough
├── requirements.txt
├── config.py
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
cd dashboard
streamlit run app.py
```

## Use Cases

- Forecast price range for any publicly traded stock
- Estimate probability of hitting a target price or stop-loss
- Compare risk profiles across assets
- Educational tool for understanding stochastic processes in finance

## Data Source

[Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` library — free, no API key required.
