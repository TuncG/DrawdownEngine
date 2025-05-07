import numpy as np
import pandas as pd
import yfinance as yf               # quick data source
from arch import arch_model

# ------------------------------
# 1.  Load data
# ------------------------------
def get_data(ticker, start):
    """Load adjusted close prices from Yahoo Finance."""
    prices = yf.download(ticker, start=start, progress=False)["Adj Close"]
    return prices

def get_log_returns(prices):
    """Calculate log-returns from prices."""
    logret = np.log(prices).diff().dropna() * 100   # % log‑returns (scale helps with numbers)
    return logret


def simulate_garch(logret):
    """Simulate GARCH paths and calculate MDaR and CED."""

    # ------------------------------
    # 2.  Fit ARMA(1,1)-GJR-GARCH(1,1)-t
    # ------------------------------
    am = arch_model(logret,
                    mean="ARX", lags=1, vol="GARCH", p=1, o=1, q=1,
                    dist="t")
    res = am.fit(update_freq=0, disp="off")
    print(res.summary())

    # ------------------------------
    # 3.  Simulate paths
    # ------------------------------
    horizon  = 252               # 1‑year
    n_paths  = 10_000
    sim = res.simulate(res.params, nobs=horizon, repetitions=n_paths)

    # sim["data"] is an (horizon x n_paths) array of simulated returns
    ret_paths = sim['data'].T / 100.0          # back to raw log‑return units

    # ------------------------------
    # 4.  Convert to price paths
    # ------------------------------
    p0 = prices.iloc[-1]                       # start each path at last real price
    price_paths = p0 * np.exp(ret_paths.cumsum(axis=1))

    return price_paths

def max_drawdown(path):
        cummax = np.maximum.accumulate(path)
        drawdown = cummax - path
        return drawdown.max()

def main():
    # Load data
    ticker = "AAPL"
    start_date = "2010-01-01"
    prices = get_data(ticker, start_date)

    # Calculate log-returns
    logret = get_log_returns(prices)

    price_paths = simulate_garch(logret)

    

    mdds = np.fromiter((max_drawdown(p) for p in price_paths), float)

    alpha  = 0.95
    mdar   = np.quantile(mdds, alpha)                 # MDaR (drawdown‑VaR)
    ced    = mdds[mdds > mdar].mean()                # CED (drawdown‑ES)
    print(f"MDaR 95%  : {mdar:.2f}")
    print(f"CED  beyond: {ced:.2f}")