# feature_engineering.py
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

# -----------------------------
# Stock Data Features
# -----------------------------
def calculate_log_returns(df):
    """
    Calculate log returns based on closing prices.
    """
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    return df

def calculate_hv(df, window=30):
    """
    Calculate historical volatility (annualized) using a rolling window.
    """
    # Multiply by sqrt(252) for annualization (252 trading days per year)
    df['HV'] = df['log_return'].rolling(window=window).std() * np.sqrt(252)
    return df

def add_sma(df, window=20):
    """
    Add a Simple Moving Average (SMA) as a new feature.
    """
    df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def compute_RSI(series, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_rsi(df, period=14):
    """
    Add RSI to the dataframe.
    """
    df[f'RSI_{period}'] = compute_RSI(df['Close'], period)
    return df

# -----------------------------
# Options Data Features
# -----------------------------
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option Greeks: Delta, Gamma, Theta, Vega, and Rho using the Black-Scholes model.
    """
    # Avoid division by zero when T is zero or negative
    if T <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    
    return delta, gamma, theta, vega, rho

def calculate_time_to_expiry(expiry_date, current_date=None):
    """
    Calculate time to expiry in years.
    """
    if current_date is None:
        current_date = datetime.now()
    if isinstance(expiry_date, str):
        expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")
    days_to_expiry = (expiry_date - current_date).days
    T = days_to_expiry / 365.0
    return max(T, 0)  # Ensure non-negative

def calculate_moneyness(S, K):
    """
    Calculate moneyness as the ratio S/K.
    """
    return S / K

# -----------------------------
# Example Pipeline
# -----------------------------
if __name__ == "__main__":
    # For stock data, load sample data (ensure CSV has 'Date' and 'Close' columns)
    data= pd.read_csv("stock_data.csv", parse_dates=['Datetime'])
    data.sort_values('Datetime', inplace=True)
    
    data= calculate_log_returns(data)
    data = calculate_hv(data)
    data = add_sma(data, window=20)
    data = add_rsi(data, period=14)
    print("Sample Stock Features:")
    print(data[['Datetime', 'Close', 'log_return', 'HV', 'SMA_20', 'RSI_14']].tail())

    # For options data, assume example option parameters for demonstration:
    S = 300          # Current stock price
    K = 310          # Strike price
    expiry = "2022-10-15"  # Option expiration date (YYYY-MM-DD)
    r = 0.02         # Risk-free rate (2%)
    sigma = 0.3      # Implied volatility (30%)
    option_type = "call"
    
    T = calculate_time_to_expiry(expiry)
    moneyness = calculate_moneyness(S, K)
    delta, gamma, theta, vega, rho = black_scholes_greeks(S, K, T, r, sigma, option_type)
    
    print("\nExample Option Features:")
    print(f"Time to Expiry (T): {T:.4f} years")
    print(f"Moneyness (S/K): {moneyness:.4f}")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta: {theta:.4f}")
    print(f"Vega: {vega:.4f}")
    print(f"Rho: {rho:.4f}")
