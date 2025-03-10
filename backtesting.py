# backtesting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_signals(df, threshold=0.05):
    """
    Generate trading signals based on the difference between HV and IV.
    +1 if HV > IV + threshold (undervalued option → buy)
    -1 if IV > HV + threshold (overvalued option → sell)
    0 otherwise.
    """
    df['Signal'] = np.where(df['HV'] > df['IV'] + threshold, 1,
                             np.where(df['IV'] > df['HV'] + threshold, -1, 0))
    return df

def backtest_strategy(df):
    """
    Backtest the trading strategy.
    The strategy takes a position based on the previous day's signal and applies it on the change in option price.
    """
    df['Position'] = df['Signal'].shift(1)  # Use previous day's signal
    df['OptionPrice_change'] = df['OptionPrice'].diff()
    df['PnL'] = df['Position'] * df['OptionPrice_change']
    df['Cumulative_PnL'] = df['PnL'].cumsum()
    return df

if __name__ == "__main__":
    # For demonstration, load a sample options dataset with features including 'HV', 'IV', and 'OptionPrice'
    df = pd.read_csv("sample_options_data.csv")
    df = generate_signals(df, threshold=0.05)
    df = backtest_strategy(df)
    
    print("Total PnL:", df['PnL'].sum())
    plt.figure(figsize=(10, 5))
    plt.plot(df['Cumulative_PnL'], label='Cumulative PnL')
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.title("Backtest: Cumulative PnL of Option Trading Strategy")
    plt.legend()
    plt.show()
