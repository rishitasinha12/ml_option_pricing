# ML-Driven Option Pricing and Trading Strategy

## Overview

This project aims to predict option prices using machine learning (XGBoost) and develop a trading strategy based on historical volatility (HV) and implied volatility (IV). The model incorporates financial indicators such as option Greeks, moneyness, and historical volatility to improve pricing accuracy.

## Project Structure

```
ml_option_pricing/
├── data_acquisition.py      # Fetch stock and options data using yfinance
├── feature_engineering.py   # Compute log returns, volatility, Greeks, time to expiry, moneyness, SMA, RSI
├── modeling.py              # Train an ML model (XGBoost) to predict option prices
├── backtesting.py           # Generate trading signals and backtest the strategy
├── tuning.py                # (Optional) Hyperparameter tuning using Optuna
├── sentiment.py             # (Optional) Integrate news sentiment analysis
├── app.py                   # Streamlit dashboard for interactive visualization
├── requirements.txt         # List of project dependencies
└── README.md                # Project overview, setup instructions, and usage
```

## Features

- Fetch real-time stock and options data from Yahoo Finance (yfinance).
- Compute financial indicators such as historical volatility, SMA, RSI, and option Greeks.
- Predict option prices using an ML model (XGBoost).
- Develop a trading strategy based on volatility differences.
- Backtest the strategy with historical data.
- (Optional) Integrate news sentiment analysis to refine predictions.
- (Optional) Interactive visualization using Streamlit.

## Installation & Setup

1. **Clone the Repository**
    ```sh
    git clone https://github.com/yourusername/ml_option_pricing.git
    cd ml_option_pricing
    ```

2. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Fetch Stock and Options Data**
    ```python
    from data_acquisition import fetch_stock_data, fetch_options_data

    ticker = "NVDA"
    df_stock = fetch_stock_data(ticker, period="1y", interval="1h")
    df_options = fetch_options_data(ticker)
    ```

2. **Feature Engineering**
    ```python
    from feature_engineering import calculate_log_returns, calculate_hv, add_sma, add_rsi

    df_stock = calculate_log_returns(df_stock)
    df_stock = calculate_hv(df_stock)
    df_stock = add_sma(df_stock)
    df_stock = add_rsi(df_stock)
    ```

3. **Train an ML Model for Option Pricing**
    ```python
    from modeling import train_option_pricing_model

    features = ['HV', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'T', 'moneyness']
    target = 'OptionPrice'
    model, mse = train_option_pricing_model(df_options, features, target)
    print("Model trained with MSE:", mse)
    ```

4. **Generate Trading Signals**
    ```python
    from backtesting import generate_signals, backtest_strategy

    df_signals = generate_signals(df_options)
    df_backtest = backtest_strategy(df_signals)
    ```

## Backtesting Strategy

- **Buy Signal (+1)**: If HV is significantly higher than IV, implying an undervalued option.
- **Sell Signal (-1)**: If IV is significantly higher than HV, implying an overvalued option.
- **Backtesting**: Apply signals to historical option price changes and analyze profitability.

## Future Improvements

- Add deep learning models (LSTMs) for price prediction.
- Improve sentiment analysis for better decision-making.
- Enhance feature selection with automated techniques.
- Deploy as a web application using Flask/Streamlit.

## Contributing

Feel free to submit issues or pull requests to improve the project.

## License

This project is licensed under the MIT License.