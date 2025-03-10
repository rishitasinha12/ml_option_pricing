import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period="2y", interval="1h"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

def fetch_options_data(ticker, expiration_index=0, period = '2y', interval = '1h'):      
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise Exception("No options data available")
    opt_chain = stock.option_chain(expirations[expiration_index])
    calls = opt_chain.calls
    puts = opt_chain.puts
    calls['Type'] = 'Call'
    puts['Type'] = 'Put'
    calls['Ticker'] = ticker
    puts['Ticker'] = ticker
    return pd.concat([calls, puts])
    
if __name__ == "__main__":
    ticker = "NVDA"
    df_stock = fetch_stock_data(ticker)
    df_stock.to_csv("stock_data.csv", index=False)
    print('stock data loaded')
    print('-----stock_data_shape-----:', df_stock.shape)
    df_options = fetch_options_data(ticker)
    df_options.to_csv("options_data.csv", index=False)
    print('options data loaded')
    print('----options_data_shape----:', df_options.shape)

     