# modeling.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_option_pricing_model(df, features, target):
    """
    Train an XGBoost regression model for option pricing.
    """
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

if __name__ == "__main__":
    # For demonstration, load synthetic model data from CSV
    df = pd.read_csv("stock_data.csv")
    features = ['HV', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'T', 'Moneyness']
    target = 'OptionPrice'
    
    model, mse = train_option_pricing_model(df, features, target)
    print("Trained Model MSE:", mse)
