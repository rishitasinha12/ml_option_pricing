# tuning.py
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def objective(trial):
    df = pd.read_csv("sample_model_data.csv")
    features = ['HV', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'T', 'moneyness']
    target = 'OptionPrice'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0)
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print("  MSE:", trial.value)
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
