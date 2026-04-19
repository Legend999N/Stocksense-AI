"""
XGBoost training
Train XGBoost model for stock price prediction
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib


def train_model(X, y, model_path="model.pkl"):
    """
    Train XGBoost model
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model_path (str): Path to save the model
    
    Returns:
        xgb.XGBRegressor: Trained model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    return model


if __name__ == "__main__":
    pass
