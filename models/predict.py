"""
Inference
Make predictions using trained model
"""

import joblib
import pandas as pd


def load_model(model_path="model.pkl"):
    """
    Load trained model from disk
    
    Args:
        model_path (str): Path to model file
    
    Returns:
        Trained model object
    """
    model = joblib.load(model_path)
    return model


def predict(model, X):
    """
    Make predictions on new data
    
    Args:
        model: Trained model object
        X (pd.DataFrame): Feature matrix
    
    Returns:
        np.ndarray: Predictions
    """
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    pass
