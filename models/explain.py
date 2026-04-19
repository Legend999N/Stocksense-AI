"""
SHAP integration
Model explainability using SHAP (SHapley Additive exPlanations)
"""

import shap


def explain_prediction(model, X):
    """
    Generate SHAP explanations for model predictions
    
    Args:
        model: Trained model object
        X (pd.DataFrame): Feature matrix
    
    Returns:
        shap.Explainer: SHAP explainer object
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


if __name__ == "__main__":
    pass
