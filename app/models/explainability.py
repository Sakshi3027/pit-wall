import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier

def get_shap_explanation(model: XGBClassifier,
                          race_features: pd.DataFrame,
                          feature_names: list) -> pd.DataFrame:
    """
    Returns SHAP values for each driver's prediction.
    Shows which features drove the podium probability up or down.
    """
    X = race_features[feature_names].fillna(0).values
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    rows = []
    for i, driver in enumerate(race_features["driver"]):
        for j, feat in enumerate(feature_names):
            rows.append({
                "driver":     driver,
                "feature":    feat,
                "shap_value": round(float(shap_vals[i][j]), 4),
                "direction":  "positive" if shap_vals[i][j] > 0 else "negative",
            })
    return pd.DataFrame(rows)

def get_top_factors(shap_df: pd.DataFrame,
                    driver: str, top_n: int = 5) -> pd.DataFrame:
    """Returns top N factors (positive and negative) for a specific driver."""
    d = shap_df[shap_df["driver"] == driver].copy()
    d["abs_shap"] = d["shap_value"].abs()
    return d.nlargest(top_n, "abs_shap").reset_index(drop=True)