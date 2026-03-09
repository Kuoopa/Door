from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ResidualWindCorrector:
    def __init__(self, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=420,
            max_depth=18,
            min_samples_split=6,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        self.feature_importances_ = None

    def fit(self, X, y_true, measured_wind):
        residual = np.asarray(y_true) - np.asarray(measured_wind)
        self.model.fit(X, residual)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X, measured_wind):
        pred_residual = self.model.predict(X)
        return np.asarray(measured_wind) + pred_residual



def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
