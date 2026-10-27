"""
src/models/hurdle.py
---------------------
Two-Stage Hurdle Model for appliance energy forecasting.

Stage 1 (classifier): probability that the interval is an active burst
                      (Appliances > active_threshold_wh) rather than standby.
Stage 2 (regressor) : trained only on active intervals; forecasts burst size.

Routing: intervals classified as active use the Stage 2 forecast; standby
intervals use persistence (Appliances_lag_1), which is near-exact while the
house idles. Configuration chosen on the validation split (docs/model_card.md).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

PERSISTENCE_FEATURE = "Appliances_lag_1"


def default_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def default_regressor():
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=200, random_state=42,
    )


class HurdleModel(RegressorMixin, BaseEstimator):
    """Standby/active classifier followed by an active-burst regressor."""

    def __init__(
        self,
        classifier=None,
        regressor=None,
        active_threshold_wh: float = 100.0,
        decision_threshold: float = 0.5,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.active_threshold_wh = active_threshold_wh
        self.decision_threshold = decision_threshold

    def fit(self, X: pd.DataFrame, y):
        if PERSISTENCE_FEATURE not in X.columns:
            raise ValueError(f"X must contain '{PERSISTENCE_FEATURE}' for the standby route")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be in (0, 1)")

        y = np.asarray(y, dtype=float)
        active = y > self.active_threshold_wh
        if active.all() or not active.any():
            raise ValueError(
                f"active_threshold_wh={self.active_threshold_wh} leaves only one class in y"
            )

        self.classifier_ = clone(self.classifier or default_classifier()).fit(X, active.astype(int))
        self.regressor_ = clone(self.regressor or default_regressor()).fit(X[active], y[active])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.active_rate_ = float(active.mean())
        return self

    def predict_active_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Stage 1: probability that each interval is an active burst."""
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        is_active = self.predict_active_proba(X) >= self.decision_threshold
        standby = X[PERSISTENCE_FEATURE].to_numpy(dtype=float)
        preds = standby.copy()
        if is_active.any():
            preds[is_active] = self.regressor_.predict(X[is_active])
        return np.clip(preds, 0.0, None)
