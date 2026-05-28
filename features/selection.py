"""
Feature selection module. Contains tools for dropping correlated variables,
Fisher scoring, Lasso regularization, and Boruta feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def drop_high_corr_features(df: pd.DataFrame, thresh: float = 0.85) -> list:
    """
    Identify and return a list of highly correlated features above a threshold.
    Removes redundancy from multimodal image features.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
    return to_drop


def apply_boruta_selection(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Run Boruta feature selection using an underlying Random Forest Classifier.
    Returns a boolean mask of confirmed significant features.
    """
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=random_state)
    feat_selector.fit(X, y)
    return feat_selector.support_


def apply_lasso_selection(X: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Identify significant features using L1 linear penalty (Lasso regression).
    """
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X, y)
    return lasso.coef_ != 0
