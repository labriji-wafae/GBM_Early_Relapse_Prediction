"""
Custom GridSearch module tailored for Leave-One-Out (LOOCV) and Leave-One-Group-Out (LOGO)
cross-validation under heavy class imbalance in clinical datasets.
"""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class GridSearchLOOCV(BaseEstimator, ClassifierMixin):
    """
    Custom GridSearch that optimizes hyperparameters and choice of decision threshold (tau)
    simultaneously using an inner Stratified K-Fold cross-validation loop.
    Compatible with Scikit-Learn API.
    """
    def __init__(
        self,
        estimator,
        param_grid: dict,
        scoring: dict,
        refit: str = 'auc',
        p_thresh: float = 0.5,
        tau_grid=None,
        tau_metric: str = 'f1_score',
        inner_cv=None,
        random_state: int = 42
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        self.p_thresh = p_thresh
        self.tau_grid = tau_grid if tau_grid is not None else np.linspace(0.1, 0.9, 9)
        self.tau_metric = tau_metric
        self.inner_cv = inner_cv if inner_cv is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        self.random_state = random_state
        
        # Attributes to be populated after fit
        self.results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.best_tau_ = p_thresh

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Execute hyperparameter search with internal cross-validation 
        and optimal decision threshold selection.
        """
        # Générer toutes les combinaisons possibles d'hyperparamètres
        keys, values = zip(*self.param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        results = []
        
        for param_dict in param_combinations:
            # Dictionnaires pour stocker les scores cumulés de l'inner CV
            inner_cv_scores = {metric_name: [] for metric_name in self.scoring.keys()}
            inner_cv_tau_scores = {tau: [] for tau in self.tau_grid}
            
            # Boucle interne pour évaluer cette combinaison précise
            for train_idx, val_idx in self.inner_cv.split(X, y):
                X_train_inner, X_val_inner = X[train_idx], X[val_idx]
                y_train_inner, y_val_inner = y[train_idx], y[val_idx]
                
                # Sécurité : Si le split interne ne contient qu'une seule classe, on ignore ce pli
                if len(np.unique(y_val_inner)) < 2 or len(np.unique(y_train_inner)) < 2:
                    continue
                
                # Instanciation et entraînement du modèle candidat
                clf = clone(self.estimator).set_params(**param_dict)
                clf.fit(X_train_inner, y_train_inner)
                
                # Récupération des probabilités de prédiction (ou de la fonction de décision)
                if hasattr(clf, "predict_proba"):
                    y_score_inner = clf.predict_proba(X_val_inner)[:, 1]
                else:
                    y_score_inner = clf.decision_function(X_val_inner)
                
                # 1. Évaluation des métriques standards basées sur les probabilités (ex: AUC)
                for metric_name, metric_func in self.scoring.items():
                    try:
                        score = metric_func(y_val_inner, y_score_inner)
                        inner_cv_scores[metric_name].append(score)
                    except ValueError:
                        pass  # Sécurité si calcul impossible sur ce sous-échantillon
                
                # 2. Recherche du meilleur seuil de décision tau sur le F1-score interne
                for tau in self.tau_grid:
                    y_pred_tau = (y_score_inner >= tau).astype(int)
                    tau_score = f1_score(y_val_inner, y_pred_tau, zero_division=0)
                    inner_cv_tau_scores[tau].append(tau_score)
            
            # Calcul des moyennes pour chaque paramètre
            mean_scores = {f"{k}_mean": np.mean(v) if v else 0.0 for k, v in inner_cv_scores.items()}
            mean_tau_scores = {tau: np.mean(v) if v else 0.0 for tau, v in inner_cv_tau_scores.items()}
            
            # Trouver le meilleur tau pour cette combinaison d'hyperparamètres
            best_param_tau = self.p_thresh
            if mean_tau_scores:
                best_param_tau = max(mean_tau_scores, key=mean_tau_scores.get)
                
            # Fusionner les informations
            combined_res = {**param_dict, **mean_scores, "optimal_tau": best_param_tau}
            results.append(combined_res)
            
        # Structuration des résultats sous forme de DataFrame Pandas
        self.results_ = pd.DataFrame(results)
        
        # Identification de la meilleure ligne basée sur la métrique de référence (ex: 'auc_mean')
        refit_column = f"{self.refit}_mean"
        if refit_column not in self.results_.columns:
            refit_column = self.results_.columns[-2] # Fallback de secours
            
        best_idx = self.results_[refit_column].idxmax()
        best_row = self.results_.loc[best_idx]
        
        # Extraction des hyperparamètres optimaux définitifs
        self.best_params_ = {k: best_row[k] for k in self.param_grid.keys()}
        self.best_score_ = best_row[refit_column]
        self.best_tau_ = best_row["optimal_tau"]
        
        # Entraînement final du meilleur modèle sur l'ensemble complet des données fournies
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using the tuned optimal threshold (best_tau_)."""
        if self.best_estimator_ is None:
            raise RuntimeError("Estimator has not been fitted yet.")
            
        if hasattr(self.best_estimator_, "predict_proba"):
            scores = self.best_estimator_.predict_proba(X)[:, 1]
        else:
            scores = self.best_estimator_.decision_function(X)
            
        return (scores >= self.best_tau_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities if the underlying model supports it."""
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)
        raise AttributeError("The underlying estimator does not support predict_proba.")
