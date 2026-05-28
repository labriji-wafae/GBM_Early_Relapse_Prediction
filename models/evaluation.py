"""
Evaluation module for patient-wise cross-validation (Leave-One-Group-Out)
and clinical classification scoring (Sensitivity, Specificity, AUC, AUPRC).
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, average_precision_score
from sklearn.base import clone
from data.data_balancing import get_balanced_data

logger = logging.getLogger(__name__)


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Sensitivity (True Positive Rate / Recall). Safe against empty classes."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    elif len(np.unique(y_true)) == 1 and np.unique(y_true)[0] == 1:
        # Uniquement la classe positive présente dans le test
        return float(np.sum(y_pred == 1) / len(y_true))
    return 0.0


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Specificity (True Negative Rate). Safe against empty classes."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    elif len(np.unique(y_true)) == 1 and np.unique(y_true)[0] == 0:
        # Uniquement la classe négative présente dans le test
        return float(np.sum(y_pred == 0) / len(y_true))
    return 0.0


def run_patient_cross_validation(X, y, groups, base_estimator, balancing_method='resample') -> pd.DataFrame:
    """
    Execute a strict Leave-One-Group-Out (LOGO) cross-validation where each group 
    represents an individual patient. Ensures NO data leakage.
    """
    unique_patients = np.unique(groups)
    results = []
    
    logger.info(f"Starting Patient-wise Cross-Validation ({len(unique_patients)} unique patients)...")

    for patient in unique_patients:
        # Split : Le patient en cours sert de Test, tous les autres servent de Train
        train_mask = (groups != patient)
        test_mask = (groups == patient)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Sécurité : Vérifier que le jeu de test contient bien des données
        if len(y_test) == 0:
            continue
            
        # ⚠️ CRUCIAL : L'équilibrage des données est appliqué UNIQUEMENT sur le Train set
        # pour éviter le data leakage (surapprentissage artificiel)
        X_train_balanced, y_train_balanced = get_balanced_data(
            X_train, y_train, method=balancing_method
        )
        
        # Cloner l'estimateur pour repartir d'un modèle vierge à chaque patient
        clf = clone(base_estimator)
        
        try:
            # Entraînement (le fit va déclencher le GridSearch interne sur le bloc balanced)
            clf.fit(X_train_balanced, y_train_balanced)
            
            # Prédictions brutes et probabilités
            y_pred = clf.predict(X_test)
            
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_prob = clf.decision_function(X_test)
            else:
                y_prob = y_pred
                
            # Calcul des métriques sur ce patient
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else 0.5
            auprc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else 0.0
            f1 = f1_score(y_test, y_pred, zero_division=0)
            sens = sensitivity_score(y_test, y_pred)
            spec = specificity_score(y_test, y_pred)
            
            results.append({
                "patient": patient,
                "auc": auc,
                "auprc": auprc,
                "f1_score": f1,
                "sensitivity": sens,
                "specificity": spec,
                "voxel_count": len(y_test),
                "relapse_ratio": float(np.sum(y_test == 1) / len(y_test))
            })
            logger.info(f"Patient {patient} processed successfully. AUC: {auc:.3f} | F1: {f1:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing patient {patient}: {str(e)}")
            continue

    return pd.DataFrame(results)
