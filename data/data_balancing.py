"""
Data balancing module for addressing heavy class imbalance in voxel-wise clinical datasets.
Provides standard resampling and advanced synthetic over-sampling techniques (SMOTE, ADASYN).
"""

import numpy as np
import logging
from sklearn.utils import resample

# Importation sécurisée des outils imblearn
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
except ImportError:
    # Fallback si imblearn n'est pas disponible (évite le plantage immédiat au chargement)
    SMOTE = ADASYN = BorderlineSMOTE = KMeansSMOTE = SVMSMOTE = None

logger = logging.getLogger(__name__)


def get_balanced_data(X_train: np.ndarray, y_train: np.ndarray, method: str = 'resample') -> tuple:
    """
    Apply over-sampling methods to balance minority and majority classes.
    Includes fallback mechanisms to guarantee pipeline execution under corner-case distributions.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    
    # Sécurité : S'il n'y a qu'une seule classe dans le bloc, aucun équilibrage n'est possible
    if len(unique) < 2:
        return X_train, y_train

    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    
    X_train_minority = X_train[y_train == minority_class]
    y_train_minority = y_train[y_train == minority_class]
    X_train_majority = X_train[y_train == majority_class]
    y_train_majority = y_train[y_train == majority_class]
    
    normalized_method = str(method).lower().strip()
    if normalized_method in ['none', 'pas d\'équilibrage']:
        return X_train, y_train

    # --- MÉTHODE 1 : Resample natif Scikit-Learn (Robuste et sans dépendances) ---
    if normalized_method == 'resample' or SMOTE is None:
        if normalized_method != 'resample' and SMOTE is None:
            logger.warning("imblearn library is missing. Falling back to naive sklearn resampling.")
            
        X_train_minority_upsampled, y_train_minority_upsampled = resample(
            X_train_minority, y_train_minority, 
            replace=True, 
            n_samples=len(y_train_majority), 
            random_state=42
        )
        X_train_balanced = np.vstack((X_train_majority, X_train_minority_upsampled))
        y_train_balanced = np.hstack((y_train_majority, y_train_minority_upsampled))
        return X_train_balanced, y_train_balanced

    # --- MÉTHODES 2 : Techniques de suréchantillonnage avancées (imblearn) ---
    try:
        if normalized_method == 'smote':
            sampler = SMOTE(random_state=42)
        elif normalized_method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif normalized_method == 'borderlinesmote':
            sampler = BorderlineSMOTE(random_state=42)
        elif normalized_method == 'kmeanssmote':
            sampler = KMeansSMOTE(random_state=42)
        elif normalized_method == 'svmsmote':
            sampler = SVMSMOTE(random_state=42)
        else:
            logger.warning(f"Unknown balancing method '{method}'. Using default sklearn resample.")
            return get_balanced_data(X_train, y_train, method='resample')

        X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
        return X_train_balanced, y_train_balanced

    except Exception as e:
        # Si SMOTE échoue, on bascule sur resample
        logger.error(f"Advanced balancing method '{method}' failed due to mathematical constraints: {str(e)}. "
                     f"Engaging fallback strategy: naive resampling.")
        return get_balanced_data(X_train, y_train, method='resample')
