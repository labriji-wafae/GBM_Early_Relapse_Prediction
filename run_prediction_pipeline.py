"""
Main pipeline entrypoint for GBM Early Relapse Prediction.
Handles dataset loading, feature correlation cleaning, feature selection benchmarks,
and automated patient-wise cross-validation (Leave-One-Group-Out).
"""

import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Modèles d'évaluation de Scikit-Learn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


from features.selection import drop_high_corr_features, apply_boruta_selection, apply_lasso_selection
from models.gridsearch_custom import GridSearchLOOCV
from models.evaluation import run_patient_cross_validation

# Configuration globale du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Configure CLI arguments for reproducible machine learning runs."""
    parser = argparse.ArgumentParser(description="GBM Early Relapse Prediction Pipeline from initial timepoint (M0).")
    
    parser.add_argument("--data_path", type=str, default="./dataframe/MsrGB_features_M0.pkl",
                        help="Path to the input pickle DataFrame containing multimodal features.")
    parser.add_argument("--fs_method", type=str, default="boruta", choices=["boruta", "lasso", "none"],
                        help="Feature selection algorithm to execute (default: boruta).")
    parser.add_argument("--balancing", type=str, default="smote", choices=["smote", "resample", "adasyn", "none"],
                        help="Class imbalance resampling technique for train fold (default: smote).")
    parser.add_argument("--corr_thresh", type=float, default=0.85,
                        help="Correlation coefficient limit to drop redundant features (default: 0.85).")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Chargement sécurisé de la base de données de caractéristiques (Features M0)
    if not os.path.exists(args.data_path):
        logger.error(f"Input database not found at {args.data_path}. Please check your local data paths.")
        return
        
    logger.info(f"Loading multimodal predictive database from: {args.data_path}")
    df = pd.read_pickle(args.data_path)
    
    # Prétraitement de base : filtrage de la qualité du signal (ex: R2 de spectroscopie > 0.6)
    if 'R2' in df.columns:
        initial_count = len(df)
        df = df[df['R2'] > 0.6]
        logger.info(f"Filtered out low-quality spectra (R2 > 0.6). Kept {len(df)}/{initial_count} voxels.")

    # 2. Séparation des variables de contrôle et de la cible (relapse : 1 si rechute précoce, 0 sinon)
    target_col = 'relapse'
    group_col = 'patientName'  # Identifiant unique de chaque patient pour le Leave-One-Group-Out
    
    if target_col not in df.columns or group_col not in df.columns:
        logger.error(f"Required columns '{target_col}' or '{group_col}' missing from the input DataFrame.")
        return

    # Identification dynamique des colonnes de features numériques (exclure métadonnées de chaînes)
    metadata_cols = [group_col, 'index', 'seg_init', 'patient_id', 'exam_id']
    feature_cols = [col for col in df.columns if col not in metadata_cols and col != target_col]
    
    X_df = df[feature_cols]
    y = df[target_col].values
    groups = df[group_col].values

    # 3. Étape de Génie Logiciel : Élimination des corrélations fortes (Redondance Multimodale)
    logger.info(f"Initial feature space vector dimensions: {X_df.shape[1]} columns.")
    correlated_features = drop_high_corr_features(X_df, thresh=args.corr_thresh)
    X_df_cleaned = X_df.drop(columns=correlated_features)
    logger.info(f"Dropped {len(correlated_features)} highly correlated features. Remaining: {X_df_cleaned.shape[1]}.")

    X = X_df_cleaned.values

    # 4. Sélection de caractéristiques supervisée (Feature Selection) sur l'espace nettoyé
    logger.info(f"Executing superviser feature selection method: '{args.fs_method}'")
    if args.fs_method == "boruta":
        selected_mask = apply_boruta_selection(X, y)
        X = X[:, selected_mask]
        logger.info(f"Boruta confirmed {np.sum(selected_mask)} significant predictors.")
    elif args.fs_method == "lasso":
        selected_mask = apply_lasso_selection(X, y, alpha=0.01)
        X = X[:, selected_mask]
        logger.info(f"Lasso constraint kept {np.sum(selected_mask)} non-zero coefficients.")
    else:
        logger.info("Skipping advanced feature selection.")

    # Vérification de sécurité s'il ne reste aucune caractéristique
    if X.shape[1] == 0:
        logger.error("No features were selected. Algorithm cannot proceed.")
        return

    # 5. Configuration de l'Estimateur Principal et de la Grille de Recherche Customisée
    # Nous utilisons un SVM à noyau RBF (Support Vector Classifier), très efficace sur les petits échantillons cliniques
    base_svc = Pipeline([
        ('scaler', RobustScaler()),  # Robust aux valeurs aberrantes de spectroscopie
        ('svc', SVC(probability=True, class_weight='balanced', random_state=42))
    ])
    
    # Définition de la grille des hyperparamètres pour le SVM
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    # Dictionnaire des métriques de validation interne (inner-cv)
    scoring_metrics = {
        'auc': roc_auc_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }

    # Instanciation de notre classe de GridSearch optimisée avec le seuil tau dynamique
    custom_gridsearch = GridSearchLOOCV(
        estimator=base_svc,
        param_grid=param_grid,
        scoring=scoring_metrics,
        refit='auc',
        p_thresh=0.5
    )

    # 6. Lancement de la Validation Croisée par Patient (Leave-One-Subject-Out)
    cv_results_df = run_patient_cross_validation(
        X=X, 
        y=y, 
        groups=groups, 
        base_estimator=custom_gridsearch, 
        balancing_method=args.balancing
    )

    # 7. Affichage et Exportation des Performances Cliniques Globales
    logger.info("\n==================================================")
    logger.info("       PIPELINE CROSS-VALIDATION SUMMARY          ")
    logger.info("==================================================")
    logger.info(f"Mean Patient-wise AUC         : {cv_results_df['auc'].mean():.3f} ± {cv_results_df['auc'].std():.3f}")
    logger.info(f"Mean Patient-wise F1-Score    : {cv_results_df['f1_score'].mean():.3f} ± {cv_results_df['f1_score'].std():.3f}")
    logger.info(f"Mean Patient-wise Sensitivity : {cv_results_df['sensitivity'].mean():.3f} ± {cv_results_df['sensitivity'].std():.3f}")
    logger.info(f"Mean Patient-wise Specificity : {cv_results_df['specificity'].mean():.3f} ± {cv_results_df['specificity'].std():.3f}")
    logger.info("==================================================")

    # Sauvegarde optionnelle des scores par patient
    output_report_path = "./patient_evaluation_report.csv"
    cv_results_df.to_csv(output_report_path, index=False)
    logger.info(f"Detailed patient performance report saved to {output_report_path}")


if __name__ == "__main__":
    main()

