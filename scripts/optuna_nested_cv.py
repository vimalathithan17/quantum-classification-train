#!/usr/bin/env python
"""
Optuna Nested Cross-Validation for Quantum Classifiers.

Implements nested CV with:
- Outer folds for model evaluation
- Inner folds for hyperparameter tuning
- SQLite study persistence
- TPE sampler and MedianPruner
- SMALL_STEPS for tuning, FULL_STEPS for final training
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import log
from qml_models import MulticlassQuantumClassifierDR


def safe_load_parquet(file_path):
    """Loads a parquet file with increased thrift limits."""
    limit = 1 * 1024**3
    try:
        return pd.read_parquet(
            file_path,
            thrift_string_size_limit=limit,
            thrift_container_size_limit=limit
        )
    except Exception as e:
        log.error(f"Error loading {file_path}: {e}")
        return None


def objective_inner_cv(trial, X_train, y_train, n_classes, n_inner_folds, small_steps):
    """
    Objective function for inner CV hyperparameter tuning.
    """
    # Suggest hyperparameters
    n_qubits = trial.suggest_int('n_qubits', n_classes, 12)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 8, 32)
    
    # Inner CV
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    inner_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Preprocess
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        # PCA
        pca = PCA(n_components=min(n_qubits, X_tr_scaled.shape[1]), random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        # Pad if needed
        if X_tr_pca.shape[1] < n_qubits:
            padding = np.zeros((X_tr_pca.shape[0], n_qubits - X_tr_pca.shape[1]))
            X_tr_pca = np.hstack([X_tr_pca, padding])
            padding_val = np.zeros((X_val_pca.shape[0], n_qubits - X_val_pca.shape[1]))
            X_val_pca = np.hstack([X_val_pca, padding_val])
        
        # Train with SMALL_STEPS
        model = MulticlassQuantumClassifierDR(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=n_classes,
            learning_rate=learning_rate,
            steps=small_steps,
            hidden_dim=hidden_dim,
            verbose=False
        )
        
        model.fit(X_tr_pca, y_tr)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_pca)
        score = f1_score(y_val, y_val_pred, average='weighted')
        inner_scores.append(score)
        
        # Report intermediate value for pruning
        trial.report(score, fold_idx)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return mean validation score
    return np.mean(inner_scores)


def main():
    parser = argparse.ArgumentParser(description="Nested CV with Optuna for quantum classifiers")
    parser.add_argument('--data_path', type=str, required=True, help="Path to parquet data file")
    parser.add_argument('--output_dir', type=str, default='./nested_cv_results',
                       help="Output directory for results")
    parser.add_argument('--sqlite_path', type=str, default='./optuna_studies.db',
                       help="SQLite database path for Optuna studies")
    parser.add_argument('--study_name', type=str, default='quantum_nested_cv',
                       help="Optuna study name")
    parser.add_argument('--n_outer', type=int, default=5, help="Number of outer CV folds")
    parser.add_argument('--n_inner', type=int, default=3, help="Number of inner CV folds")
    parser.add_argument('--n_trials', type=int, default=20, help="Number of Optuna trials per outer fold")
    parser.add_argument('--small_steps', type=int, default=30, help="Training steps for inner tuning")
    parser.add_argument('--full_steps', type=int, default=100, help="Training steps for outer evaluation")
    parser.add_argument('--metric', type=str, default='f1_weighted', 
                       choices=['f1_weighted', 'f1_macro', 'accuracy'],
                       help="Metric for model selection")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    log.info(f"Nested CV with Optuna")
    log.info(f"Data: {args.data_path}")
    log.info(f"Output: {args.output_dir}")
    log.info(f"SQLite: {args.sqlite_path}")
    log.info(f"Outer folds: {args.n_outer}, Inner folds: {args.n_inner}")
    log.info(f"Trials per outer fold: {args.n_trials}")
    log.info(f"Small steps: {args.small_steps}, Full steps: {args.full_steps}")
    
    # Load data
    log.info("Loading data...")
    df = safe_load_parquet(args.data_path)
    if df is None:
        log.error("Failed to load data")
        return 1
    
    # Separate features and labels
    if 'class' not in df.columns:
        log.error("Data must have a 'class' column")
        return 1
    
    y = df['class'].values
    feature_cols = [col for col in df.columns if col not in ['case_id', 'class']]
    X = df[feature_cols].values
    
    log.info(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    log.info(f"Number of classes: {n_classes}")
    
    # Outer CV loop
    outer_cv = StratifiedKFold(n_splits=args.n_outer, shuffle=True, random_state=args.random_seed)
    outer_results = []
    
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y_encoded)):
        log.info(f"\n{'='*60}")
        log.info(f"Outer Fold {outer_fold + 1}/{args.n_outer}")
        log.info(f"{'='*60}")
        
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y_encoded[outer_train_idx], y_encoded[outer_test_idx]
        
        # Create Optuna study for this outer fold
        study_name_fold = f"{args.study_name}_fold_{outer_fold}"
        storage = f"sqlite:///{args.sqlite_path}"
        
        sampler = TPESampler(seed=args.random_seed + outer_fold)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(
            study_name=study_name_fold,
            storage=storage,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        log.info(f"Running Optuna optimization with {args.n_trials} trials...")
        
        # Run optimization on outer_train with inner CV
        study.optimize(
            lambda trial: objective_inner_cv(
                trial, X_outer_train, y_outer_train, 
                n_classes, args.n_inner, args.small_steps
            ),
            n_trials=args.n_trials,
            show_progress_bar=False
        )
        
        # Get best hyperparameters
        best_params = study.best_params
        log.info(f"Best hyperparameters: {best_params}")
        log.info(f"Best inner CV score: {study.best_value:.4f}")
        
        # Retrain on full outer_train with FULL_STEPS
        log.info(f"Retraining with best params on full outer train set...")
        
        # Preprocess outer train
        scaler = StandardScaler()
        X_outer_train_scaled = scaler.fit_transform(X_outer_train)
        X_outer_test_scaled = scaler.transform(X_outer_test)
        
        # PCA
        n_qubits = best_params['n_qubits']
        pca = PCA(n_components=min(n_qubits, X_outer_train_scaled.shape[1]), random_state=42)
        X_outer_train_pca = pca.fit_transform(X_outer_train_scaled)
        X_outer_test_pca = pca.transform(X_outer_test_scaled)
        
        # Pad if needed
        if X_outer_train_pca.shape[1] < n_qubits:
            padding = np.zeros((X_outer_train_pca.shape[0], n_qubits - X_outer_train_pca.shape[1]))
            X_outer_train_pca = np.hstack([X_outer_train_pca, padding])
            padding_test = np.zeros((X_outer_test_pca.shape[0], n_qubits - X_outer_test_pca.shape[1]))
            X_outer_test_pca = np.hstack([X_outer_test_pca, padding_test])
        
        # Train final model for this fold
        final_model = MulticlassQuantumClassifierDR(
            n_qubits=best_params['n_qubits'],
            n_layers=best_params['n_layers'],
            n_classes=n_classes,
            learning_rate=best_params['learning_rate'],
            steps=args.full_steps,
            hidden_dim=best_params['hidden_dim'],
            verbose=False
        )
        
        final_model.fit(X_outer_train_pca, y_outer_train)
        
        # Evaluate on outer test set
        y_outer_pred = final_model.predict(X_outer_test_pca)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        outer_accuracy = accuracy_score(y_outer_test, y_outer_pred)
        outer_f1_weighted = f1_score(y_outer_test, y_outer_pred, average='weighted')
        outer_f1_macro = f1_score(y_outer_test, y_outer_pred, average='macro')
        
        log.info(f"Outer fold {outer_fold + 1} results:")
        log.info(f"  Accuracy: {outer_accuracy:.4f}")
        log.info(f"  F1 (weighted): {outer_f1_weighted:.4f}")
        log.info(f"  F1 (macro): {outer_f1_macro:.4f}")
        
        # Save results
        outer_results.append({
            'fold': outer_fold,
            'accuracy': outer_accuracy,
            'f1_weighted': outer_f1_weighted,
            'f1_macro': outer_f1_macro,
            'best_params': best_params,
            'inner_cv_score': study.best_value
        })
        
        # Save fold model
        fold_dir = os.path.join(args.output_dir, f'fold_{outer_fold}')
        os.makedirs(fold_dir, exist_ok=True)
        joblib.dump(final_model, os.path.join(fold_dir, 'model.joblib'))
        joblib.dump(scaler, os.path.join(fold_dir, 'scaler.joblib'))
        joblib.dump(pca, os.path.join(fold_dir, 'pca.joblib'))
        joblib.dump(best_params, os.path.join(fold_dir, 'best_params.joblib'))
    
    # Aggregate results
    log.info(f"\n{'='*60}")
    log.info("Nested CV Results Summary")
    log.info(f"{'='*60}")
    
    results_df = pd.DataFrame(outer_results)
    log.info(f"Mean Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    log.info(f"Mean F1 (weighted): {results_df['f1_weighted'].mean():.4f} ± {results_df['f1_weighted'].std():.4f}")
    log.info(f"Mean F1 (macro): {results_df['f1_macro'].mean():.4f} ± {results_df['f1_macro'].std():.4f}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'nested_cv_summary.csv')
    results_df.to_csv(summary_path, index=False)
    log.info(f"Results saved to {summary_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
