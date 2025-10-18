#!/usr/bin/env python
"""
Nested cross-validation with Optuna hyperparameter tuning for quantum classifiers.

Implements:
- Outer StratifiedKFold (n_outer=5) for model evaluation
- Inner StratifiedKFold (n_inner=3) for hyperparameter tuning
- Optuna with TPESampler and MedianPruner
- SQLite storage (./optuna_studies.db)
- Selection by validation weighted-F1 by default
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import log
from qml_models import (
    MulticlassQuantumClassifierDR,
    MulticlassQuantumClassifierDataReuploadingDR
)


# Configuration
SMALL_STEPS = 30  # Reduced steps for inner CV trials
FULL_STEPS = 100  # Full steps for final model training on outer fold
SQLITE_STORAGE = 'sqlite:///optuna_studies.db'


def get_scaler(scaler_name):
    """Returns a scaler object from a string name."""
    s = scaler_name.strip().lower()
    if s in ('m', 'minmax', 'min_max', 'minmaxscaler'):
        return MinMaxScaler()
    if s in ('s', 'standard', 'standardscaler'):
        return StandardScaler()
    if s in ('r', 'robust', 'robustscaler'):
        return RobustScaler()
    return MinMaxScaler()


def create_model(trial, n_classes, n_features, steps=SMALL_STEPS):
    """
    Create a model instance with hyperparameters suggested by Optuna trial.
    
    Args:
        trial: Optuna trial object
        n_classes: Number of classes
        n_features: Number of features
        steps: Number of training steps
    
    Returns:
        Configured model instance
    """
    # Suggest hyperparameters
    n_qubits = trial.suggest_int('n_qubits', n_classes, min(12, n_features), step=2)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    hidden_size = trial.suggest_int('hidden_size', 8, 32, step=4)
    use_classical_readout = trial.suggest_categorical('use_classical_readout', [True, False])
    model_type = trial.suggest_categorical('model_type', ['standard', 'reuploading'])
    scaler_name = trial.suggest_categorical('scaler', ['MinMax', 'Standard', 'Robust'])
    
    # Create model
    model_params = {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'n_classes': n_classes,
        'learning_rate': learning_rate,
        'steps': steps,
        'verbose': False,
        'checkpoint_dir': None
    }
    
    if model_type == 'standard':
        model_params['hidden_size'] = hidden_size
        model_params['use_classical_readout'] = use_classical_readout
        model = MulticlassQuantumClassifierDR(**model_params)
    else:  # reuploading
        # Note: Reuploading model may not have classical readout implemented yet
        # For now, skip those params if not supported
        try:
            model_params['hidden_size'] = hidden_size
            model_params['use_classical_readout'] = use_classical_readout
            model = MulticlassQuantumClassifierDataReuploadingDR(**model_params)
        except TypeError:
            # Fallback if reuploading doesn't support these params yet
            model_params.pop('hidden_size', None)
            model_params.pop('use_classical_readout', None)
            model = MulticlassQuantumClassifierDataReuploadingDR(**model_params)
    
    return model, scaler_name


def inner_cv_objective(trial, X_train, y_train, n_classes, n_inner=3, seed=42):
    """
    Objective function for inner CV hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        X_train: Training features for outer fold
        y_train: Training labels for outer fold
        n_classes: Number of classes
        n_inner: Number of inner CV folds
        seed: Random seed
    
    Returns:
        float: Mean validation weighted-F1 score across inner folds
    """
    try:
        # Create model with suggested hyperparameters
        model, scaler_name = create_model(trial, n_classes, X_train.shape[1], steps=SMALL_STEPS)
        
        # Inner cross-validation
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        scores = []
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
            
            # Preprocess
            imputer = SimpleImputer(strategy='median')
            X_inner_train_imputed = imputer.fit_transform(X_inner_train)
            X_inner_val_imputed = imputer.transform(X_inner_val)
            
            scaler = get_scaler(scaler_name)
            X_inner_train_scaled = scaler.fit_transform(X_inner_train_imputed)
            X_inner_val_scaled = scaler.transform(X_inner_val_imputed)
            
            # Train and evaluate
            model.fit(X_inner_train_scaled, y_inner_train, validation_frac=0.0)
            y_pred = model.predict(X_inner_val_scaled)
            
            # Use weighted F1 score
            score = f1_score(y_inner_val, y_pred, average='weighted')
            scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(np.mean(scores), len(scores))
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)
    
    except Exception as e:
        log.warning(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return poor score for failed trials


def nested_cv(data_path, n_outer=5, n_inner=3, n_trials=20, seed=42, 
              output_dir='./nested_cv_results'):
    """
    Perform nested cross-validation with Optuna hyperparameter tuning.
    
    Args:
        data_path: Path to parquet data file
        n_outer: Number of outer CV folds
        n_inner: Number of inner CV folds
        n_trials: Number of Optuna trials per outer fold
        seed: Random seed
        output_dir: Directory to save results
    """
    # Load data
    log.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    if 'case_id' in df.columns:
        df = df.drop(columns=['case_id'])
    
    X = df.drop(columns=['class']).values
    y_categorical = df['class']
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_categorical)
    n_classes = len(le.classes_)
    
    log.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    log.info(f"Classes: {list(le.classes_)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Outer cross-validation
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    outer_results = []
    
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        log.info(f"\n=== Outer Fold {outer_fold + 1}/{n_outer} ===")
        
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        
        log.info(f"Outer train: {len(X_outer_train)} samples, Outer test: {len(X_outer_test)} samples")
        
        # Create Optuna study for this outer fold
        study_name = f'nested_cv_fold_{outer_fold}'
        study = optuna.create_study(
            storage=SQLITE_STORAGE,
            sampler=TPESampler(seed=seed + outer_fold),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            direction='maximize',
            study_name=study_name,
            load_if_exists=False
        )
        
        # Run inner CV for hyperparameter tuning
        log.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        study.optimize(
            lambda trial: inner_cv_objective(trial, X_outer_train, y_outer_train, n_classes, n_inner, seed),
            n_trials=n_trials,
            show_progress_bar=False
        )
        
        log.info(f"Best trial value (weighted-F1): {study.best_value:.4f}")
        log.info(f"Best hyperparameters: {study.best_params}")
        
        # Save study results
        trials_df = study.trials_dataframe()
        trials_path = os.path.join(output_dir, f'trials_fold_{outer_fold}.csv')
        trials_df.to_csv(trials_path, index=False)
        log.info(f"Trial results saved to {trials_path}")
        
        # Retrain with best hyperparameters on full outer train set with FULL_STEPS
        log.info(f"Retraining with best hyperparameters using {FULL_STEPS} steps...")
        best_params = study.best_params
        
        # Create final model with best hyperparameters
        model_type = best_params['model_type']
        model_params = {
            'n_qubits': best_params['n_qubits'],
            'n_layers': best_params['n_layers'],
            'n_classes': n_classes,
            'learning_rate': best_params['learning_rate'],
            'steps': FULL_STEPS,
            'verbose': False,
            'checkpoint_dir': None
        }
        
        if model_type == 'standard':
            model_params['hidden_size'] = best_params.get('hidden_size', 16)
            model_params['use_classical_readout'] = best_params.get('use_classical_readout', True)
            final_model = MulticlassQuantumClassifierDR(**model_params)
        else:
            try:
                model_params['hidden_size'] = best_params.get('hidden_size', 16)
                model_params['use_classical_readout'] = best_params.get('use_classical_readout', True)
                final_model = MulticlassQuantumClassifierDataReuploadingDR(**model_params)
            except TypeError:
                model_params.pop('hidden_size', None)
                model_params.pop('use_classical_readout', None)
                final_model = MulticlassQuantumClassifierDataReuploadingDR(**model_params)
        
        # Preprocess outer train and test
        imputer = SimpleImputer(strategy='median')
        X_outer_train_imputed = imputer.fit_transform(X_outer_train)
        X_outer_test_imputed = imputer.transform(X_outer_test)
        
        scaler = get_scaler(best_params['scaler'])
        X_outer_train_scaled = scaler.fit_transform(X_outer_train_imputed)
        X_outer_test_scaled = scaler.transform(X_outer_test_imputed)
        
        # Train final model
        final_model.fit(X_outer_train_scaled, y_outer_train, validation_frac=0.0)
        
        # Evaluate on outer test set
        y_pred = final_model.predict(X_outer_test_scaled)
        y_pred_proba = final_model.predict_proba(X_outer_test_scaled)
        
        # Compute metrics
        test_accuracy = accuracy_score(y_outer_test, y_pred)
        test_f1_weighted = f1_score(y_outer_test, y_pred, average='weighted')
        test_f1_macro = f1_score(y_outer_test, y_pred, average='macro')
        
        log.info(f"Outer fold {outer_fold + 1} test accuracy: {test_accuracy:.4f}")
        log.info(f"Outer fold {outer_fold + 1} test weighted-F1: {test_f1_weighted:.4f}")
        log.info(f"Outer fold {outer_fold + 1} test macro-F1: {test_f1_macro:.4f}")
        
        # Save fold results
        fold_result = {
            'fold': outer_fold,
            'test_accuracy': test_accuracy,
            'test_f1_weighted': test_f1_weighted,
            'test_f1_macro': test_f1_macro,
            'best_params': best_params,
            'n_train_samples': len(X_outer_train),
            'n_test_samples': len(X_outer_test)
        }
        outer_results.append(fold_result)
        
        # Save fold model
        model_path = os.path.join(output_dir, f'model_fold_{outer_fold}.joblib')
        joblib.dump({
            'model': final_model,
            'scaler': scaler,
            'imputer': imputer,
            'label_encoder': le,
            'best_params': best_params
        }, model_path)
        log.info(f"Fold model saved to {model_path}")
    
    # Aggregate results
    log.info(f"\n=== Nested CV Results ===")
    results_df = pd.DataFrame(outer_results)
    
    log.info(f"Mean test accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}")
    log.info(f"Mean test weighted-F1: {results_df['test_f1_weighted'].mean():.4f} ± {results_df['test_f1_weighted'].std():.4f}")
    log.info(f"Mean test macro-F1: {results_df['test_f1_macro'].mean():.4f} ± {results_df['test_f1_macro'].std():.4f}")
    
    # Save aggregated results
    results_path = os.path.join(output_dir, 'nested_cv_results.csv')
    results_df.to_csv(results_path, index=False)
    log.info(f"Nested CV results saved to {results_path}")
    
    # Save summary statistics
    summary = {
        'mean_test_accuracy': results_df['test_accuracy'].mean(),
        'std_test_accuracy': results_df['test_accuracy'].std(),
        'mean_test_f1_weighted': results_df['test_f1_weighted'].mean(),
        'std_test_f1_weighted': results_df['test_f1_weighted'].std(),
        'mean_test_f1_macro': results_df['test_f1_macro'].mean(),
        'std_test_f1_macro': results_df['test_f1_macro'].std(),
        'n_outer_folds': n_outer,
        'n_inner_folds': n_inner,
        'n_trials_per_fold': n_trials,
        'small_steps': SMALL_STEPS,
        'full_steps': FULL_STEPS
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Nested cross-validation with Optuna for quantum classifiers"
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to parquet data file')
    parser.add_argument('--n_outer', type=int, default=5,
                       help='Number of outer CV folds (default: 5)')
    parser.add_argument('--n_inner', type=int, default=3,
                       help='Number of inner CV folds (default: 3)')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of Optuna trials per outer fold (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./nested_cv_results',
                       help='Output directory (default: ./nested_cv_results)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run nested CV
    nested_cv(
        data_path=args.data_path,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        n_trials=args.n_trials,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
