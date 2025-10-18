#!/usr/bin/env python
"""
Optuna nested cross-validation harness for quantum classifiers.
Uses SQLite storage, TPE sampler, and MedianPruner.
Outer folds: 5 (default), Inner folds: 3 (default)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qml_models import MulticlassQuantumClassifierDR
from logging_utils import log

# Constants
SMALL_STEPS = 30  # Budget for inner tuning
FULL_STEPS = 100  # Budget for final model training in outer fold


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


def objective(trial, X_train, y_train, n_classes, n_qubits_range, n_layers_range):
    """
    Optuna objective function for hyperparameter tuning.
    """
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', n_layers_range[0], n_layers_range[1])
    hidden_size = trial.suggest_int('hidden_size', 8, 32, step=8)
    activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    scaler_name = trial.suggest_categorical('scaler', ['standard', 'minmax', 'robust'])
    
    # Scale data
    scaler = get_scaler(scaler_name)
    X_scaled = scaler.fit_transform(X_train)
    
    # Truncate or pad to match n_qubits
    n_qubits = n_qubits_range[0]  # Use fixed n_qubits for now
    if X_scaled.shape[1] > n_qubits:
        X_scaled = X_scaled[:, :n_qubits]
    elif X_scaled.shape[1] < n_qubits:
        pad_width = n_qubits - X_scaled.shape[1]
        X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode='constant')
    
    # 3-fold inner CV
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_scaled, y_train)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Train classifier with small budget
        clf = MulticlassQuantumClassifierDR(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=n_classes,
            learning_rate=learning_rate,
            steps=SMALL_STEPS,
            verbose=False,
            hidden_size=hidden_size,
            activation=activation,
            resume_mode='none'
        )
        
        try:
            clf.fit(X_tr, y_tr)
            y_val_pred = clf.predict(X_val)
            score = f1_score(y_val, y_val_pred, average='weighted')
            scores.append(score)
        except Exception as e:
            log.warning(f"Trial {trial.number}, Fold {fold_idx} failed: {e}")
            scores.append(0.0)
        
        # Report intermediate value for pruning
        trial.report(np.mean(scores), fold_idx)
        
        # Prune if needed
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)


def main():
    parser = argparse.ArgumentParser(description="Optuna nested CV for quantum classifiers")
    
    # Data arguments
    parser.add_argument('--data_file', type=str, required=True, help='Path to input parquet data file')
    parser.add_argument('--output_dir', type=str, default='optuna_results', help='Output directory')
    parser.add_argument('--study_name', type=str, default='quantum_nested_cv', help='Optuna study name')
    parser.add_argument('--db_path', type=str, default='./optuna_studies.db', help='SQLite database path')
    
    # Model arguments
    parser.add_argument('--n_qubits', type=int, default=8, help='Number of qubits (fixed for now)')
    parser.add_argument('--min_layers', type=int, default=2, help='Minimum number of layers')
    parser.add_argument('--max_layers', type=int, default=5, help='Maximum number of layers')
    
    # CV arguments
    parser.add_argument('--outer_folds', type=int, default=5, help='Number of outer CV folds')
    parser.add_argument('--inner_folds', type=int, default=3, help='Number of inner CV folds (fixed at 3 in code)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials per outer fold')
    
    # Other arguments
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log.info("="*80)
    log.info("Optuna Nested Cross-Validation for Quantum Classifiers")
    log.info("="*80)
    log.info(f"Data file: {args.data_file}")
    log.info(f"Study name: {args.study_name}")
    log.info(f"Database: {args.db_path}")
    log.info(f"Outer folds: {args.outer_folds}, Inner folds: {args.inner_folds}")
    log.info(f"Trials per fold: {args.n_trials}")
    log.info(f"Small steps (tuning): {SMALL_STEPS}, Full steps (final): {FULL_STEPS}")
    
    # Load data
    log.info("Loading data...")
    df = pd.read_parquet(args.data_file)
    
    if 'class' not in df.columns:
        log.error("Data must contain 'class' column")
        return 1
    
    if 'case_id' in df.columns:
        df = df.drop(columns=['case_id'])
    
    X = df.drop(columns=['class']).values
    y_raw = df['class'].values
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    
    log.info(f"Data shape: {X.shape}")
    log.info(f"Number of classes: {n_classes}")
    log.info(f"Classes: {list(le.classes_)}")
    
    # Outer CV loop
    outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.random_state)
    outer_results = []
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        log.info("\n" + "="*80)
        log.info(f"OUTER FOLD {outer_fold + 1}/{args.outer_folds}")
        log.info("="*80)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create Optuna study for this outer fold
        study_name = f"{args.study_name}_fold_{outer_fold}"
        storage = f"sqlite:///{args.db_path}"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=TPESampler(seed=args.random_state + outer_fold),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            direction='maximize',
            load_if_exists=True
        )
        
        # Run inner CV optimization
        log.info(f"Running inner CV optimization with {args.n_trials} trials...")
        study.optimize(
            lambda trial: objective(
                trial, X_train, y_train, n_classes,
                (args.n_qubits, args.n_qubits),
                (args.min_layers, args.max_layers)
            ),
            n_trials=args.n_trials,
            show_progress_bar=args.verbose
        )
        
        best_params = study.best_params
        log.info(f"Best parameters: {best_params}")
        log.info(f"Best inner CV score: {study.best_value:.4f}")
        
        # Train final model with best parameters and full budget
        log.info(f"Training final model with {FULL_STEPS} steps...")
        
        scaler = get_scaler(best_params['scaler'])
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Truncate or pad
        if X_train_scaled.shape[1] > args.n_qubits:
            X_train_scaled = X_train_scaled[:, :args.n_qubits]
            X_test_scaled = X_test_scaled[:, :args.n_qubits]
        elif X_train_scaled.shape[1] < args.n_qubits:
            pad_width = args.n_qubits - X_train_scaled.shape[1]
            X_train_scaled = np.pad(X_train_scaled, ((0, 0), (0, pad_width)), mode='constant')
            X_test_scaled = np.pad(X_test_scaled, ((0, 0), (0, pad_width)), mode='constant')
        
        clf = MulticlassQuantumClassifierDR(
            n_qubits=args.n_qubits,
            n_layers=best_params['n_layers'],
            n_classes=n_classes,
            learning_rate=best_params['learning_rate'],
            steps=FULL_STEPS,
            verbose=args.verbose,
            hidden_size=best_params['hidden_size'],
            activation=best_params['activation'],
            resume_mode='none'
        )
        
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate on outer test fold
        y_test_pred = clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        log.info(f"Outer fold {outer_fold + 1} test accuracy: {test_acc:.4f}")
        log.info(f"Outer fold {outer_fold + 1} test F1 (weighted): {test_f1:.4f}")
        
        # Save results
        outer_results.append({
            'outer_fold': outer_fold,
            'test_accuracy': test_acc,
            'test_f1_weighted': test_f1,
            'best_params': best_params,
            'best_inner_score': study.best_value
        })
        
        # Save fold model
        fold_model_path = os.path.join(args.output_dir, f'model_fold_{outer_fold}.joblib')
        joblib.dump(clf, fold_model_path)
        log.info(f"Saved fold model to {fold_model_path}")
    
    # Summary
    log.info("\n" + "="*80)
    log.info("NESTED CV SUMMARY")
    log.info("="*80)
    
    test_accs = [r['test_accuracy'] for r in outer_results]
    test_f1s = [r['test_f1_weighted'] for r in outer_results]
    
    log.info(f"Mean test accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    log.info(f"Mean test F1 (weighted): {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    
    # Save summary
    summary_df = pd.DataFrame(outer_results)
    summary_path = os.path.join(args.output_dir, 'nested_cv_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary saved to {summary_path}")
    
    # Save study visualization (if possible)
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Load last study for visualization
        study_name = f"{args.study_name}_fold_{args.outer_folds - 1}"
        storage = f"sqlite:///{args.db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(args.output_dir, 'optimization_history.png'))
        
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(args.output_dir, 'param_importances.png'))
        
        log.info(f"Study plots saved to {args.output_dir}")
    except Exception as e:
        log.warning(f"Could not save study plots: {e}")
    
    log.info("\nNested CV complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
