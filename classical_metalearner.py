import argparse
import copy
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import centralized logger and assembly utilities
from logging_utils import log
from metalearner import (
    ENCODER_DIR,
    RANDOM_STATE,
    TUNING_JOURNAL_FILE,
    _per_class_specificity,
    assemble_meta_data,
    ensure_writable_db,
    ensure_writable_results_dir,
    set_seed,
)
# Added the missing import here
from utils.metrics_utils import compute_metrics

# Shared local constants
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')

# Global dictionary to track the best model during tuning
best_trial_data = {
    'score': -np.inf,
    'model': None
}

def apply_transformer_weighting(X_train, X_val, model_type, transformer_weight):
    """Dynamically applies Scaling, Duplication, or Weights array based on algorithm."""
    transformer_cols = [col for col in X_train.columns if 'Transformer' in str(col)]
    if not transformer_cols or transformer_weight <= 1.0:
        return X_train, X_val, None

    X_train_out = X_train.copy()
    X_val_out = X_val.copy() if X_val is not None and not X_val.empty else X_val
    feature_weights = None

    if model_type in ['logistic_regression', 'svc']:
        # Approach 3: Feature Scaling (For gradient/weight-based models)
        for col in transformer_cols:
            X_train_out[col] = X_train_out[col] * transformer_weight
            if X_val_out is not None and not X_val_out.empty:
                X_val_out[col] = X_val_out[col] * transformer_weight

    elif model_type in ['random_forest', 'catboost']:
        # Approach 2: Feature Duplication with UNIQUE column names to prevent CatBoost crash
        num_duplications = int(transformer_weight) - 1
        if num_duplications > 0:
            train_dups = []
            val_dups = []
            
            for i in range(num_duplications):
                # Copy and rename train columns
                train_dup = X_train_out[transformer_cols].copy()
                train_dup.columns = [f"{col}_dup_{i}" for col in transformer_cols]
                train_dups.append(train_dup)
                
                # Copy and rename val columns
                if X_val_out is not None and not X_val_out.empty:
                    val_dup = X_val_out[transformer_cols].copy()
                    val_dup.columns = [f"{col}_dup_{i}" for col in transformer_cols]
                    val_dups.append(val_dup)
                    
            X_train_out = pd.concat([X_train_out] + train_dups, axis=1)
            if X_val_out is not None and not X_val_out.empty:
                X_val_out = pd.concat([X_val_out] + val_dups, axis=1)

    elif model_type in ['xgboost', 'lightgbm']:
        # Approach 1: Explicit Feature Weighting
        feature_weights = np.ones(X_train_out.shape[1])
        for i, col in enumerate(X_train_out.columns):
            if 'Transformer' in str(col):
                feature_weights[i] = transformer_weight

    return X_train_out, X_val_out, feature_weights

def fit_with_weights(model, X, y, model_type, feature_weights):
    """Handles edge cases for scikit-learn wrappers passing explicit feature weights."""
    if feature_weights is not None:
        if model_type == 'xgboost':
            model.fit(X, y, feature_weights=feature_weights)
        elif model_type == 'lightgbm':
            try:
                # LightGBM uses 'feature_weight' via kwargs
                model.fit(X, y, feature_weight=feature_weights)
            except TypeError:
                log.warning("LightGBM version does not support explicit feature_weight kwarg. Running without it.")
                model.fit(X, y)
    else:
        model.fit(X, y)

def objective(trial, X_train, y_train, X_val, y_val, n_classes, indicator_cols, transformer_weight):
    """Optuna objective function for tuning Classical Meta-Learner."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # We strictly enforce using Logistic Regression and XGBoost for this dataset size
    detector = trial.suggest_categorical('meta_model', [
        'logistic_regression', 'xgboost', 'lightgbm', 'random_forest', 'catboost', 'svc'
    ])
    
    # 1. Dynamically route and apply Feature Importance Logic
    X_train_mod, X_val_mod, f_weights = apply_transformer_weighting(X_train, X_val, detector, transformer_weight)

    if detector == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('lgb_lr', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('lgb_max_depth', 2, 5),
            'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
            'verbosity': -1,
        }
        model = LGBMClassifier(**params)
    elif detector == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('xgb_lr', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 2, 5),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
            'verbosity': 0,
        }
        model = XGBClassifier(**params)
    elif detector == 'catboost':
        params = {
            'iterations': trial.suggest_int('cb_iterations', 50, 200),
            'learning_rate': trial.suggest_float('cb_lr', 1e-3, 0.1, log=True),
            'depth': trial.suggest_int('cb_depth', 2, 5),
            'random_seed': RANDOM_STATE,
            'verbose': False,
        }
        model = CatBoostClassifier(**params)
    elif detector == 'svc':
        params = {
            'C': trial.suggest_float('svc_C', 1e-2, 100.0, log=True),
            'gamma': trial.suggest_categorical('svc_gamma', ['scale', 'auto']),
            'random_state': RANDOM_STATE,
            'probability': True,
        }
        model = SVC(**params)
    elif detector == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
            'max_depth': trial.suggest_categorical('rf_max_depth', [3, 5, 10]),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
        model = RandomForestClassifier(**params)
    else:
        params = {
            'C': trial.suggest_float('lr_C', 1e-4, 10.0, log=True),
            'max_iter': 1000,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
        model = LogisticRegression(**params)
        
    # 2. Fit and Predict using adapted arrays/weights
    fit_with_weights(model, X_train_mod, y_train, detector, f_weights)
    predictions = model.predict(X_val_mod)
    
    # Compute metrics
    m = compute_metrics(y_val, predictions, n_classes)
    
    # Exclusively tracking f1_weighted to ensure maximum performance
    combined_metric = m['f1_weighted']
    
    # Save the absolute best model globally
    global best_trial_data
    if combined_metric > best_trial_data['score']:
        best_trial_data['score'] = combined_metric
        best_trial_data['model'] = copy.deepcopy(model)
    
    # Save trial results
    log.info(f"Trial {trial.number} Results:")
    log.info(f"  -> Model:                {detector}")
    log.info(f"  -> Combined Score (F1):  {combined_metric:.4f}")
    log.info(f"  -> Accuracy:             {m['accuracy']:.4f}")
    log.info(f"  -> Weighted P:           {m['precision_weighted']:.4f}")
    log.info(f"  -> Weighted R:           {m['recall_weighted']:.4f}")
    log.info(f"  -> Weighted S:           {m['specificity_weighted']:.4f}")
    
    for k, v in m.items():
        if isinstance(v, (int, float)):
            trial.set_user_attr(k, float(v))
            
    return float(combined_metric)


def main():
    parser = argparse.ArgumentParser(
        description="Direct Train & Tune Classical Meta-Learner for ensemble stacking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('--preds_dir', nargs='+', required=True,
                          help='Directories with base learner predictions (can specify multiple)')
    required.add_argument('--indicator_file', type=str, required=True,
                          help='Parquet file with indicator features and labels')
    
    # Tuning Configuration
    tune_args = parser.add_argument_group('tuning configuration')
    tune_args.add_argument('--n_trials', type=int, default=50,
                           help='Number of Optuna trials for tuning (default: 50)')
    tune_args.add_argument('--transformer_weight', type=float, default=5.0,
                           help='Weight multiplier to prioritize Transformer base model predictions (default: 5.0)')

    # Logging config
    log_args = parser.add_argument_group('logging')
    log_args.add_argument('--use_wandb', action='store_true',
                          help='Enable W&B experiment tracking')
    log_args.add_argument('--wandb_project', type=str, default=None,
                          help='W&B project name')
    log_args.add_argument('--wandb_run_name', type=str, default=None,
                          help='W&B run name')
                          
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le, indicator_cols = assemble_meta_data(args.preds_dir, args.indicator_file)
    
    if X_meta_train is None:
        log.critical("Failed to assemble meta-dataset. Exiting.")
        return

    n_classes = len(le.classes_)
    log.info(f"Meta-learner will be trained on {n_classes} classes.")

    global OUTPUT_DIR
    OUTPUT_DIR = ensure_writable_results_dir(OUTPUT_DIR)

    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            log.info("Initialized Weights & Biases logging")
        except Exception as e:
            log.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False
    
    log.info(f"--- Starting Direct Training & Tuning for Classical Meta-Learner ({args.n_trials} trials) ---")

    study_name = 'classical_metalearner_tuning'
    writable_journal_path = ensure_writable_db('classical_' + TUNING_JOURNAL_FILE)
    
    storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=writable_journal_path))
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)

    callbacks = []
    if args.use_wandb:
        try:
            from optuna.integration.wandb import WeightsAndBiasesCallback
            wandb_callback = WeightsAndBiasesCallback(
                metric_name="combined_metric",
                wandb_kwargs={"project": args.wandb_project, "name": args.wandb_run_name}
            )
            callbacks.append(wandb_callback)
        except Exception as e:
            pass

    # Create a single Train/Validation split to be used for all trials (80/20)
    log.info("Creating internal validation split (20%) for trial training and evaluation.")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_meta_train, y_meta_train, test_size=0.20, stratify=y_meta_train, random_state=RANDOM_STATE
    )

    def single_run_objective(trial):
        return objective(
            trial, X_train_split, y_train_split, X_val_split, y_val_split, 
            n_classes, indicator_cols, args.transformer_weight
        )

    study.optimize(single_run_objective, n_trials=args.n_trials, callbacks=callbacks)

    log.info("--- Tuning Complete ---")
    log.info(f"Best hyperparameters found: {study.best_params}")
    
    params_file = os.path.join(OUTPUT_DIR, 'best_classical_metalearner_params.json')
    with open(params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    # --- Final Test Set Evaluation using the absolutely best model ---
    global best_trial_data
    best_model = best_trial_data['model']
    
    if best_model is not None:
        import joblib
        model_path = os.path.join(OUTPUT_DIR, 'classical_meta_learner_final.joblib')
        joblib.dump(best_model, model_path)
        log.info(f"\nFinal best classical meta-learner saved to {model_path}!")

        if not X_meta_test.empty and not y_meta_test.empty:
            
            # Must apply the exact same transformation to the test set!
            _, X_test_mod, _ = apply_transformer_weighting(
                X_meta_train, X_meta_test, study.best_params['meta_model'], args.transformer_weight
            )
            
            test_preds = best_model.predict(X_test_mod)
            test_metrics = compute_metrics(y_meta_test, test_preds, n_classes)
            
            # Save test confusion matrix diagram
            log.info("--- Generating Test Set Confusion Matrix Diagram ---")
            test_cm = confusion_matrix(y_meta_test, test_preds, labels=list(range(n_classes)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f"Test Set Confusion Matrix ({study.best_params['meta_model']})")
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            test_cm_path = os.path.join(OUTPUT_DIR, 'test_confusion_matrix.png')
            plt.savefig(test_cm_path)
            plt.close()
            log.info(f"Saved test confusion matrix diagram to {test_cm_path}")

            log.info("--- Final Test Set Evaluation ---")
            log.info(f"  -> Accuracy:       {test_metrics['accuracy']:.4f}")
            log.info(f"  -> Weighted P:     {test_metrics['precision_weighted']:.4f}")
            log.info(f"  -> Weighted R:     {test_metrics['recall_weighted']:.4f}")
            log.info(f"  -> Weighted S:     {test_metrics['specificity_weighted']:.4f}")
            log.info(f"  -> Weighted F1:    {test_metrics['f1_weighted']:.4f}")
            log.info(f"\nClassification Report:\n{classification_report(y_meta_test, test_preds)}")
            
            if args.use_wandb:
                try:
                    import wandb
                    wandb_metrics = {f"test/{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
                    wandb.log(wandb_metrics)
                    log.info("Logged all test metrics to WandB.")
                except Exception as e:
                    log.warning(f"Failed to log test metrics to WandB: {e}")

    # Finish wandb run
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

if __name__ == '__main__':
    main()
