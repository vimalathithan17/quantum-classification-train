import argparse
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

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

# Shared local constants
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')

def objective(trial, X_train, y_train, X_val, y_val, n_classes, indicator_cols):
    """Optuna objective function for tuning Classical Meta-Learner."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # We don't drop the indicator columns for classical models, we let the trees/logistic learn them directly
    # So X_train is passed directly
    
    detector = trial.suggest_categorical('meta_model', [
        'lightgbm', 'random_forest', 'logistic_regression', 
        'xgboost', 'catboost', 'mlp', 'svc'
    ])
    
    if detector == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('lgb_lr', 1e-4, 0.5, log=True),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
            'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
            'verbosity': -1,
        }
        model = LGBMClassifier(**params)
    elif detector == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('xgb_lr', 1e-4, 0.5, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
            'verbosity': 0,
        }
        model = XGBClassifier(**params)
    elif detector == 'catboost':
        params = {
            'iterations': trial.suggest_int('cb_iterations', 50, 500),
            'learning_rate': trial.suggest_float('cb_lr', 1e-4, 0.5, log=True),
            'depth': trial.suggest_int('cb_depth', 3, 10),
            'random_seed': RANDOM_STATE,
            'verbose': False,
        }
        model = CatBoostClassifier(**params)
    elif detector == 'mlp':
        layer_choice = trial.suggest_categorical('mlp_layers', ['64', '128', '64_32', '128_64'])
        layer_mapping = {
            '64': (64,),
            '128': (128,),
            '64_32': (64, 32),
            '128_64': (128, 64)
        }
        params = {
            'hidden_layer_sizes': layer_mapping[layer_choice],
            'learning_rate_init': trial.suggest_float('mlp_lr', 1e-4, 1e-1, log=True),
            'alpha': trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
            'random_state': RANDOM_STATE,
            'max_iter': 500,
        }
        model = MLPClassifier(**params)
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
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
            'max_depth': trial.suggest_categorical('rf_max_depth', [None, 5, 10, 20]),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
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
        
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    
    # Compute metrics
    from sklearn.metrics import precision_recall_fscore_support
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_val, predictions, average='weighted', zero_division=0)
    
    # Save trial results
    log.info(f"Trial {trial.number}: meta_model={detector}, f1_weighted={f1_weighted:.4f}")
    return float(f1_weighted)


def main():
    parser = argparse.ArgumentParser(
        description="Train or tune a Classical Meta-Learner (LightGBM/RF/Logistic) for ensemble stacking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('--preds_dir', nargs='+', required=True,
                         help='Directories with base learner predictions (can specify multiple)')
    required.add_argument('--indicator_file', type=str, required=True,
                         help='Parquet file with indicator features and labels')
    
    # Operation mode
    mode_args = parser.add_argument_group('operation mode')
    mode_args.add_argument('--mode', type=str, default='train', choices=['train', 'tune'],
                          help='Mode: train final model or tune hyperparameters (default: train)')
    mode_args.add_argument('--n_trials', type=int, default=50,
                          help='Number of Optuna trials for tuning (default: 50)')

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
            # Initialize wandb globally for this process
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            log.info("Initialized Weights & Biases logging")
        except Exception as e:
            log.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False
    
    if args.mode == 'tune':
        log.info(f"--- Starting Hyperparameter Tuning for Classical Meta-Learner ({args.n_trials} trials) ---")

        study_name = 'classical_metalearner_tuning'
        writable_journal_path = ensure_writable_db('classical_' + TUNING_JOURNAL_FILE)
        log.info(f"Using journal file: {writable_journal_path}")

        storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=writable_journal_path))
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)

        callbacks = []
        if args.use_wandb:
            try:
                from optuna.integration.wandb import WeightsAndBiasesCallback
                # Optuna callback tracks each trial natively in WandB
                wandb_callback = WeightsAndBiasesCallback(
                    metric_name="f1_weighted",
                    wandb_kwargs={"project": args.wandb_project, "name": args.wandb_run_name}
                )
                callbacks.append(wandb_callback)
            except Exception as e:
                log.warning(f"Failed to setup WandB callback: {e}")

        def cv_objective(trial):
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            for train_idx, val_idx in skf.split(X_meta_train, y_meta_train):
                X_train_fold = X_meta_train.iloc[train_idx]
                y_train_fold = y_meta_train.iloc[train_idx]
                X_val_fold = X_meta_train.iloc[val_idx]
                y_val_fold = y_meta_train.iloc[val_idx]
                
                fold_score = objective(trial, X_train_fold, y_train_fold, X_val_fold, y_val_fold, n_classes, indicator_cols)
                scores.append(fold_score)
            
            return np.mean(scores)

        study.optimize(cv_objective, n_trials=args.n_trials, callbacks=callbacks)

        log.info("--- Tuning Complete ---")
        log.info(f"Best hyperparameters found: {study.best_params}")
        
        params_file = os.path.join(OUTPUT_DIR, 'best_classical_metalearner_params.json')
        with open(params_file, 'w') as f:
            json.dump(study.best_params, f, indent=4)
            
    elif args.mode == 'train':
        log.info("--- Training Final Classical Meta-Learner ---")
        params_path = os.path.join(OUTPUT_DIR, 'best_classical_metalearner_params.json')
        try:
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            log.info(f"Loaded best parameters: {best_params}")
        except FileNotFoundError:
            best_params = {'meta_model': 'lightgbm'}
            log.warning("Tuned parameters not found. Using LightGBM defaults.")
            
        model_type = best_params.pop('meta_model', 'lightgbm')
        
        if model_type == 'lightgbm':
            # Map LightGBM optuna params
            model_params = {k.replace('lgb_', ''): v for k,v in best_params.items() if k.startswith('lgb_')}
            if 'lr' in model_params:
                model_params['learning_rate'] = model_params.pop('lr')
            model_params['random_state'] = RANDOM_STATE
            model = LGBMClassifier(**model_params)
        elif model_type == 'xgboost':
            model_params = {k.replace('xgb_', ''): v for k,v in best_params.items() if k.startswith('xgb_')}
            if 'lr' in model_params:
                model_params['learning_rate'] = model_params.pop('lr')
            model_params['random_state'] = RANDOM_STATE
            model = XGBClassifier(**model_params)
        elif model_type == 'catboost':
            model_params = {k.replace('cb_', ''): v for k,v in best_params.items() if k.startswith('cb_')}
            if 'lr' in model_params:
                model_params['learning_rate'] = model_params.pop('lr')
            model_params['random_seed'] = RANDOM_STATE
            model_params['verbose'] = False
            model = CatBoostClassifier(**model_params)
        elif model_type == 'mlp':
            model_params = {k.replace('mlp_', ''): v for k,v in best_params.items() if k.startswith('mlp_')}
            if 'lr' in model_params:
                model_params['learning_rate_init'] = model_params.pop('lr')
            if 'layers' in model_params:
                layer_choice = model_params.pop('layers')
                layer_mapping = {
                    '64': (64,),
                    '128': (128,),
                    '64_32': (64, 32),
                    '128_64': (128, 64)
                }
                if isinstance(layer_choice, list):
                    model_params['hidden_layer_sizes'] = tuple(layer_choice)
                elif isinstance(layer_choice, tuple):
                    model_params['hidden_layer_sizes'] = layer_choice
                elif layer_choice in layer_mapping:
                    model_params['hidden_layer_sizes'] = layer_mapping[layer_choice]
                else:
                    # Default fallback
                    model_params['hidden_layer_sizes'] = (64, 32)
            model_params['random_state'] = RANDOM_STATE
            model_params['max_iter'] = 500
            model = MLPClassifier(**model_params)
        elif model_type == 'svc':
            model_params = {k.replace('svc_', ''): v for k,v in best_params.items() if k.startswith('svc_')}
            model_params['random_state'] = RANDOM_STATE
            model_params['probability'] = True
            model = SVC(**model_params)
        elif model_type == 'random_forest':
            model_params = {k.replace('rf_', ''): v for k,v in best_params.items() if k.startswith('rf_')}
            model_params['random_state'] = RANDOM_STATE
            model = RandomForestClassifier(**model_params)
        else:
            model_params = {k.replace('lr_', ''): v for k,v in best_params.items() if k.startswith('lr_')}
            model_params['random_state'] = RANDOM_STATE
            model = LogisticRegression(**model_params)
            
        model.fit(X_meta_train, y_meta_train)
        
        import joblib
        model_path = os.path.join(OUTPUT_DIR, 'classical_meta_learner_final.joblib')
        joblib.dump(model, model_path)
        log.info(f"Final classical meta-learner saved to {model_path}!")

        # Perform evaluation on inference test set if provided
        if not X_meta_test.empty and not y_meta_test.empty:
            test_preds = model.predict(X_meta_test)
            acc = accuracy_score(y_meta_test, test_preds)
            
            # Save test confusion matrix diagram
            log.info("--- Generating Test Set Confusion Matrix Diagram ---")
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            test_cm = confusion_matrix(y_meta_test, test_preds, labels=list(range(n_classes)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title('Test Set Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            test_cm_path = os.path.join(OUTPUT_DIR, 'test_confusion_matrix.png')
            plt.savefig(test_cm_path)
            plt.close()
            log.info(f"Saved test confusion matrix diagram to {test_cm_path}")

            from sklearn.metrics import precision_recall_fscore_support
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_meta_test, test_preds, average='macro', zero_division=0)
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_meta_test, test_preds, average='weighted', zero_division=0)
            
            log.info(f"Evaluation on supplied target tests -> Meta-Test accuracy: {acc:.4f}")
            log.info(f"Classification Report:\n{classification_report(y_meta_test, test_preds)}")
            
            if args.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "test/accuracy": float(acc),
                        "test/precision_macro": float(precision_macro),
                        "test/recall_macro": float(recall_macro),
                        "test/f1_macro": float(f1_macro),
                        "test/precision_weighted": float(precision_weighted),
                        "test/recall_weighted": float(recall_weighted),
                        "test/f1_weighted": float(f1_weighted)
                    })
                    log.info("Logged evaluation metrics to WandB.")
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
