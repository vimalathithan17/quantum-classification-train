# Final Universal Tuning Script (`tune_models.py`)
import pandas as pd
import numpy as np
import random
import optuna
import argparse
import os
import json
import shutil
import signal
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from utils.masked_transformers import MaskedTransformer
from lightgbm import LGBMClassifier
from umap import UMAP
import wandb

# Additional imports for comprehensive metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from utils.metrics_utils import compute_metrics
import numpy as _np  # local alias to avoid shadowing pennylane.numpy

# Import the centralized logger
from logging_utils import log

# Ensure your corrected, multiclass 'qml_models.py' is in the same directory
from qml_models import (
    MulticlassQuantumClassifierDR, 
    MulticlassQuantumClassifierDataReuploadingDR,
    ConditionalMulticlassQuantumClassifierFS,
    ConditionalMulticlassQuantumClassifierDataReuploadingFS
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not required for QML scripts
    log.info(f"Random seed set to {seed}")


# Directories configurable via environment
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
TUNING_RESULTS_DIR = os.environ.get('TUNING_RESULTS_DIR', 'tuning_results')
OPTUNA_DB_PATH = os.environ.get('OPTUNA_DB_PATH', './optuna_studies.db')
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))

# Initialize random seeds for reproducibility
set_seed(RANDOM_STATE)

# Global flag for interruption handling
interrupted = False


def handle_interruption(signum, frame):
    """Handle interruption signals (SIGINT, SIGTERM) gracefully."""
    global interrupted
    if not interrupted:
        interrupted = True
        log.warning(f"\nReceived interruption signal ({signum}). Finishing current trial and saving progress...")
        log.warning("Press Ctrl+C again to force exit (may lose current trial).")
    else:
        log.error("\nForced interruption. Exiting immediately.")
        sys.exit(1)


def is_db_writable(db_path):
    """Check if a database file is writable."""
    if not os.path.exists(db_path):
        # If DB doesn't exist, check if parent directory is writable
        parent_dir = os.path.dirname(db_path) or '.'
        return os.access(parent_dir, os.W_OK)
    return os.access(db_path, os.W_OK)


def ensure_writable_db(db_path):
    """
    Ensure the database is writable. If read-only, copy it to a writable location.
    Returns the path to the writable database.
    """
    if is_db_writable(db_path):
        log.info(f"Database at {db_path} is writable.")
        return db_path
    
    log.warning(f"Database at {db_path} is read-only. Copying to a writable location...")
    
    # Try multiple locations for writable copy
    import tempfile
    candidate_paths = [
        os.path.join(os.getcwd(), 'optuna_studies_working.db'),
        os.path.join(tempfile.gettempdir(), 'optuna_studies_working.db')
    ]
    
    writable_path = None
    for candidate in candidate_paths:
        # Check if we can write to this location
        if os.path.exists(candidate):
            if is_db_writable(candidate):
                writable_path = candidate
                break
        else:
            # Check if parent directory is writable
            parent_dir = os.path.dirname(candidate) or '.'
            if os.access(parent_dir, os.W_OK):
                writable_path = candidate
                break
    
    if writable_path is None:
        raise RuntimeError("Could not find a writable location for the database copy")
    
    # Copy the database if source exists
    if os.path.exists(db_path):
        try:
            shutil.copy2(db_path, writable_path)
            # Ensure the copy is writable (shutil.copy2 preserves permissions)
            os.chmod(writable_path, 0o644)
            log.info(f"Copied database to {writable_path}")
        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to copy database to {writable_path}: {e}")
    else:
        log.info(f"Source database does not exist yet. Will create new database at {writable_path}")
    
    return writable_path


def ensure_writable_results_dir(results_dir):
    """
    Ensure the tuning results directory is writable. If not, create a copy in current working dir.
    Returns the path to the writable directory.
    """
    # Try to create directory if it doesn't exist
    try:
        os.makedirs(results_dir, exist_ok=True)
    except (OSError, PermissionError):
        pass
    
    # Check if writable
    if os.path.exists(results_dir) and os.access(results_dir, os.W_OK):
        log.info(f"Results directory '{results_dir}' is writable.")
        return results_dir
    
    # Not writable - try fallback in current directory
    log.warning(f"Results directory '{results_dir}' is not writable. Creating fallback directory...")
    
    fallback_dir = os.path.join(os.getcwd(), os.path.basename(results_dir.rstrip('/')))
    
    try:
        os.makedirs(fallback_dir, exist_ok=True)
        
        if os.access(fallback_dir, os.W_OK):
            log.info(f"Using fallback results directory: '{fallback_dir}'")
            
            # Copy existing files from original to fallback if possible
            if os.path.exists(results_dir):
                for filename in os.listdir(results_dir):
                    src = os.path.join(results_dir, filename)
                    dst = os.path.join(fallback_dir, filename)
                    try:
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
                            log.info(f"Copied: {filename}")
                    except Exception as e:
                        log.warning(f"Could not copy {filename}: {e}")
            
            return fallback_dir
    except Exception as e:
        log.error(f"Could not create fallback directory '{fallback_dir}': {e}")
    
    # Last resort - return original and hope for the best
    log.warning(f"No writable results directory available. Results may not be saved.")
    return results_dir


def safe_load_parquet(file_path):
    """Loads a parquet file with increased thrift limits, returning None on failure."""
    if not os.path.exists(file_path):
        log.error(f"File not found at {file_path}")
        return None
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


def _log_fold_metrics_to_wandb(trial_wandb_run, fold, acc, f1_weighted, f1_macro, 
                                 prec_weighted, prec_macro, rec_weighted, rec_macro, 
                                 spec_macro, spec_weighted):
    """
    Log fold-level metrics to wandb trial run.
    
    Args:
        trial_wandb_run: Active wandb run object
        fold: Fold index (0-based)
        acc: Accuracy score
        f1_weighted: Weighted F1 score
        f1_macro: Macro F1 score
        prec_weighted: Weighted precision
        prec_macro: Macro precision
        rec_weighted: Weighted recall
        rec_macro: Macro recall
        spec_macro: Macro specificity
        spec_weighted: Weighted specificity
    """
    if trial_wandb_run:
        try:
            wandb.log({
                f'fold_{fold+1}/accuracy': float(acc),
                f'fold_{fold+1}/f1_weighted': float(f1_weighted),
                f'fold_{fold+1}/f1_macro': float(f1_macro),
                f'fold_{fold+1}/precision_weighted': float(prec_weighted),
                f'fold_{fold+1}/precision_macro': float(prec_macro),
                f'fold_{fold+1}/recall_weighted': float(rec_weighted),
                f'fold_{fold+1}/recall_macro': float(rec_macro),
                f'fold_{fold+1}/specificity_macro': float(spec_macro),
                f'fold_{fold+1}/specificity_weighted': float(spec_weighted),
            })
        except Exception as e:
            log.warning(f"Failed to log fold metrics to wandb: {e}")


def cleanup_old_trials(results_dir, study, keep_best=True, keep_latest_n=2):
    """
    Clean up old trial directories, keeping only the best trial and the latest N trials.
    
    Args:
        results_dir: Directory containing trial_* subdirectories
        study: Optuna study object
        keep_best: Whether to keep the best trial directory
        keep_latest_n: Number of latest trials to keep (in addition to best)
    """
    try:
        # Get list of trial directories
        trial_dirs = []
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and item.startswith('trial_'):
                try:
                    trial_num = int(item.split('_')[1])
                    trial_dirs.append((trial_num, item_path))
                except (ValueError, IndexError):
                    continue
        
        if not trial_dirs:
            return
        
        # Sort by trial number
        trial_dirs.sort(key=lambda x: x[0])
        
        # Identify trials to keep
        trials_to_keep = set()
        
        # Keep best trial
        if keep_best and len(study.trials) > 0:
            try:
                best_trial_num = study.best_trial.number
                trials_to_keep.add(best_trial_num)
                log.info(f"Keeping best trial directory: trial_{best_trial_num}")
            except ValueError:
                log.warning("No best trial found (no completed trials)")
        
        # Keep latest N trials
        latest_trials = [t[0] for t in trial_dirs[-keep_latest_n:]]
        trials_to_keep.update(latest_trials)
        log.info(f"Keeping latest {keep_latest_n} trial directories: {latest_trials}")
        
        # Remove other trial directories
        for trial_num, trial_path in trial_dirs:
            if trial_num not in trials_to_keep:
                try:
                    shutil.rmtree(trial_path)
                    log.info(f"Removed old trial directory: {trial_path}")
                except Exception as e:
                    log.warning(f"Could not remove trial directory {trial_path}: {e}")
    
    except Exception as e:
        log.warning(f"Error during trial directory cleanup: {e}")

def get_scaler(scaler_name):
    """Returns a scaler object from a string name."""
    if not scaler_name:
        return MinMaxScaler()
    s = scaler_name.strip().lower()
    if s in ('m', 'minmax', 'min_max', 'minmaxscaler'):
        return MinMaxScaler()
    if s in ('s', 'standard', 'standardscaler'):
        return StandardScaler()
    if s in ('r', 'robust', 'robustscaler'):
        return RobustScaler()
    return MinMaxScaler()

def objective(trial, args, X, y, n_classes, min_qbits, max_qbits, scaler_options, use_pretrained=False):
    """Defines one trial with Stratified K-Fold for a given pipeline configuration."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Log suggested parameters
    # Only suggest scalers for Approach 1 (DRE). Conditional models (Approach 2)
    # learn from missingness and should not be tuned with sklearn scalers.
    params = {
        'n_qubits': trial.suggest_int('n_qubits', min_qbits, max_qbits, step=2),
        'n_layers': trial.suggest_int('n_layers', args.min_layers, args.max_layers)
    }
    if args.approach == 1:
        params['scaler'] = trial.suggest_categorical('scaler', scaler_options)

    log.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=2)}")

    # Initialize trial-level wandb run if requested
    trial_wandb_run = None
    if args.use_wandb:
        try:
            # Create a descriptive run name for the trial
            approach_name = "DRE" if args.approach == 1 else "CF"
            trial_run_name = f"trial_{trial.number}_{approach_name}_{args.datatype}_{args.qml_model}"
            if args.approach == 1 and 'scaler' in params:
                trial_run_name += f"_{params['scaler']}"
            trial_run_name += f"_q{params['n_qubits']}_l{params['n_layers']}"
            
            # Initialize wandb for this trial
            trial_wandb_run = wandb.init(
                project=args.wandb_project,
                name=trial_run_name,
                config={
                    'trial_number': trial.number,
                    'approach': args.approach,
                    'datatype': args.datatype,
                    'qml_model': args.qml_model,
                    'dim_reducer': args.dim_reducer if args.approach == 1 else None,
                    'use_pretrained': use_pretrained,
                    'n_qubits': params['n_qubits'],
                    'n_layers': params['n_layers'],
                    'scaler': params.get('scaler', None),
                    'steps': args.steps,
                    'min_qbits': min_qbits,
                    'max_qbits': max_qbits,
                    'min_layers': args.min_layers,
                    'max_layers': args.max_layers,
                },
                reinit=True,
                group=f"tuning_{args.datatype}_app{args.approach}_{args.qml_model}"
            )
            log.info(f"Initialized trial-level W&B run: {trial_run_name}")
        except Exception as e:
            log.warning(f"Failed to initialize trial-level wandb: {e}")
            trial_wandb_run = None

    # Build scaler only for Approach 1; for Approach 2 keep scaler as None
    if args.approach == 1:
        scaler = get_scaler(params['scaler'])
        # Note: scaler will be wrapped with MaskedTransformer when added to steps_list
    else:
        scaler = None
    steps = args.steps  # Use steps from command-line arguments
    n_qubits = params['n_qubits']
    n_layers = params['n_layers']
    
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    if args.approach == 1:
        steps_list = []
        # Common steps for Approach 1
        # Skip imputer for pretrained features (already clean)
        if not use_pretrained:
            steps_list.append(('imputer', MaskedTransformer(SimpleImputer(strategy='median'), fallback='raise')))
        steps_list.append(('scaler', MaskedTransformer(scaler, fallback='raise')))

        # Dimensionality reduction: always apply if embed_dim > n_qubits
        # For pretrained features, this reduces from embed_dim to n_qubits
        if args.dim_reducer == 'pca':
            steps_list.append(('dim_reducer', MaskedTransformer(PCA(n_components=n_qubits), fallback='raise')))
        else:
            steps_list.append(('dim_reducer', MaskedTransformer(UMAP(n_components=n_qubits, random_state=RANDOM_STATE), fallback='raise')))

        # Generate wandb run name for Approach 1 (DRE) per user's requested pattern
        # Desired format: tune_DRE_<datatype>_<qml_model>_q{qbits}_l{layers}_{scaler}
        if args.use_wandb:
            base_name = f"tune_DRE_{args.datatype}_{args.qml_model}"
            base_name += f"_q{n_qubits}_l{n_layers}"
            scaler_name = params.get('scaler', None)
            if scaler_name:
                base_name += f"_{scaler_name}"
            wandb_run_name = base_name
        else:
            wandb_run_name = None
        # If user provided a custom wandb run name, respect it (do not append trial number)
        if args.wandb_run_name:
            wandb_run_name = args.wandb_run_name
        
        # Perform cross-validation with a FRESH model per fold (orthodox CV)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Starting training...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Generate wandb run name for this fold
            if args.use_wandb:
                base_name = f"tune_DRE_{args.datatype}_{args.qml_model}_q{n_qubits}_l{n_layers}"
                scaler_name = params.get('scaler', None)
                if scaler_name:
                    base_name += f"_{scaler_name}"
                wandb_run_name_fold = f"{base_name}_f{fold+1}"
            else:
                wandb_run_name_fold = None
            if args.wandb_run_name:
                wandb_run_name_fold = args.wandb_run_name

            # Instantiate fresh model for this fold
            if args.qml_model == 'standard':
                qml_model = MulticlassQuantumClassifierDR(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=wandb_run_name_fold)
            else:  # reuploading
                qml_model = MulticlassQuantumClassifierDataReuploadingDR(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=wandb_run_name_fold)

            pipeline = Pipeline(steps_list + [('qml', qml_model)])

            pipeline.fit(X_train, y_train)
            
            # predictions
            y_val_pred = pipeline.predict(X_val)
            y_val_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_val_proba = pipeline.predict_proba(X_val)
                except Exception as e:
                    log.warning(f"pipeline.predict_proba failed: {e}")
                    y_val_proba = None

            # compute metrics using centralized compute_metrics
            # Use n_classes from function parameter (full dataset) not from validation fold
            metrics = compute_metrics(y_val, y_val_pred, n_classes)
            acc = float(metrics['accuracy'])
            f1_macro = float(metrics['f1_macro'])
            f1_weighted = float(metrics['f1_weighted'])
            prec_macro = float(metrics['precision_macro'])
            prec_weighted = float(metrics['precision_weighted'])
            rec_macro = float(metrics['recall_macro'])
            rec_weighted = float(metrics['recall_weighted'])
            spec_macro = float(metrics['specificity_macro'])
            spec_weighted = float(metrics['specificity_weighted'])
            cm = metrics['confusion_matrix']

            # pack fold metrics
            fold_metrics = {
                'accuracy': acc,
                'precision_macro': prec_macro,
                'recall_macro': rec_macro,
                'f1_macro': f1_macro,
                'precision_weighted': prec_weighted,
                'recall_weighted': rec_weighted,
                'f1_weighted': f1_weighted,
                'specificity_macro': spec_macro,
                'specificity_weighted': spec_weighted,
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_val, y_val_pred, zero_division=0)
            }

            # logging
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: metrics: f1_weighted={f1_weighted:.4f}, acc={acc:.4f}")
            
            # Log fold metrics to trial-level wandb run
            _log_fold_metrics_to_wandb(trial_wandb_run, fold, acc, f1_weighted, f1_macro,
                                       prec_weighted, prec_macro, rec_weighted, rec_macro,
                                       spec_macro, spec_weighted)
            
            # Save per-fold metrics to disk
            fold_dir = os.path.join(TUNING_RESULTS_DIR, f"trial_{trial.number}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, f"fold_{fold+1}_metrics.json"), 'w') as fh:
                json.dump(fold_metrics, fh, indent=2)

            # attach fold metrics to the Optuna trial so you can inspect them later
            trial.set_user_attr(f"fold_{fold+1}_metrics", fold_metrics)

            # report to optuna for pruning
            trial.report(float(f1_weighted), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # accumulate objective metric per fold
            scores.append(float(f1_weighted))

    elif args.approach == 2:
         
        # For Approach 2: perform fold-specific selection and instantiate a fresh QML model per fold.
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Starting training...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            

            # Perform fold-specific feature selection using raw features (with NaNs)
            # so LightGBM can make use of native missing-value handling. Do NOT
            # impute or scale prior to selection to preserve missingness signal.
            log.info("    - Performing LightGBM selection on raw data (no impute/scale) for Approach 2...")
            lgb = LGBMClassifier(n_estimators=50, learning_rate=0.1, feature_fraction=0.7,
                                 n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)
            X_train_for_selection = X_train
            actual_k = min(n_qubits, X_train_for_selection.shape[1])
            lgb.fit(X_train_for_selection.values, y_train)
            importances = lgb.feature_importances_
            top_idx = np.argsort(importances)[-actual_k:][::-1]
            selected_cols = X_train.columns[top_idx]
            log.info(f"    - Selected features (fold {fold+1}): {list(selected_cols)}")

            X_train_selected = X_train[selected_cols]
            X_val_selected = X_val[selected_cols]

            # Prepare the data tuple (mask, fill). Conditional models learn from
            # missingness masks so we do NOT scale or impute during training.
            is_missing_train = X_train_selected.isnull().astype(int).values
            X_train_filled = X_train_selected.fillna(0.0).values
            is_missing_val = X_val_selected.isnull().astype(int).values
            X_val_filled = X_val_selected.fillna(0.0).values

            # No scaling for conditional models; pass filled arrays directly.
            X_train_scaled = X_train_filled
            X_val_scaled = X_val_filled

            # Generate wandb run name for this fold and instantiate a fresh QML model
            if args.use_wandb:
                base_name = f"tune_CF_{args.datatype}_{args.qml_model}_q{n_qubits}_l{n_layers}"
                wandb_run_name_fold = f"{base_name}_f{fold+1}"
            else:
                wandb_run_name_fold = None
            if args.wandb_run_name:
                wandb_run_name_fold = args.wandb_run_name

            if args.qml_model == 'standard':
                qml_model = ConditionalMulticlassQuantumClassifierFS(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=wandb_run_name_fold)
            else: # reuploading
                qml_model = ConditionalMulticlassQuantumClassifierDataReuploadingFS(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=wandb_run_name_fold)

            qml_model.fit((X_train_scaled, is_missing_train), y_train.values)
            
            # obtain predictions from your QML model API
            try:
                y_val_pred = qml_model.predict((X_val_scaled, is_missing_val))
            except Exception:
                # if predict expects flat arrays or different signature adapt here
                y_val_pred = qml_model.predict(np.asarray(X_val_scaled))

            # optionally probabilities
            y_val_proba = None
            if hasattr(qml_model, "predict_proba"):
                try:
                    y_val_proba = qml_model.predict_proba((X_val_scaled, is_missing_val))
                except Exception:
                    try:
                        y_val_proba = qml_model.predict_proba(np.asarray(X_val_scaled))
                    except Exception as e:
                        log.warning(f"predict_proba not available or failed for fold {fold+1}: {e}")
                        y_val_proba = None

            # compute metrics using centralized compute_metrics
            # Use n_classes from function parameter (full dataset) not from validation fold
            metrics = compute_metrics(y_val.values, y_val_pred, n_classes)
            acc = float(metrics['accuracy'])
            f1_macro = float(metrics['f1_macro'])
            f1_weighted = float(metrics['f1_weighted'])
            prec_macro = float(metrics['precision_macro'])
            prec_weighted = float(metrics['precision_weighted'])
            rec_macro = float(metrics['recall_macro'])
            rec_weighted = float(metrics['recall_weighted'])
            spec_macro = float(metrics['specificity_macro'])
            spec_weighted = float(metrics['specificity_weighted'])
            cm = metrics['confusion_matrix']

            fold_metrics = {
                'accuracy': acc,
                'precision_macro': prec_macro,
                'recall_macro': rec_macro,
                'f1_macro': f1_macro,
                'precision_weighted': prec_weighted,
                'recall_weighted': rec_weighted,
                'f1_weighted': f1_weighted,
                'specificity_macro': spec_macro,
                'specificity_weighted': spec_weighted,
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_val.values, y_val_pred, zero_division=0)
            }

            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: metrics: f1_weighted={f1_weighted:.4f}, acc={acc:.4f}")
            
            # Log fold metrics to trial-level wandb run
            _log_fold_metrics_to_wandb(trial_wandb_run, fold, acc, f1_weighted, f1_macro,
                                       prec_weighted, prec_macro, rec_weighted, rec_macro,
                                       spec_macro, spec_weighted)
            
            fold_dir = os.path.join(TUNING_RESULTS_DIR, f"trial_{trial.number}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, f"fold_{fold+1}_metrics.json"), 'w') as fh:
                json.dump(fold_metrics, fh, indent=2)

            trial.set_user_attr(f"fold_{fold+1}_metrics", fold_metrics)
            trial.report(float(f1_weighted), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            scores.append(float(f1_weighted))

    mean_f1 = float(_np.mean(scores))
    std_f1 = float(_np.std(scores))
    # optionally attach aggregate metrics to trial
    trial.set_user_attr('mean_f1_weighted', mean_f1)
    trial.set_user_attr('std_f1_weighted', std_f1)

    # Log aggregate trial metrics to wandb
    if trial_wandb_run:
        try:
            wandb.log({
                'mean_f1_weighted': mean_f1,
                'std_f1_weighted': std_f1,
            })
            log.info(f"Logged aggregate metrics to W&B for trial {trial.number}")
        except Exception as e:
            log.warning(f"Failed to log aggregate metrics to wandb: {e}")
        
        # Finish the trial-level wandb run
        try:
            wandb.finish()
            log.info(f"Finished trial-level W&B run for trial {trial.number}")
        except Exception as e:
            log.warning(f"Failed to finish trial-level wandb run: {e}")

    log.info(f"--- Trial {trial.number} Finished: mean_f1_weighted = {mean_f1:.4f} ± {std_f1:.4f} ---")
    return mean_f1

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for QML models using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Tune Approach 1 (DRE) with standard QML
  python tune_models.py --datatype GeneExpr --approach 1 --qml_model standard --n_trials 50
  
  # Tune Approach 2 (CFE) with data-reuploading
  python tune_models.py --datatype CNV --approach 2 --qml_model reuploading --n_trials 100
  
  # Continue existing study with custom name
  python tune_models.py --datatype Meth --approach 1 --study_name my_study --n_trials 20
        """)
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('--datatype', type=str, required=True,
                         help='Data modality to tune (e.g., GeneExpr, CNV, Meth, miRNA, Prot, SNV)')
    required.add_argument('--approach', type=int, required=True, choices=[1, 2],
                         help='Preprocessing approach: 1=DRE (Dimensionality Reduction), 2=CFE (Feature Selection)')
    
    # Model configuration
    model_args = parser.add_argument_group('model configuration')
    model_args.add_argument('--qml_model', type=str, default='standard', choices=['standard', 'reuploading'],
                           help='QML circuit type (default: standard)')
    model_args.add_argument('--dim_reducer', type=str, default='umap', choices=['pca', 'umap'],
                           help='Dimensionality reducer for Approach 1/DRE (default: umap)')
    
    # Pretrained features configuration
    feature_args = parser.add_argument_group('pretrained features (optional)')
    feature_args.add_argument('--use_pretrained_features', action='store_true',
                             help='Use pretrained contrastive embeddings instead of raw features')
    feature_args.add_argument('--pretrained_features_dir', type=str, default=None,
                             help='Directory containing pretrained feature .npy files')
    
    # Tuning configuration
    tuning_args = parser.add_argument_group('tuning parameters')
    tuning_args.add_argument('--n_trials', type=int, default=9,
                            help='Number of new Optuna trials to run (default: 9)')
    tuning_args.add_argument('--total_trials', type=int, default=None,
                            help='Target total trials for existing study (computes remaining needed)')
    tuning_args.add_argument('--study_name', type=str, default=None,
                            help='Custom study name (auto-generated if not provided)')
    tuning_args.add_argument('--steps', type=int, default=100,
                            help='Training steps per trial (default: 100)')
    
    # Search space configuration
    search_args = parser.add_argument_group('hyperparameter search space')
    search_args.add_argument('--min_qbits', type=int, default=None,
                            help='Minimum qubits to search (default: num_classes)')
    search_args.add_argument('--max_qbits', type=int, default=12,
                            help='Maximum qubits to search (default: 12)')
    search_args.add_argument('--min_layers', type=int, default=2,
                            help='Minimum circuit layers (default: 2)')
    search_args.add_argument('--max_layers', type=int, default=5,
                            help='Maximum circuit layers (default: 5)')
    search_args.add_argument('--scalers', type=str, default='smr',
                            help='Scalers to try: s=Standard, m=MinMax, r=Robust (default: smr)')
    
    # Logging and monitoring
    log_args = parser.add_argument_group('logging and monitoring')
    log_args.add_argument('--verbose', action='store_true',
                         help='Enable detailed training logs')
    log_args.add_argument('--validation_frequency', type=int, default=10,
                         help='Validation metric frequency in steps (default: 10)')
    log_args.add_argument('--use_wandb', action='store_true',
                         help='Enable Weights & Biases experiment tracking')
    log_args.add_argument('--wandb_project', type=str, default=None,
                         help='W&B project name')
    log_args.add_argument('--wandb_run_name', type=str, default=None,
                         help='W&B run name (auto-generated if omitted)')
    
    args = parser.parse_args()
    
    # Setup interruption handlers
    signal.signal(signal.SIGINT, handle_interruption)
    signal.signal(signal.SIGTERM, handle_interruption)

    log.info(f"Starting hyperparameter tuning with arguments: {args}")

    # --- Scaler selection logic ---
    scaler_map = {'s': 'Standard', 'm': 'MinMax', 'r': 'Robust'}
    scaler_options = [scaler_map[char] for char in args.scalers if char in scaler_map]
    if not scaler_options:
        log.error("No valid scalers specified. Use 's', 'm', or 'r'. Exiting.")
        return
    log.info(f"Using scalers: {scaler_options}")

    # Check for pretrained features mode
    use_pretrained = args.use_pretrained_features and args.pretrained_features_dir is not None
    
    if use_pretrained:
        # Approach 2 (CFE) requires raw features with NaN for conditional encoding
        # Pretrained features don't have NaN (encoder handles missingness), so not compatible
        if args.approach == 2:
            log.error("Pretrained features are not compatible with Approach 2 (CFE/Conditional).")
            log.error("Approach 2 requires raw features with NaN values for missingness-aware encoding.")
            log.error("Use --approach 1 with pretrained features instead.")
            return
        
        # Load pretrained embeddings from numpy files
        pretrained_file = os.path.join(args.pretrained_features_dir, f'{args.datatype}_embeddings.npy')
        labels_file = os.path.join(args.pretrained_features_dir, 'labels.npy')
        case_ids_file = os.path.join(args.pretrained_features_dir, 'case_ids.npy')
        
        if not os.path.exists(pretrained_file):
            log.error(f"Pretrained features not found: {pretrained_file}")
            return
        if not os.path.exists(labels_file):
            log.error(f"Labels file not found: {labels_file}")
            return
        if not os.path.exists(case_ids_file):
            log.error(f"Case IDs file not found: {case_ids_file}")
            return
        
        log.info(f"Loading pretrained features from: {args.pretrained_features_dir}")
        X_np = np.load(pretrained_file)
        y_np = np.load(labels_file)
        case_ids = np.load(case_ids_file)
        
        # Convert to DataFrame/Series for compatibility with existing pipeline
        X = pd.DataFrame(X_np, index=case_ids)
        y_categorical = pd.Series(y_np, index=case_ids)
        
        # Labels are already encoded in pretrained features
        le = LabelEncoder()
        le.fit(y_categorical)
        y = pd.Series(le.transform(y_categorical), index=y_categorical.index)
        n_classes = len(le.classes_)
        
        log.info(f"Loaded pretrained features: X shape = {X.shape}, embed_dim = {X.shape[1]}")
        log.info(f"Detected {n_classes} classes: {list(le.classes_)}")
    else:
        # Original flow: load from parquet
        df = safe_load_parquet(os.path.join(SOURCE_DIR, f'data_{args.datatype}_.parquet'))
        if df is None: 
            log.error("Failed to load data, exiting.")
            return

        # CRITICAL: Sort by case_id for consistent ordering across all scripts
        df = df.sort_values('case_id').set_index('case_id')
        X = df.drop(columns=['class'])
        y_categorical = df['class']

        # Perform label encoding for the multiclass target
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_categorical), index=y_categorical.index)
        n_classes = len(le.classes_)
        log.info(f"Detected {n_classes} classes: {list(le.classes_)}")

    # Determine qubit search range
    min_qbits = args.min_qbits if args.min_qbits is not None else n_classes
    min_qbits = max(min_qbits, n_classes)
    if min_qbits % 2 != 0:
        min_qbits += 1
    max_qbits = args.max_qbits
    if max_qbits <= min_qbits:
        max_qbits = min_qbits + 2

    # Use custom study name if provided, otherwise generate one
    if args.study_name:
        study_name = args.study_name
        log.info(f"Using custom study name: {study_name}")
    else:
        pretrained_suffix = '_pretrained' if use_pretrained else ''
        study_name = f'multiclass_qml_tuning_{args.datatype}_app{args.approach}_{args.dim_reducer}_{args.qml_model}{pretrained_suffix}'
        log.info(f"Using auto-generated study name: {study_name}")
    
    # Ensure database is writable
    writable_db_path = ensure_writable_db(OPTUNA_DB_PATH)
    log.info(f"Using sqlite database: {writable_db_path}")
    
    # Ensure results directory is writable
    global TUNING_RESULTS_DIR
    TUNING_RESULTS_DIR = ensure_writable_results_dir(TUNING_RESULTS_DIR)
    log.info(f"Using results directory: {TUNING_RESULTS_DIR}")

    storage = f"sqlite:///{writable_db_path}"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )
    
    # Add fixed 'steps' to the study's user attributes
    study.set_user_attr('steps', args.steps)
    
    # Calculate number of trials to run
    existing_trials = len(study.trials)
    log.info(f"Study '{study_name}' has {existing_trials} existing trial(s)")
    
    if args.total_trials is not None:
        # Calculate remaining trials needed to reach total_trials
        n_trials_to_run = max(0, args.total_trials - existing_trials)
        if n_trials_to_run == 0:
            log.info(f"Target of {args.total_trials} total trials already reached. No new trials will be run.")
        else:
            log.info(f"Target: {args.total_trials} total trials. Running {n_trials_to_run} more trial(s).")
    else:
        # Use n_trials as number of NEW trials to run
        n_trials_to_run = args.n_trials
        log.info(f"Running {n_trials_to_run} new trial(s) (current total: {existing_trials})")
    
    if n_trials_to_run > 0:
        # Create a callback to check for interruption
        def interruption_callback(study, trial):
            if interrupted:
                log.warning("Interruption detected. Stopping optimization after this trial.")
                study.stop()
        
        try:
            study.optimize(
                lambda t: objective(t, args, X, y, n_classes, min_qbits, max_qbits, scaler_options, use_pretrained),
                n_trials=n_trials_to_run,
                callbacks=[interruption_callback]
            )
        except KeyboardInterrupt:
            log.warning("\nOptimization interrupted by user.")
        
        final_trial_count = len(study.trials)
        log.info(f"Optimization complete. Study now has {final_trial_count} total trial(s).")
    else:
        log.info("No trials to run. Retrieving existing best parameters.")

    log.info("--- Hyperparameter Tuning Complete ---")
    
    # Check if we have any completed trials
    if len(study.trials) == 0:
        log.error("No trials have been completed. Cannot save best parameters.")
        return
    
    # Clean up old trial directories (keep best + latest 2)
    cleanup_old_trials(TUNING_RESULTS_DIR, study, keep_best=True, keep_latest_n=2)
    
    try:
        log.info(f"Best hyperparameters found: {study.best_params}")
        log.info(f"Best value: {study.best_value:.4f}")
        
        best_params = study.best_params.copy()
        best_params['steps'] = args.steps
        
        # Log study-level summary to wandb
        if args.use_wandb:
            try:
                # Create a summary run for the entire study
                summary_run_name = f"study_summary_{args.datatype}_app{args.approach}_{args.qml_model}"
                study_run = wandb.init(
                    project=args.wandb_project,
                    name=summary_run_name,
                    config={
                        'study_name': study_name,
                        'datatype': args.datatype,
                        'approach': args.approach,
                        'qml_model': args.qml_model,
                        'dim_reducer': args.dim_reducer if args.approach == 1 else None,
                        'total_trials': len(study.trials),
                        'n_classes': n_classes,
                        'best_params': best_params,
                    },
                    reinit=True,
                    group=f"study_{args.datatype}_app{args.approach}_{args.qml_model}"
                )
                
                # Log best metrics
                wandb.log({
                    'best_mean_f1_weighted': study.best_value,
                    'best_n_qubits': best_params.get('n_qubits', 0),
                    'best_n_layers': best_params.get('n_layers', 0),
                    'best_scaler': best_params.get('scaler', 'N/A'),
                    'total_trials': len(study.trials),
                })
                
                # Log all trial results for comparison in a single batch
                all_trials_summary = {}
                for trial in study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        all_trials_summary[f'all_trials/trial_{trial.number}_mean_f1'] = trial.value
                        all_trials_summary[f'all_trials/trial_{trial.number}_n_qubits'] = trial.params.get('n_qubits', 0)
                        all_trials_summary[f'all_trials/trial_{trial.number}_n_layers'] = trial.params.get('n_layers', 0)
                        # Scaler is approach-specific: present for approach 1, absent for approach 2
                        all_trials_summary[f'all_trials/trial_{trial.number}_scaler'] = trial.params.get('scaler', 'N/A')
                
                if all_trials_summary:
                    wandb.log(all_trials_summary)
                
                log.info(f"Logged study summary to W&B: {summary_run_name}")
                wandb.finish()
            except Exception as e:
                log.warning(f"Failed to log study summary to wandb: {e}")
        
        os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)
        params_file = os.path.join(TUNING_RESULTS_DIR, f'best_params_{study_name}.json')
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        log.info(f"Saved best parameters to '{params_file}'")
        
        # Save trials dataframe (flat CSV) for offline analysis
        try:
            df_trials = study.trials_dataframe()
            df_file = os.path.join(TUNING_RESULTS_DIR, f'trials_{study_name}.csv')
            df_trials.to_csv(df_file, index=False)
            log.info(f"Saved trials dataframe to '{df_file}'")
        except Exception as e:
            log.warning(f"Could not save trials dataframe: {e}")
        
        # Save Optuna visualization plots
        try:
            from optuna import visualization
            optuna_plots_dir = os.path.join(TUNING_RESULTS_DIR, 'optuna_plots')
            os.makedirs(optuna_plots_dir, exist_ok=True)
            
            plot_configs = [
                ('param_importances.png', visualization.plot_param_importances),
                ('optimization_history.png', visualization.plot_optimization_history),
                ('slice.png', visualization.plot_slice),
                ('contour.png', visualization.plot_contour),
            ]
            
            for fn, plotter in plot_configs:
                try:
                    fig = plotter(study)
                    out_path = os.path.join(optuna_plots_dir, fn)
                    # Try to save as image if plotly/kaleido is available
                    try:
                        fig.write_image(out_path)
                        log.info(f"Saved Optuna plot: {fn}")
                    except Exception:
                        # Fallback: save as HTML
                        html_path = out_path.replace('.png', '.html')
                        fig.write_html(html_path)
                        log.info(f"Saved Optuna plot as HTML: {html_path.split('/')[-1]}")
                except Exception as e:
                    log.warning(f"Could not generate {fn}: {e}")
        except Exception as e:
            log.warning(f"Could not generate Optuna visualization plots: {e}")
            
    except ValueError as e:
        log.error(f"Could not retrieve best parameters: {e}")
        log.info("This may happen if no trials completed successfully.")

if __name__ == "__main__":
    main()
