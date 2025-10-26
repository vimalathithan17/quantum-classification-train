# Final Universal Tuning Script (`tune_models.py`)
import pandas as pd
import numpy as np
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

# Additional imports for comprehensive metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
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

# Directories configurable via environment
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
TUNING_RESULTS_DIR = os.environ.get('TUNING_RESULTS_DIR', 'tuning_results')
OPTUNA_DB_PATH = os.environ.get('OPTUNA_DB_PATH', './optuna_studies.db')
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))

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

def _per_class_specificity(cm_arr):
    """Compute per-class specificity from confusion matrix."""
    K = cm_arr.shape[0]
    speci = _np.zeros(K, dtype=float)
    total = cm_arr.sum()
    for i in range(K):
        TP = cm_arr[i, i]
        FP = cm_arr[:, i].sum() - TP
        FN = cm_arr[i, :].sum() - TP
        TN = total - (TP + FP + FN)
        denom = TN + FP
        speci[i] = float(TN / denom) if denom > 0 else 0.0
    return speci


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

def objective(trial, args, X, y, n_classes, min_qbits, max_qbits, scaler_options):
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

    # Build scaler only for Approach 1; for Approach 2 keep scaler as None
    if args.approach == 1:
        scaler = get_scaler(params['scaler'])
        # wrap scaler so it ignores all-zero rows during fit/transform
        scaler = MaskedTransformer(scaler)
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
        steps_list.append(('imputer', MaskedTransformer(SimpleImputer(strategy='median'))))
        steps_list.append(('scaler', scaler))

        if args.dim_reducer == 'pca':
            steps_list.append(('dim_reducer', MaskedTransformer(PCA(n_components=n_qubits))))
        else:
            steps_list.append(('dim_reducer', MaskedTransformer(UMAP(n_components=n_qubits, random_state=RANDOM_STATE))))

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

            # compute metrics
            acc = float(accuracy_score(y_val, y_val_pred))
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
            prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val, y_val_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_val, y_val_pred)

            # per-class specificity
            per_class_spec = _per_class_specificity(cm)
            spec_macro = float(_np.mean(per_class_spec))
            # weighted specificity (by support)
            support = _np.bincount(y_val)
            spec_weighted = float(_np.sum(per_class_spec * support) / support.sum()) if support.sum() > 0 else spec_macro

            # pack fold metrics
            fold_metrics = {
                'accuracy': acc,
                'precision_macro': float(prec_macro),
                'recall_macro': float(rec_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(prec_weighted),
                'recall_weighted': float(rec_weighted),
                'f1_weighted': float(f1_weighted),
                'specificity_macro': spec_macro,
                'specificity_weighted': spec_weighted,
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_val, y_val_pred, zero_division=0)
            }

            # logging
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: metrics: f1_weighted={f1_weighted:.4f}, acc={acc:.4f}")
            
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

            # compute metrics (same as above)
            acc = float(accuracy_score(y_val.values, y_val_pred))
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_val.values, y_val_pred, average='macro', zero_division=0)
            prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val.values, y_val_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_val.values, y_val_pred)
            per_class_spec = _per_class_specificity(cm)
            spec_macro = float(_np.mean(per_class_spec))
            support = _np.bincount(y_val.values)
            spec_weighted = float(_np.sum(per_class_spec * support) / support.sum()) if support.sum() > 0 else spec_macro

            fold_metrics = {
                'accuracy': acc,
                'precision_macro': float(prec_macro),
                'recall_macro': float(rec_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(prec_weighted),
                'recall_weighted': float(rec_weighted),
                'f1_weighted': float(f1_weighted),
                'specificity_macro': spec_macro,
                'specificity_weighted': spec_weighted,
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_val.values, y_val_pred, zero_division=0)
            }

            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: metrics: f1_weighted={f1_weighted:.4f}, acc={acc:.4f}")
            
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

    log.info(f"--- Trial {trial.number} Finished: mean_f1_weighted = {mean_f1:.4f} ± {std_f1:.4f} ---")
    return mean_f1

def main():
    parser = argparse.ArgumentParser(description="Universal QML tuning framework for multiclass problems.")
    parser.add_argument('--datatype', type=str, required=True, help="Data type (e.g., CNV, Meth)")
    parser.add_argument('--approach', type=int, required=True, choices=[1, 2], help="1: Classical+QML, 2: Conditional QML")
    parser.add_argument('--dim_reducer', type=str, default='pca', choices=['pca', 'umap'], help="For Approach 1: PCA or UMAP")
    parser.add_argument('--qml_model', type=str, default='standard', choices=['standard', 'reuploading'], help="QML circuit type")
    parser.add_argument('--n_trials', type=int, default=9, help="Number of NEW Optuna trials to run (if study exists, these are added to existing trials)")
    parser.add_argument('--total_trials', type=int, default=None, help="Target TOTAL number of trials. If study exists, computes remaining trials needed to reach this total.")
    parser.add_argument('--study_name', type=str, default=None, help="Override the auto-generated study name")
    parser.add_argument('--min_qbits', type=int, default=None, help="Minimum number of qubits for tuning.")
    parser.add_argument('--max_qbits', type=int, default=12, help="Maximum number of qubits for tuning.")
    parser.add_argument('--min_layers', type=int, default=2, help="Minimum number of layers for tuning.")
    parser.add_argument('--max_layers', type=int, default=5, help="Maximum number of layers for tuning.")
    parser.add_argument('--steps', type=int, default=100, help="Number of training steps for tuning.")
    parser.add_argument('--scalers', type=str, default='smr', help="String indicating which scalers to try (s: Standard, m: MinMax, r: Robust). E.g., 'sm' for Standard and MinMax.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for QML model training steps.")
    parser.add_argument('--validation_frequency', type=int, default=10, help="Compute validation metrics every N steps (default: 10)")
    parser.add_argument('--use_wandb', action='store_true', help="Enable Weights & Biases logging during tuning")
    parser.add_argument('--wandb_project', type=str, default=None, help="W&B project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="W&B run name (optional, auto-generated if not provided)")
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

    df = safe_load_parquet(os.path.join(SOURCE_DIR, f'data_{args.datatype}_.parquet'))
    if df is None: 
        log.error("Failed to load data, exiting.")
        return

    X = df.drop(columns=['case_id', 'class'])
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
        study_name = f'multiclass_qml_tuning_{args.datatype}_app{args.approach}_{args.dim_reducer}_{args.qml_model}'
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
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
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
                lambda t: objective(t, args, X, y, n_classes, min_qbits, max_qbits, scaler_options),
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
