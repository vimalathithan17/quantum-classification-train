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
from lightgbm import LGBMClassifier
from umap import UMAP

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
    params = {
        'scaler': trial.suggest_categorical('scaler', scaler_options),
        'n_qubits': trial.suggest_int('n_qubits', min_qbits, max_qbits, step=2),
        'n_layers': trial.suggest_int('n_layers', args.min_layers, args.max_layers)
    }

    log.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=2)}")

    scaler = get_scaler(params['scaler'])
    steps = args.steps  # Use steps from command-line arguments
    n_qubits = params['n_qubits']
    n_layers = params['n_layers']
    
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    if args.approach == 1:
        steps_list = []
        # Common steps for Approach 1
        steps_list.append(('imputer', SimpleImputer(strategy='median')))
        steps_list.append(('scaler', scaler))

        if args.dim_reducer == 'pca':
            steps_list.append(('dim_reducer', PCA(n_components=n_qubits)))
        else:
            steps_list.append(('dim_reducer', UMAP(n_components=n_qubits, random_state=RANDOM_STATE)))

        if args.qml_model == 'standard':
            qml_model = MulticlassQuantumClassifierDR(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=f"tune_DR_{datatype}_{trial.number}" if args.use_wandb else None)
        else: # reuploading
            qml_model = MulticlassQuantumClassifierDataReuploadingDR(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=f"tune_DRE_{datatype}_{trial.number}" if args.use_wandb else None)
            
        pipeline = Pipeline(steps_list + [('qml', qml_model)])
        
        # Perform cross-validation on the entire pipeline
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Starting training...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_val, y_val)
            scores.append(score)
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Completed with score {score:.4f}")

    elif args.approach == 2:
         
        if args.qml_model == 'standard':
            qml_model = ConditionalMulticlassQuantumClassifierFS(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=f"tune_CQFS_{datatype}_{trial.number}" if args.use_wandb else None)
        else: # reuploading
            qml_model = ConditionalMulticlassQuantumClassifierDataReuploadingFS(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose, validation_frequency=args.validation_frequency, use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_run_name=f"tune_CQDFS_{datatype}_{trial.number}" if args.use_wandb else None)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Starting training...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Perform fold-specific feature selection and preprocessing using LightGBM importances
            temp_imputer = SimpleImputer(strategy='median')
            X_train_imputed = temp_imputer.fit_transform(X_train)

            # Scale the data *before* feature selection
            scaler.fit(X_train_imputed)
            X_train_scaled_for_selection = scaler.transform(X_train_imputed)

            # Fit a LightGBM classifier to compute feature importances and pick top-k
            # Use a lightweight LightGBM configuration: fewer trees, feature subsampling
            lgb = LGBMClassifier(n_estimators=50, learning_rate=0.1, feature_fraction=0.7, 
                                 n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)
            # Guard: if the number of features is less than requested, pick all
            actual_k = min(n_qubits, X_train_scaled_for_selection.shape[1])
            lgb.fit(X_train_scaled_for_selection, y_train)
            importances = lgb.feature_importances_
            top_idx = np.argsort(importances)[-actual_k:][::-1]
            selected_cols = X_train.columns[top_idx]
            # Log selected features for this fold (Approach 2)
            log.info(f"    - Selected features (fold {fold+1}): {list(selected_cols)}")

            X_train_selected = X_train[selected_cols]
            X_val_selected = X_val[selected_cols]

            # Prepare the data tuple (mask, fill, scale) correctly for the model
            is_missing_train = X_train_selected.isnull().astype(int).values
            X_train_filled = X_train_selected.fillna(0.0).values
            is_missing_val = X_val_selected.isnull().astype(int).values
            X_val_filled = X_val_selected.fillna(0.0).values

            # Re-fit the scaler on the *selected* training data before transforming
            scaler.fit(X_train_filled)
            X_train_scaled = scaler.transform(X_train_filled)
            X_val_scaled = scaler.transform(X_val_filled)

            qml_model.fit((X_train_scaled, is_missing_train), y_train.values)
            score = qml_model.score((X_val_scaled, is_missing_val), y_val.values)
            scores.append(score)
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Completed with score {score:.4f}")

    average_accuracy = np.mean(scores)
    log.info(f"--- Trial {trial.number} Finished: Average Accuracy = {average_accuracy:.4f} ---")
    return average_accuracy

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
    except ValueError as e:
        log.error(f"Could not retrieve best parameters: {e}")
        log.info("This may happen if no trials completed successfully.")

if __name__ == "__main__":
    main()
