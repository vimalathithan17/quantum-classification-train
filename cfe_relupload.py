import pandas as pd
import os
import joblib
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the centralized logger
from logging_utils import log

# Import the corrected multiclass model
from qml_models import ConditionalMulticlassQuantumClassifierDataReuploadingFS

# --- Configuration ---
# Directories (configurable via environment variables)
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'base_learner_outputs_app2_reuploading')
TUNING_RESULTS_DIR = os.environ.get('TUNING_RESULTS_DIR', 'tuning_results')
ENCODER_DIR = os.environ.get('ENCODER_DIR', 'master_label_encoder')
ID_COL = 'case_id'
LABEL_COL = 'class'
DATA_TYPES_TO_TRAIN = ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))

os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_load_parquet(file_path):
    """Loads a parquet file with increased thrift limits."""
    limit = 1 * 1024**3
    try:
        return pd.read_parquet(
            file_path,
            thrift_string_size_limit=limit,
            thrift_container_size_limit=limit
        )
    except FileNotFoundError:
        log.error(f"File not found at {file_path}")
        return None
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

# --- Load the master label encoder ---
try:
    label_encoder_path = os.path.join(ENCODER_DIR, 'label_encoder.joblib')
    le = joblib.load(label_encoder_path)
    n_classes = len(le.classes_)
    log.info(f"Master label encoder loaded. Found {n_classes} classes: {list(le.classes_)}")
except FileNotFoundError:
    log.critical(f"Master label encoder not found at '{label_encoder_path}'.")
    log.critical("Please run the 'create_master_label_encoder.py' script first.")
    exit()

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train CFE Data Re-uploading models.")
parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for QML model training steps.")
parser.add_argument('--override_steps', type=int, default=None, help="Override the number of training steps from the tuned parameters.")
parser.add_argument('--n_qbits', type=int, default=None, help="Override number of qubits to use for training/pipeline.")
parser.add_argument('--n_layers', type=int, default=None, help="Override number of layers for QML ansatz.")
parser.add_argument('--steps', type=int, default=None, help="Override the number of training steps for QML models.")
parser.add_argument('--scaler', type=str, default=None, help="Override scaler choice: 's' (Standard), 'm' (MinMax), 'r' (Robust) or full name.")
parser.add_argument('--datatypes', nargs='+', type=str, default=None, help="Optional list of data types to train (overrides DATA_TYPES_TO_TRAIN). Example: --datatypes CNV Prot")
parser.add_argument('--skip_tuning', action='store_true', help="Skip loading tuned parameters and use command-line arguments or defaults instead.")
parser.add_argument('--skip_cross_validation', action='store_true', help="Skip cross-validation and only train final model on full training set.")
parser.add_argument('--cv_only', action='store_true', help="Perform only cross-validation to generate OOF predictions and skip final training (useful for meta-learner training).")
parser.add_argument('--max_training_time', type=float, default=None, help="Maximum training time in hours (overrides fixed steps). Example: --max_training_time 11")
parser.add_argument('--checkpoint_frequency', type=int, default=50, help="Save checkpoint every N steps (default: 50)")
parser.add_argument('--keep_last_n', type=int, default=3, help="Keep last N checkpoints (default: 3)")
parser.add_argument('--checkpoint_fallback_dir', type=str, default=None, help="Fallback directory for checkpoints if primary is read-only")
parser.add_argument('--validation_frequency', type=int, default=10, help="Compute validation metrics every N steps (default: 10)")
parser.add_argument('--use_wandb', action='store_true', help="Enable Weights & Biases logging")
parser.add_argument('--wandb_project', type=str, default=None, help="W&B project name")
parser.add_argument('--wandb_run_name', type=str, default=None, help="W&B run name")
args = parser.parse_args()

# Validate mutually exclusive arguments
if args.skip_cross_validation and args.cv_only:
    log.critical("Error: --skip_cross_validation and --cv_only are mutually exclusive. Choose one or neither.")
    exit(1)

# --- Main Training Loop ---
data_types = args.datatypes if args.datatypes is not None else DATA_TYPES_TO_TRAIN

for data_type in data_types:
    # --- Find and Load Tuned Hyperparameters (if not skipping) ---
    config = {}
    param_file_found = None
    
    if not args.skip_tuning:
        search_pattern = f"_{data_type}_app2_"
        if os.path.exists(TUNING_RESULTS_DIR):
            for filename in os.listdir(TUNING_RESULTS_DIR):
                if search_pattern in filename and "_reuploading" in filename:
                    param_file_found = os.path.join(TUNING_RESULTS_DIR, filename)
                    break
        
        if param_file_found:
            log.info(f"--- Training Base Learner for: {data_type} (Approach 2, Re-uploading) ---")
            with open(param_file_found, 'r') as f:
                config = json.load(f)
            log.info(f"Loaded parameters: {json.dumps(config, indent=2)}")
        else:
            log.warning(f"No tuned parameter file found for {data_type} (Approach 2, Re-uploading). Using CLI args or defaults.")
    else:
        log.info(f"--- Training Base Learner for: {data_type} (Approach 2, Re-uploading, skipping tuned params) ---")

    # --- Set parameters from CLI arguments or use defaults ---
    if args.steps is not None:
        config['steps'] = args.steps
    elif 'steps' not in config:
        config['steps'] = 100  # default
        log.info(f"Using default steps: {config['steps']}")
    
    if args.override_steps:
        config['steps'] = args.override_steps
        log.info(f"Overriding steps with: {args.override_steps}")
    
    if args.n_qbits is not None:
        config['n_qubits'] = args.n_qbits
    elif 'n_qubits' not in config:
        config['n_qubits'] = 10  # default for CFE approaches
        log.info(f"Using default n_qubits: {config['n_qubits']}")
    
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
    elif 'n_layers' not in config:
        config['n_layers'] = 3  # default
        log.info(f"Using default n_layers: {config['n_layers']}")
    
    if args.scaler is not None:
        config['scaler'] = args.scaler
    elif 'scaler' not in config:
        config['scaler'] = 'MinMax'  # default
        log.info(f"Using default scaler: {config['scaler']}")
    
    log.info(f"Final parameters - n_qubits: {config['n_qubits']}, n_layers: {config['n_layers']}, steps: {config['steps']}, scaler: {config['scaler']}")

    # --- Load Data and Encode Labels ---
    file_path = os.path.join(SOURCE_DIR, f'data_{data_type}_.parquet')
    df = safe_load_parquet(file_path)
    if df is None:
        continue
        
    # Ensure deterministic ordering and indexing by case_id
    df = df.sort_values(ID_COL).set_index(ID_COL)
    X = df.drop(columns=[LABEL_COL])
    y_categorical = df[LABEL_COL]
    y = pd.Series(le.transform(y_categorical), index=y_categorical.index)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    y_train, y_test = pd.Series(y_train, index=X_train.index), pd.Series(y_test, index=X_test.index)

    # --- Generate Out-of-Fold Predictions Correctly ---
    if not args.skip_cross_validation:
        log.info("  - Generating out-of-fold predictions...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        oof_preds = np.zeros((len(X_train), n_classes))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            log.info(f"    - Processing Fold {fold + 1}/3...")
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 1. Feature selection INSIDE the fold to prevent data leakage
            imputer_for_fs = SimpleImputer(strategy='median')
            X_train_fold_imputed = imputer_for_fs.fit_transform(X_train_fold)

            n_qubits = config.get('n_qubits', 10)

            log.info("      - Using LightGBM importance-based selection...")
            scaler = get_scaler(config.get('scaler', 'MinMax'))
            scaler.fit(X_train_fold_imputed)
            X_train_fold_scaled = scaler.transform(X_train_fold_imputed)

            # Lightweight LightGBM: fewer trees, feature subsampling, no verbose output
            lgb = LGBMClassifier(n_estimators=50, learning_rate=0.1, feature_fraction=0.7,
                                 n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)
            actual_k = min(n_qubits, X_train_fold_scaled.shape[1])
            lgb.fit(X_train_fold_scaled, y_train_fold)
            importances = lgb.feature_importances_
            top_idx = np.argsort(importances)[-actual_k:][::-1]
            selected_cols = X_train_fold.columns[top_idx]

            X_train_fold_selected = X_train_fold[selected_cols]
            X_val_fold_selected = X_val_fold[selected_cols]

            # 2. Prepare data tuple (mask, fill, scale) for this fold
            is_missing_train = X_train_fold_selected.isnull().astype(int).values
            X_train_filled = X_train_fold_selected.fillna(0.0).values
            is_missing_val = X_val_fold_selected.isnull().astype(int).values
            X_val_filled = X_val_fold_selected.fillna(0.0).values

            scaler = get_scaler(config.get('scaler', 'MinMax'))
            scaler.fit(X_train_filled)
            X_train_scaled = scaler.transform(X_train_filled)
            X_val_scaled = scaler.transform(X_val_filled)

            # 3. Train model on this fold and predict on the validation part
            checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoints_{data_type}_fold{fold+1}') if args.max_training_time else None
            model = ConditionalMulticlassQuantumClassifierDataReuploadingFS(
                n_qubits=n_qubits, n_layers=config['n_layers'],
                steps=config['steps'], n_classes=n_classes, verbose=args.verbose,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fallback_dir=args.checkpoint_fallback_dir,
                checkpoint_frequency=args.checkpoint_frequency,
                keep_last_n=args.keep_last_n,
                max_training_time=args.max_training_time,
                validation_frequency=args.validation_frequency,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name or f'cfe_relupload_{data_type}_fold{fold_idx}'
            )
            model.fit((X_train_scaled, is_missing_train), y_train_fold.values)
            val_preds = model.predict_proba((X_val_scaled, is_missing_val))
            oof_preds[val_idx] = val_preds

        oof_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
        pd.DataFrame(oof_preds, index=X_train.index, columns=oof_cols).to_csv(os.path.join(OUTPUT_DIR, f'train_oof_preds_{data_type}.csv'))
        log.info("  - Saved out-of-fold training predictions.")
    else:
        log.info("  - Skipping cross-validation as requested.")
    
    # If cv_only is set, skip final training and move to next data type
    if args.cv_only:
        log.info("  - Skipping final training as --cv_only was specified.")
        log.info(f"--- Completed OOF prediction generation for {data_type} ---")
        continue

    # --- Train Final Model on Full Training Data ---
    log.info("  - Training final model on full training data...")
    # Re-run feature selection on the full training data to determine the final feature set
    imputer_for_fs = SimpleImputer(strategy='median')
    X_train_imputed = imputer_for_fs.fit_transform(X_train)

    n_qubits = config.get('n_qubits', 10)
    log.info("    - Using LightGBM importance-based selection for final model...")
    scaler_for_fs = get_scaler(config.get('scaler', 'MinMax'))
    scaler_for_fs.fit(X_train_imputed)
    X_train_scaled_for_selection = scaler_for_fs.transform(X_train_imputed)

    lgb_final = LGBMClassifier(n_estimators=50, learning_rate=0.1, feature_fraction=0.7,
                               n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)
    actual_k = min(n_qubits, X_train_scaled_for_selection.shape[1])
    lgb_final.fit(X_train_scaled_for_selection, y_train)
    importances = lgb_final.feature_importances_
    top_idx = np.argsort(importances)[-actual_k:][::-1]
    final_selected_cols = X.columns[top_idx]
    joblib.dump(final_selected_cols, os.path.join(OUTPUT_DIR, f'selected_features_{data_type}.joblib'))
    log.info(f"    - Saved {len(final_selected_cols)} selected features for {data_type}.")

    X_train_selected = X_train[final_selected_cols]
    is_missing_train = X_train_selected.isnull().astype(int).values
    X_train_filled = X_train_selected.fillna(0.0).values
    final_scaler = get_scaler(config.get('scaler', 'MinMax'))
    X_train_scaled = final_scaler.fit_transform(X_train_filled)
    
    checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoints_{data_type}') if args.max_training_time else None
    final_model = ConditionalMulticlassQuantumClassifierDataReuploadingFS(
                n_qubits=n_qubits, n_layers=config['n_layers'],
                steps=config['steps'], n_classes=n_classes, verbose=args.verbose,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fallback_dir=args.checkpoint_fallback_dir,
                checkpoint_frequency=args.checkpoint_frequency,
                keep_last_n=args.keep_last_n,
                max_training_time=args.max_training_time,
                validation_frequency=args.validation_frequency,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name or f'cfe_relupload_{data_type}_fold{fold_idx}'
            )
    final_model.fit((X_train_scaled, is_missing_train), y_train.values)

    # Log best weights step if available
    if hasattr(final_model, 'best_step') and hasattr(final_model, 'best_loss'):
        log.info(f"  - Best weights were obtained at step {final_model.best_step} with loss: {final_model.best_loss:.4f}")

    # --- Generate Predictions on Test Set ---
    log.info("  - Generating predictions on the hold-out test set...")
    X_test_selected = X_test[final_selected_cols]
    is_missing_test = X_test_selected.isnull().astype(int).values
    X_test_filled = X_test_selected.fillna(0.0).values
    X_test_scaled = final_scaler.transform(X_test_filled)

    test_preds = final_model.predict_proba((X_test_scaled, is_missing_test))
    test_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(test_preds, index=X_test.index, columns=test_cols).to_csv(os.path.join(OUTPUT_DIR, f'test_preds_{data_type}.csv'))
    log.info("  - Saved test predictions.")

    # --- Save all components for inference ---
    joblib.dump(final_selected_cols, os.path.join(OUTPUT_DIR, f'selector_{data_type}.joblib'))
    joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, f'scaler_{data_type}.joblib'))
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, f'qml_model_{data_type}.joblib'))
    log.info(f"  - Saved final selector, scaler, and QML model for {data_type}.")
    log.info(f"  - Final selected features: {list(final_selected_cols)}")

    # --- Classification report on the hold-out test set ---
    try:
        test_preds_labels = np.argmax(test_preds, axis=1)
        acc = accuracy_score(y_test, test_preds_labels)
        log.info(f"Test Accuracy for {data_type}: {acc:.4f}")
        
        report = classification_report(y_test, test_preds_labels, labels=list(range(n_classes)), target_names=le.classes_)
        log.info(f"Classification Report for {data_type}:\n{report}")

        # Confusion matrix (raw)
        cm = confusion_matrix(y_test, test_preds_labels, labels=list(range(n_classes)))
        log.info(f"Confusion Matrix for {data_type} (rows=true, cols=pred):\n{cm}")

        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}.csv')
        cm_df.to_csv(cm_path)
        log.info(f"Saved confusion matrix to {cm_path}")

        # Normalized confusion matrix
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=(row_sums != 0))
        cmn_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
        cmn_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}_normalized.csv')
        cmn_df.to_csv(cmn_path)
        log.info(f"Saved normalized confusion matrix to {cmn_path}")
    except Exception as e:
        log.warning(f"Could not compute classification report or confusion matrix for {data_type}: {e}")
