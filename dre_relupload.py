import pandas as pd
import os
import joblib
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import the centralized logger
from logging_utils import log

# Import the corrected multiclass model with the improved "DR" naming
from qml_models import MulticlassQuantumClassifierDataReuploadingDR

# --- Configuration ---
# Directories (configurable via environment variables)
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'base_learner_outputs_app1_reuploading')
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
    log.info(f"Master label encoder loaded successfully. Found {n_classes} classes: {list(le.classes_)}")
except FileNotFoundError:
    log.critical(f"Master label encoder not found at '{label_encoder_path}'. Please run the 'create_master_encoder.py' script first.")
    exit()

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train DRE Data Re-uploading models.")
parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for QML model training steps.")
parser.add_argument('--override_steps', type=int, default=None, help="Override the number of training steps from the tuned parameters.")
parser.add_argument('--n_qbits', type=int, default=None, help="Override number of qubits to use for training/pipeline.")
parser.add_argument('--n_layers', type=int, default=None, help="Override number of layers for QML ansatz.")
parser.add_argument('--steps', type=int, default=None, help="Override the number of training steps for QML models.")
parser.add_argument('--scaler', type=str, default=None, help="Override scaler choice: 's' (Standard), 'm' (MinMax), 'r' (Robust) or full name.")
parser.add_argument('--datatypes', nargs='+', type=str, default=None, help="Optional list of data types to train (overrides DATA_TYPES_TO_TRAIN). Example: --datatypes CNV Prot")
parser.add_argument('--skip_tuning', action='store_true', help="Skip loading tuned parameters and use command-line arguments or defaults instead.")
parser.add_argument('--max_training_time', type=float, default=None, help="Maximum training time in hours (overrides fixed steps). Example: --max_training_time 11")
parser.add_argument('--checkpoint_frequency', type=int, default=50, help="Save checkpoint every N steps (default: 50)")
parser.add_argument('--keep_last_n', type=int, default=3, help="Keep last N checkpoints (default: 3)")
args = parser.parse_args()

# --- Main Training Loop ---
data_types = args.datatypes if args.datatypes is not None else DATA_TYPES_TO_TRAIN

for data_type in data_types:
    # --- Find and Load Tuned Hyperparameters (if not skipping) ---
    config = {}
    param_file_found = None
    
    if not args.skip_tuning:
        if os.path.exists(TUNING_RESULTS_DIR):
            for filename in os.listdir(TUNING_RESULTS_DIR):
                if f"_{data_type}_" in filename and "_app1_" in filename and "_reuploading" in filename:
                    param_file_found = os.path.join(TUNING_RESULTS_DIR, filename)
                    break
        
        if param_file_found:
            log.info(f"--- Training Base Learner for: {data_type} (Data Re-uploading) ---")
            log.info(f"    (using params from {os.path.basename(param_file_found)})")
            with open(param_file_found, 'r') as f:
                config = json.load(f)
            log.info(f"Loaded parameters: {json.dumps(config, indent=2)}")
        else:
            log.warning(f"No tuned parameter file found for {data_type} with re-uploading. Using CLI args or defaults.")
    else:
        log.info(f"--- Training Base Learner for: {data_type} (Data Re-uploading, skipping tuned params) ---")
    
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
        config['n_qubits'] = n_classes  # default to number of classes
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
        
    # Ensure deterministic ordering by sorting on case_id and set the index
    df = df.sort_values(ID_COL).set_index(ID_COL)
    X = df.drop(columns=[LABEL_COL])
    y_categorical = df[LABEL_COL]
    
    # Use the pre-loaded master encoder to transform labels
    y = pd.Series(le.transform(y_categorical), index=y_categorical.index)
    
    # Deterministic train/test split using the shared RANDOM_STATE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    scaler = get_scaler(config.get('scaler', 'MinMax'))

    # --- Build the appropriate pipeline using TUNED params ---
    log.info("  - Using standard pipeline for all data types...")
    
    # Create checkpoint directory for this data type
    checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoints_{data_type}') if args.max_training_time else None
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler),
        ('pca', PCA(n_components=config['n_qubits'])),
        ('qml', MulticlassQuantumClassifierDataReuploadingDR(
            n_qubits=config['n_qubits'], 
            n_layers=config['n_layers'], 
            steps=config['steps'], 
            n_classes=n_classes, 
            verbose=args.verbose,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=args.checkpoint_frequency,
            keep_last_n=args.keep_last_n,
            max_training_time=args.max_training_time
        ))
    ])
    # --- Generate and Save Multiclass Predictions ---
    log.info("  - Generating OOF predictions with cross_val_predict...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    # Pass the unfitted pipeline to cross_val_predict. It handles fitting for each fold internally.
    oof_preds = cross_val_predict(pipeline, X_train, y_train, cv=skf, method='predict_proba', n_jobs=-1)
    oof_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(oof_preds, index=X_train.index, columns=oof_cols).to_csv(os.path.join(OUTPUT_DIR, f'train_oof_preds_{data_type}.csv'))
    log.info("  - OOF predictions generated and saved.")

    # --- Train Final Model on Full Training Data and Predict on Test Set ---
    log.info("  - Fitting final pipeline on the full training set...")
    pipeline.fit(X_train, y_train)
    
    log.info("  - Generating predictions on the hold-out test set...")
    test_preds = pipeline.predict_proba(X_test)
    test_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(test_preds, index=X_test.index, columns=test_cols).to_csv(os.path.join(OUTPUT_DIR, f'test_preds_{data_type}.csv'))
    
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, f'pipeline_{data_type}.joblib'))
    log.info(f"  - Saved test predictions and final pipeline for {data_type}.")

    # --- Classification report and confusion matrix on the hold-out test set ---
    try:
        y_test_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        log.info(f"Test Accuracy for {data_type}: {acc:.4f}")
        
        report = classification_report(y_test, y_test_pred, labels=list(range(n_classes)), target_names=le.classes_)
        log.info(f"Classification Report for {data_type}:\n{report}")

        # Confusion matrix (raw)
        cm = confusion_matrix(y_test, y_test_pred, labels=list(range(n_classes)))
        log.info(f"Confusion Matrix for {data_type} (rows=true, cols=pred):\n{cm}")

        # Save confusion matrix as CSV with class labels
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}.csv')
        cm_df.to_csv(cm_path)
        log.info(f"Saved confusion matrix to {cm_path}")

        # Normalized confusion matrix (per-row / true-class)
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=(row_sums != 0))
        cmn_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
        cmn_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}_normalized.csv')
        cmn_df.to_csv(cmn_path)
        log.info(f"Saved normalized confusion matrix to {cmn_path}")
    except Exception as e:
        log.warning(f"Could not compute classification report or confusion matrix for {data_type}: {e}")
