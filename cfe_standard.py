import pandas as pd
import os
import joblib
import json
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.metrics_utils import compute_metrics

# Import the centralized logger
from logging_utils import log

# Import the corrected multiclass model
from qml_models import ConditionalMulticlassQuantumClassifierFS


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

# --- Configuration ---
# Directories (configurable via environment variables)
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'base_learner_outputs_app2_standard')
TUNING_RESULTS_DIR = os.environ.get('TUNING_RESULTS_DIR', 'tuning_results')
ENCODER_DIR = os.environ.get('ENCODER_DIR', 'master_label_encoder')
ID_COL = 'case_id'
LABEL_COL = 'class'
DATA_TYPES_TO_TRAIN = ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))

# Initialize random seeds for reproducibility
set_seed(RANDOM_STATE)

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
parser = argparse.ArgumentParser(
    description="Train Approach 2 (CFE) base learners with Standard QML circuits",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Examples:
  # Train all modalities with tuned parameters
  python cfe_standard.py
  
  # Train specific modalities
  python cfe_standard.py --datatypes Meth SNV
  
  # Skip tuning, use custom parameters
  python cfe_standard.py --skip_tuning --n_qbits 10 --n_layers 4
    """)

data_args = parser.add_argument_group('data selection')
data_args.add_argument('--datatypes', nargs='+', type=str, default=None,
                      help='Modalities to train (default: all)')

# Pretrained features (for integration with performance extensions)
feature_args = parser.add_argument_group('pretrained features')
feature_args.add_argument('--use_pretrained_features', action='store_true',
                         help='Use pretrained embeddings instead of LightGBM feature selection')
feature_args.add_argument('--pretrained_features_dir', type=str, default=None,
                         help='Directory containing pretrained feature .npy files')

model_args = parser.add_argument_group('model parameters (override tuned values)')
model_args.add_argument('--n_qbits', type=int, default=None,
                       help='Number of qubits (default: from tuning)')
model_args.add_argument('--n_layers', type=int, default=None,
                       help='Circuit layers (default: from tuning or 3)')
model_args.add_argument('--steps', type=int, default=None,
                       help='Training steps (default: from tuning or 100)')
model_args.add_argument('--scaler', type=str, default=None,
                       help="Scaler: 's'=Standard, 'm'=MinMax, 'r'=Robust")
model_args.add_argument('--skip_tuning', action='store_true',
                       help='Use CLI args instead of tuned parameters')

mode_args = parser.add_argument_group('training mode (mutually exclusive)')
mode_args.add_argument('--skip_cross_validation', action='store_true',
                      help='Train only final model (no CV)')
mode_args.add_argument('--cv_only', action='store_true',
                      help='Generate OOF predictions only')

train_args = parser.add_argument_group('training configuration')
train_args.add_argument('--max_training_time', type=float, default=None,
                       help='Max training hours (overrides --steps)')
train_args.add_argument('--validation_frequency', type=int, default=10,
                       help='Validation frequency (default: 10 steps)')
train_args.add_argument('--validation_frac', type=float, default=0.1,
                       help='Fraction of training data for validation during QML training (default: 0.1)')

checkpoint_args = parser.add_argument_group('checkpointing')
checkpoint_args.add_argument('--checkpoint_frequency', type=int, default=50,
                            help='Checkpoint frequency (default: 50 steps)')
checkpoint_args.add_argument('--keep_last_n', type=int, default=3,
                            help='Checkpoints to keep (default: 3)')
checkpoint_args.add_argument('--checkpoint_fallback_dir', type=str, default=None,
                            help='Alternative checkpoint directory')
checkpoint_args.add_argument('--resume', type=str, default=None,
                            choices=['best', 'latest', 'auto'],
                            help='Resume from checkpoint: best, latest, or auto (default: None)')

log_args = parser.add_argument_group('logging')
log_args.add_argument('--verbose', action='store_true',
                     help='Detailed training logs')
log_args.add_argument('--use_wandb', action='store_true',
                     help='Enable W&B tracking')
log_args.add_argument('--wandb_project', type=str, default=None,
                     help='W&B project')
log_args.add_argument('--wandb_run_name', type=str, default=None,
                     help='W&B run name')

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
                if search_pattern in filename and "_standard" in filename:
                    param_file_found = os.path.join(TUNING_RESULTS_DIR, filename)
                    break
        
        if param_file_found:
            log.info(f"--- Training Base Learner for: {data_type} (Approach 2, Standard) ---")
            with open(param_file_found, 'r') as f:
                config = json.load(f)
            log.info(f"Loaded parameters: {json.dumps(config, indent=2)}")
        else:
            log.warning(f"No tuned parameter file found for {data_type} (Approach 2, Standard). Using CLI args or defaults.")
    else:
        log.info(f"--- Training Base Learner for: {data_type} (Approach 2, Standard, skipping tuned params) ---")

    # --- Set parameters from CLI arguments or use defaults ---
    if args.steps is not None:
        config['steps'] = args.steps
        log.info(f"Using steps from CLI: {args.steps}")
    elif 'steps' not in config:
        config['steps'] = 100  # default
        log.info(f"Using default steps: {config['steps']}")
    
    if args.n_qbits is not None:
        config['n_qubits'] = args.n_qbits
    elif 'n_qubits' not in config:
        # Unify default with number of classes for consistency
        config['n_qubits'] = n_classes
        log.info(f"Using default n_qubits (equal to n_classes): {config['n_qubits']}")
    
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
    elif 'n_layers' not in config:
        config['n_layers'] = 3  # default
        log.info(f"Using default n_layers: {config['n_layers']}")
    
    if args.scaler is not None:
        config['scaler'] = args.scaler
    elif 'scaler' not in config:
        config['scaler'] = 'MinMax'  # default (ignored for conditional models)
        log.info(f"Using default scaler: {config['scaler']}")
    # Warn: scaler is ignored for conditional models (Approach 2)
    if config.get('scaler') is not None:
        log.warning("Scaler parameter is ignored for conditional models in CFE. Missingness mask provides signal; no scaling is applied.")
    
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
    # Convert y back to pandas Series for easier indexing with iloc
    y_train, y_test = pd.Series(y_train, index=X_train.index), pd.Series(y_test, index=X_test.index)

    # --- Check for pretrained features ---
    use_pretrained = args.use_pretrained_features and args.pretrained_features_dir is not None
    
    if use_pretrained:
        pretrained_file = os.path.join(args.pretrained_features_dir, f'{data_type}_embeddings.npy')
        if os.path.exists(pretrained_file):
            log.info(f"  - Loading pretrained features from {pretrained_file}")
            embeddings = np.load(pretrained_file)
            
            # Load case_ids for proper alignment
            case_ids_file = os.path.join(args.pretrained_features_dir, 'case_ids.npy')
            if os.path.exists(case_ids_file):
                pretrained_case_ids = np.load(case_ids_file, allow_pickle=True)
                
                # Create mapping from case_id to embedding index
                case_id_to_idx = {str(cid): i for i, cid in enumerate(pretrained_case_ids)}
                
                # Select embeddings aligned with X_train and X_test by case_id
                try:
                    train_embed_idx = [case_id_to_idx[str(cid)] for cid in X_train.index]
                    test_embed_idx = [case_id_to_idx[str(cid)] for cid in X_test.index]
                except KeyError as e:
                    log.error(f"  - Case ID {e} not found in pretrained features. Ensure data was processed with same samples.")
                    missing_train = [cid for cid in X_train.index if str(cid) not in case_id_to_idx]
                    missing_test = [cid for cid in X_test.index if str(cid) not in case_id_to_idx]
                    if missing_train:
                        log.error(f"  - Missing from train: {missing_train[:10]}..." if len(missing_train) > 10 else f"  - Missing from train: {missing_train}")
                    if missing_test:
                        log.error(f"  - Missing from test: {missing_test[:10]}..." if len(missing_test) > 10 else f"  - Missing from test: {missing_test}")
                    raise
                
                X_train = pd.DataFrame(embeddings[train_embed_idx], index=X_train.index)
                X_test = pd.DataFrame(embeddings[test_embed_idx], index=X_test.index)
                
                log.info(f"  - Pretrained features aligned by case_id: train={X_train.shape}, test={X_test.shape}")
                log.info("  - Note: LightGBM feature selection will be skipped, using PCA instead")
            else:
                # Fallback: use positional indices (assumes same order as sorted data)
                log.warning(f"  - case_ids.npy not found, using positional alignment (assumes same order)")
                all_indices = np.arange(len(embeddings))
                train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y.values)
                X_train = pd.DataFrame(embeddings[train_idx], index=X_train.index)
                X_test = pd.DataFrame(embeddings[test_idx], index=X_test.index)
                log.info(f"  - Pretrained features loaded: train={X_train.shape}, test={X_test.shape}")
        else:
            log.warning(f"  - Pretrained file not found: {pretrained_file}, falling back to standard pipeline")
            use_pretrained = False

    # --- Generate Out-of-Fold Predictions Correctly ---
    if not args.skip_cross_validation:
        log.info("  - Generating out-of-fold predictions...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        oof_preds = np.zeros((len(X_train), n_classes))
        
        # This will hold the feature columns selected for the final model
        final_selected_cols = None 
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            log.info(f"    - Processing Fold {fold + 1}/3...")
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 1. Feature selection INSIDE the fold to prevent data leakage
            # Use the raw features (with NaNs) for LightGBM feature importance so
            # LightGBM can use its native handling of missingness. Do NOT impute
            # or scale here for selection; this preserves missingness signal.
            n_qubits = config.get('n_qubits', n_classes)  # fallback to n_classes for consistency

            log.info("      - Using LightGBM importance-based selection on raw data (no impute/scale)...")
            # Lightweight LightGBM: fewer trees, feature subsampling, no verbose output
            lgb = LGBMClassifier(n_estimators=50, learning_rate=0.1, feature_fraction=0.7,
                                 n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)
            # Pass the raw DataFrame values (contains np.nan) so LightGBM's native
            # missing value handling is used for feature selection.
            X_train_fold_for_selection = X_train_fold
            actual_k = min(n_qubits, X_train_fold_for_selection.shape[1])
            lgb.fit(X_train_fold_for_selection.values, y_train_fold)
            importances = lgb.feature_importances_
            top_idx = np.argsort(importances)[-actual_k:][::-1]
            selected_cols = X_train_fold.columns[top_idx]

            X_train_fold_selected = X_train_fold[selected_cols]
            X_val_fold_selected = X_val_fold[selected_cols]

            # 2. Prepare data tuple (mask, fill) for this fold.
            # We do not perform imputation/scaling for the conditional QML model —
            # the model will learn from the missingness mask. We still replace
            # NaNs by 0.0 when forming the numeric input array (placeholder values).
            is_missing_train = X_train_fold_selected.isnull().astype(int).values
            X_train_filled = X_train_fold_selected.fillna(0.0).values
            is_missing_val = X_val_fold_selected.isnull().astype(int).values
            X_val_filled = X_val_fold_selected.fillna(0.0).values
            # Shape assertions to ensure tuple inputs are aligned
            if X_train_filled.shape != is_missing_train.shape:
                raise ValueError(f"Train arrays shape mismatch for {data_type}: filled={X_train_filled.shape}, mask={is_missing_train.shape}")
            if X_val_filled.shape != is_missing_val.shape:
                raise ValueError(f"Val arrays shape mismatch for {data_type}: filled={X_val_filled.shape}, mask={is_missing_val.shape}")
            # No scaling applied for conditional models; use filled arrays directly.
            X_train_scaled = X_train_filled
            X_val_scaled = X_val_filled

            # 3. Train model on this fold and predict on the validation part
            checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoints_{data_type}_fold{fold+1}') if (args.max_training_time or args.resume) else None
            model = ConditionalMulticlassQuantumClassifierFS(
                n_qubits=n_qubits, n_layers=config['n_layers'], 
                steps=config['steps'], n_classes=n_classes, verbose=args.verbose,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fallback_dir=args.checkpoint_fallback_dir,
                checkpoint_frequency=args.checkpoint_frequency,
                keep_last_n=args.keep_last_n,
                max_training_time=args.max_training_time,
                validation_frequency=args.validation_frequency,
                validation_frac=args.validation_frac,
                resume=args.resume,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name or f'cfe_standard_{data_type}_fold{fold+1}'
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
    # Re-run feature selection on the full training data — again use raw data
    # (with NaNs) so LightGBM can incorporate missingness when computing importances.
    n_qubits = config.get('n_qubits', n_classes)  # fallback to n_classes for consistency
    log.info("    - Using LightGBM importance-based selection for final model on raw data (no impute/scale)...")
    lgb_final = LGBMClassifier(n_estimators=50, learning_rate=0.1, feature_fraction=0.7,
                               n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)
    X_train_for_selection = X_train
    actual_k = min(n_qubits, X_train_for_selection.shape[1])
    lgb_final.fit(X_train_for_selection.values, y_train)
    importances = lgb_final.feature_importances_
    top_idx = np.argsort(importances)[-actual_k:][::-1]
    final_selected_cols = X.columns[top_idx]
    log.info(f"    - Selected {len(final_selected_cols)} features for {data_type}.")
    log.info(f"    - Final selected features: {list(final_selected_cols)}")

    X_train_selected = X_train[final_selected_cols]
    is_missing_train = X_train_selected.isnull().astype(int).values
    X_train_filled = X_train_selected.fillna(0.0).values
    # No final scaler for conditional models (they learn from missingness).
    final_scaler = None
    if config.get('scaler') is not None:
        log.warning("Scaler parameter present but will be ignored for final conditional model training.")
    X_train_scaled = X_train_filled
    
    checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoints_{data_type}') if (args.max_training_time or args.resume) else None
    final_model = ConditionalMulticlassQuantumClassifierFS(
                n_qubits=n_qubits, n_layers=config['n_layers'], 
                steps=config['steps'], n_classes=n_classes, verbose=args.verbose,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fallback_dir=args.checkpoint_fallback_dir,
                checkpoint_frequency=args.checkpoint_frequency,
                keep_last_n=args.keep_last_n,
                max_training_time=args.max_training_time,
                validation_frequency=args.validation_frequency,
                validation_frac=args.validation_frac,
                resume=args.resume,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name or f'cfe_standard_{data_type}_final'
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
    if X_test_filled.shape != is_missing_test.shape:
        raise ValueError(f"Test arrays shape mismatch for {data_type}: filled={X_test_filled.shape}, mask={is_missing_test.shape}")
    X_test_filled = X_test_selected.fillna(0.0).values
    # No scaler to apply; use filled arrays directly.
    X_test_scaled = X_test_filled

    test_preds = final_model.predict_proba((X_test_scaled, is_missing_test))
    test_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(test_preds, index=X_test.index, columns=test_cols).to_csv(os.path.join(OUTPUT_DIR, f'test_preds_{data_type}.csv'))
    log.info("  - Saved test predictions.")

    # --- Save all components for inference ---
    joblib.dump(final_selected_cols, os.path.join(OUTPUT_DIR, f'selected_features_{data_type}.joblib'))
    # Save a sentinel (None) for the scaler so inference can detect absence of scaling.
    joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, f'scaler_{data_type}.joblib'))
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, f'qml_model_{data_type}.joblib'))
    log.info(f"  - Saved final selector (selected_features), scaler(sentinel), and QML model for {data_type}.")
    
    # --- Save preprocessing config for inference ---
    preprocessing_config = {
        'data_type': data_type,
        'n_qubits': config['n_qubits'],
        'n_layers': config['n_layers'],
        'steps': config['steps'],
        'scaler': config.get('scaler', 'MinMax'),
        'n_classes': n_classes,
        'class_names': list(le.classes_),
        'use_pretrained': use_pretrained,
        'random_state': RANDOM_STATE,
        'selected_features': list(final_selected_cols) if final_selected_cols is not None else None,
        'feature_dim': X_train.shape[1]
    }
    config_path = os.path.join(OUTPUT_DIR, f'preprocessing_config_{data_type}.json')
    with open(config_path, 'w') as f:
        json.dump(preprocessing_config, f, indent=2)
    log.info(f"  - Saved preprocessing config to {config_path}")

    # --- Classification report on the hold-out test set ---
    try:
        test_preds_labels = np.argmax(test_preds, axis=1)
        acc = accuracy_score(y_test, test_preds_labels)
        log.info(f"Test Accuracy for {data_type}: {acc:.4f}")
        
        # Compute comprehensive metrics
        metrics = compute_metrics(y_test, test_preds_labels, n_classes)
        log.info(f"Comprehensive Metrics for {data_type}:")
        log.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        log.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        log.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        log.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        log.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        log.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        log.info(f"  Specificity (macro): {metrics['specificity_macro']:.4f}")
        log.info(f"  Specificity (weighted): {metrics['specificity_weighted']:.4f}")
        
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
        
        # Save comprehensive metrics to JSON
        metrics_json = {
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_macro': float(metrics['recall_macro']),
            'recall_weighted': float(metrics['recall_weighted']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'specificity_macro': float(metrics['specificity_macro']),
            'specificity_weighted': float(metrics['specificity_weighted'])
        }
        metrics_path = os.path.join(OUTPUT_DIR, f'test_metrics_{data_type}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        log.info(f"Saved comprehensive metrics to {metrics_path}")
    except Exception as e:
        log.warning(f"Could not compute classification report or confusion matrix for {data_type}: {e}")
