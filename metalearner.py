import pandas as pd
import os
import argparse
import optuna
import joblib
import json
import shutil
from sklearn.model_selection import train_test_split
# No feature scaling required for meta-learner (base learner outputs are probabilities)
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix
)
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import numpy as _np  # local alias to avoid shadowing pennylane.numpy
import sys

# Import the centralized logger
from logging_utils import log

# Import both DR model types for experimentation
from qml_models import (
    GatedMulticlassQuantumClassifierDR,
    GatedMulticlassQuantumClassifierDataReuploadingDR,
)

# Environment-configurable directories
ENCODER_DIR = os.environ.get('ENCODER_DIR', 'master_label_encoder')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')
TUNING_JOURNAL_FILE = os.environ.get('TUNING_JOURNAL_FILE', 'tuning_journal.log')
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))


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
        os.path.join(os.getcwd(), 'tuning_journal_working.log'),
        os.path.join(tempfile.gettempdir(), 'tuning_journal_working.log')
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
    Ensure the results directory is writable. If not, create a copy in current working dir.
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


def assemble_meta_data(preds_dirs, indicator_file):
    """Loads and combines base learner predictions from multiple directories."""
    log.info(f"--- Assembling data from: {preds_dirs} ---")

    # Load the master label encoder
    try:
        encoder_path = os.path.join(ENCODER_DIR, 'label_encoder.joblib')
        le = joblib.load(encoder_path)
        log.info(f"Master label encoder loaded from '{encoder_path}'")
    except FileNotFoundError:
        log.critical(f"Master label encoder not found in '{ENCODER_DIR}'.")
        log.critical("Please run the 'create_master_label_encoder.py' script first.")
        return None, None, None, None, None

    # Load indicator features and encode labels using the master encoder
    try:
        indicators = pd.read_parquet(indicator_file)
        indicators.set_index('case_id', inplace=True)
    except FileNotFoundError:
        log.error(f"Indicator file not found at {indicator_file}")
        return None, None, None, None, None

    labels_categorical = indicators['class']
    labels = pd.Series(le.transform(labels_categorical), index=labels_categorical.index)
    indicators = indicators.drop(columns=['class'])
    # The indicator file encodes missingness: 1 == data missing for that base learner.
    # Convert these missingness indicators into inclusion masks where
    # mask == 1 means the base-learner output is present and should be used.
    # This makes downstream code (which multiplies base_preds * mask) straightforward.
    try:
        # Fill NaNs conservatively as "not missing" (0) before inversion
        indicators = indicators.fillna(0)
        # Ensure numeric type then invert: missing(1) -> present(0) -> invert -> present(1)
        indicators = 1.0 - indicators.astype(float)
        # Clip to {0,1} in case of non-binary values
        indicators = indicators.clip(0.0, 1.0)
    except Exception:
        # If something goes wrong, leave indicators as-is and let consumers fail loudly
        log.warning("Failed to convert indicator missingness to inclusion masks; leaving original values")

    oof_preds_list = []
    test_preds_list = []

    # Loop through each provided prediction directory
    for preds_dir in preds_dirs:
        log.info(f"  - Loading predictions from '{preds_dir}'...")
        try:
            oof_files = [f for f in os.listdir(preds_dir) if f.startswith('train_oof_preds_')]
            test_files = [f for f in os.listdir(preds_dir) if f.startswith('test_preds_')]
            
            if not oof_files and not test_files:
                log.warning(f"No prediction files found in '{preds_dir}'. Skipping.")
                continue

            def _load_pred_file(path):
                """Read a predictions CSV and set case_id as the index (do not include it as a feature)."""
                df = pd.read_csv(path)
                # If case_id column exists, set it as the index and drop the column from features
                if 'case_id' in df.columns:
                    df = df.set_index('case_id')
                    # Standardize index name for downstream joins
                    df.index.name = 'case_id'
                else:
                    # If first column looks like an id column, set it as index
                    first_col = df.columns[0]
                    if first_col.lower() in ('case_id', 'caseid', 'id'):
                        df = df.set_index(first_col)
                        df.index.name = 'case_id'

                # Ensure we don't accidentally include an index named case_id as a column
                if 'case_id' in df.columns:
                    df = df.drop(columns=['case_id'])

                # Drop duplicate indices if present (keep first occurrence)
                if df.index.duplicated().any():
                    log.warning(f"Duplicate case_id values found in {path}; keeping first occurrence")
                    df = df[~df.index.duplicated(keep='first')]

                return df

            for f in oof_files:
                p = os.path.join(preds_dir, f)
                try:
                    oof_preds_list.append(_load_pred_file(p))
                except Exception as e:
                    log.error(f"Failed to read OOF predictions from {p}: {e}")
            for f in test_files:
                p = os.path.join(preds_dir, f)
                try:
                    test_preds_list.append(_load_pred_file(p))
                except Exception as e:
                    log.error(f"Failed to read test predictions from {p}: {e}")
        except FileNotFoundError:
            log.error(f"Prediction directory not found: '{preds_dir}'. Skipping.")
            continue
    
    if not oof_preds_list:
        log.error("No out-of-fold prediction files found in any of the provided directories.")
        return None, None, None, None, None

    # Concatenate all found predictions (align by case_id index)
    if oof_preds_list:
        X_meta_train_preds = pd.concat(oof_preds_list, axis=1, join='outer')
    else:
        X_meta_train_preds = pd.DataFrame()
    if test_preds_list:
        X_meta_test_preds = pd.concat(test_preds_list, axis=1, join='outer')
    else:
        X_meta_test_preds = pd.DataFrame()

    # Filter prediction columns to those that look like base-learner outputs
    pred_prefix = 'pred_'
    if not X_meta_train_preds.empty:
        pred_cols = [c for c in X_meta_train_preds.columns if str(c).startswith(pred_prefix)]
        if pred_cols:
            X_meta_train_preds = X_meta_train_preds[pred_cols]
    if not X_meta_test_preds.empty:
        pred_cols_test = [c for c in X_meta_test_preds.columns if str(c).startswith(pred_prefix)]
        if pred_cols_test:
            X_meta_test_preds = X_meta_test_preds[pred_cols_test]

    # Join with indicator features. Use inner join to keep only aligned samples
    # that have both predictions and indicator rows. Ensure indices are named
    # consistently and unique.
    if not X_meta_train_preds.empty:
        X_meta_train_preds.index.name = 'case_id'
    if not X_meta_test_preds.empty:
        X_meta_test_preds.index.name = 'case_id'

    X_meta_train = X_meta_train_preds.join(indicators, how='inner') if not X_meta_train_preds.empty else pd.DataFrame()
    X_meta_test = X_meta_test_preds.join(indicators, how='inner') if not X_meta_test_preds.empty else pd.DataFrame()

    # Drop any rows with missing values after the join (conservative)
    if not X_meta_train.empty:
        X_meta_train = X_meta_train.dropna()
    if not X_meta_test.empty:
        X_meta_test = X_meta_test.dropna()

    # Validate resulting datasets
    if X_meta_train.empty:
        log.error("Assembled meta-training set is empty after joining predictions and indicators. Check input files and indices (case_id).")
        return None, None, None, None, None

    if X_meta_test.empty:
        log.warning("Assembled meta-test set is empty after joining predictions and indicators. Continuing but test set will be empty.")
    
    # Align labels with the final set of samples. Use reindex to avoid KeyErrors
    # and drop any samples without labels (should be rare).
    y_meta_train = labels.reindex(X_meta_train.index)
    missing_train_labels = y_meta_train.isna().sum()
    if missing_train_labels > 0:
        log.warning(f"{missing_train_labels} samples in meta-train have no label after alignment; dropping them")
        keep_idx = y_meta_train.dropna().index
        X_meta_train = X_meta_train.loc[keep_idx]
        y_meta_train = y_meta_train.loc[keep_idx]

    if not X_meta_test.empty:
        y_meta_test = labels.reindex(X_meta_test.index)
        missing_test_labels = y_meta_test.isna().sum()
        if missing_test_labels > 0:
            log.warning(f"{missing_test_labels} samples in meta-test have no label after alignment; dropping them")
            keep_idx = y_meta_test.dropna().index
            X_meta_test = X_meta_test.loc[keep_idx]
            y_meta_test = y_meta_test.loc[keep_idx]
    else:
        y_meta_test = pd.Series(dtype=int)

    log.info(f"Meta-training data shape: {X_meta_train.shape}")
    log.info(f"Meta-test data shape: {X_meta_test.shape}")
    
    # Ensure indices are named and unique for safe reindexing downstream
    for df_name, df in (('train', X_meta_train), ('test', X_meta_test)):
        if df is not None and not df.empty:
            df.index.name = 'case_id'
            if df.index.duplicated().any():
                dup_count = df.index.duplicated().sum()
                log.warning(f"{dup_count} duplicated case_id(s) found in meta-{df_name}; keeping first occurrence")
                df = df[~df.index.duplicated(keep='first')]
            if df_name == 'train':
                X_meta_train = df
            else:
                X_meta_test = df

    # Align and coerce labels to int, dropping any samples without labels
    y_meta_train = y_meta_train.reindex(X_meta_train.index)
    if y_meta_train.isna().any():
        missing = int(y_meta_train.isna().sum())
        log.warning(f"Dropping {missing} meta-train sample(s) with missing labels after reindexing")
        keep_idx = y_meta_train.dropna().index
        X_meta_train = X_meta_train.loc[keep_idx]
        y_meta_train = y_meta_train.loc[keep_idx]
    # Ensure integer dtype for labels
    y_meta_train = y_meta_train.astype(int)

    if not X_meta_test.empty:
        y_meta_test = y_meta_test.reindex(X_meta_test.index)
        if y_meta_test.isna().any():
            missing = int(y_meta_test.isna().sum())
            log.warning(f"Dropping {missing} meta-test sample(s) with missing labels after reindexing")
            keep_idx = y_meta_test.dropna().index
            X_meta_test = X_meta_test.loc[keep_idx]
            y_meta_test = y_meta_test.loc[keep_idx]
        y_meta_test = y_meta_test.astype(int)
    else:
        y_meta_test = pd.Series(dtype=int)

    # Also return the list of indicator column names (only those following
    # the 'is_missing_' prefix) so callers can split meta-features into
    # base-learner outputs and indicator masks when needed.
    indicator_cols = [c for c in indicators.columns if str(c).startswith('is_missing_')]
    return X_meta_train, y_meta_train, X_meta_test, y_meta_test, le, indicator_cols


def _build_mask_from_indicators(df, base_cols, indicator_cols):
    """Build a mask array that aligns per-column with base prediction columns.

    The base prediction columns have names like 'pred_{datatype}_{class}'. The
    indicator columns are per-datatype with names like 'is_missing_{datatype}_'.
    This function expands the per-datatype indicators into a per-prediction
    mask (repeating each indicator for the number of class-columns produced by
    that base learner) so the resulting mask has the same shape as the base
    predictions matrix.
    """
    if len(indicator_cols) == 0:
        # No indicators available; assume all present
        return _np.ones((len(df), len(base_cols)), dtype=float)

    # Prepare mapping from normalized datatype -> full indicator column name
    indicator_map = {}
    for ic in indicator_cols:
        key = ic.replace('is_missing_', '').strip('_').lower()
        indicator_map[key] = ic

    mask_columns = []
    # If the df doesn't contain the indicator columns (e.g., when passing a
    # subset DataFrame), fall back to zeros/ones appropriately
    indicators_df = df[indicator_cols].copy() if all(c in df.columns for c in indicator_cols) else None

    for col in base_cols:
        # Expect pattern 'pred_{datatype}_{class...}', extract datatype
        rem = col[len('pred_'):] if col.startswith('pred_') else col
        datatype = rem.split('_', 1)[0].lower()

        ind_col = indicator_map.get(datatype)
        if ind_col is None:
            # Try fuzzy match (substring) as a fallback
            for k, v in indicator_map.items():
                if datatype in k or k in datatype:
                    ind_col = v
                    break

        if indicators_df is None or ind_col is None or ind_col not in indicators_df.columns:
            # Default to "present" (1.0) if we can't find an indicator
            mask_columns.append(_np.ones(len(df), dtype=float))
        else:
            # indicators were inverted earlier in assemble_meta_data so values
            # represent presence (1.0) or absence (0.0). Use them directly.
            mask_columns.append(indicators_df[ind_col].astype(float).values)

    # Stack into a 2D numpy array with shape (n_samples, n_base_cols)
    if mask_columns:
        mask_arr = _np.column_stack(mask_columns)
    else:
        mask_arr = _np.empty((len(df), 0))
    return mask_arr

def objective(trial, X_train, y_train, X_val, y_val, n_classes, args, indicator_cols=None):
    """Defines one trial for tuning the meta-learner."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Log suggested parameters
    # Meta-features are probabilities (0..1) from base learners plus indicator features.
    # They do not require additional scaling; don't tune scalers for the meta-learner.
    # qml_model and n_layers remain tunable. Use a fixed learning rate if provided
    params = {
        # Only gated variants are considered in this workflow
        'qml_model': trial.suggest_categorical('qml_model', ['gated_standard', 'gated_reuploading']),
        'n_layers': trial.suggest_int('n_layers', 3, 6),
    }
    # Use CLI-provided learning_rate (default 0.5) for tuning — we do not sample LR in Optuna
    params['learning_rate'] = float(args.learning_rate)
    params['steps'] = 100  # Fixed number of steps for tuning
    log.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=2)}")

    # NOTE: Do not scale meta-features — base learner outputs are probabilities [0,1]
    # Keep DataFrame form so we can split base preds and indicators when using the gated model.
    X_train_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    X_val_df = X_val if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val)

    # Construct a clear W&B run name for the meta-learner (no trial suffix by default)
    wandb_name = None
    if args.use_wandb:
        # Use DataFrame shape if available for clarity
        n_meta_feats = X_train_df.shape[1]
        wandb_name = f"meta_{params['qml_model']}_q{n_meta_feats}_l{params['n_layers']}_lr{params['learning_rate']:.4g}"
    if args.wandb_run_name:
        wandb_name = args.wandb_run_name

    # Determine n_qubits: for gated variants we only need qubits for base-learner outputs
    if params['qml_model'].startswith('gated'):
        if indicator_cols is None:
            raise ValueError("indicator_cols must be provided when using the gated meta-learner")
        base_cols = [c for c in X_train_df.columns if c not in indicator_cols]
        n_qubits_effective = len(base_cols)
    else:
        n_qubits_effective = X_train_df.shape[1]

    model_params = {
        'n_qubits': n_qubits_effective,
        'n_layers': params['n_layers'],
        'learning_rate': params['learning_rate'],
        'steps': params['steps'],
        'n_classes': n_classes,
        'verbose': args.verbose,
        'validation_frequency': args.validation_frequency,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_run_name': wandb_name
    }

    # Instantiate gated variant according to the trial suggestion
    if params['qml_model'] == 'gated_standard':
        model = GatedMulticlassQuantumClassifierDR(**model_params)
    else:  # gated_reuploading
        model = GatedMulticlassQuantumClassifierDataReuploadingDR(**model_params)

    log.info(f"Trial {trial.number}: Training {params['qml_model']} model...")
    # For gated variants always split into base-predictions and indicator mask
    base_cols = [c for c in X_train_df.columns if c not in indicator_cols]
    X_base_train = X_train_df[base_cols].values
    # Build a mask that matches the shape of X_base_train (one column per base-pred)
    X_mask_train = _build_mask_from_indicators(X_train_df, base_cols, indicator_cols)
    log.info(f"Trial {trial.number}: X_base_train shape={X_base_train.shape}, X_mask_train shape={X_mask_train.shape}")
    if X_base_train.shape != X_mask_train.shape:
        raise ValueError(f"Shape mismatch before fit: base_preds {X_base_train.shape} vs mask {X_mask_train.shape}")
    model.fit((X_base_train, X_mask_train), y_train.values)

    log.info(f"Trial {trial.number}: Evaluating...")
    # Construct validation inputs for gated model the same way
    base_cols_val = [c for c in X_val_df.columns if c not in indicator_cols]
    X_base_val = X_val_df[base_cols_val].values
    X_mask_val = _build_mask_from_indicators(X_val_df, base_cols_val, indicator_cols)
    log.info(f"Trial {trial.number}: X_base_val shape={X_base_val.shape}, X_mask_val shape={X_mask_val.shape}")
    if X_base_val.shape != X_mask_val.shape:
        raise ValueError(f"Shape mismatch before predict: base_preds {X_base_val.shape} vs mask {X_mask_val.shape}")
    predictions = model.predict((X_base_val, X_mask_val))
    
    # Compute comprehensive metrics
    accuracy = float(accuracy_score(y_val.values, predictions))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_val.values, predictions, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val.values, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_val.values, predictions)
    
    # Per-class specificity
    per_class_spec = _per_class_specificity(cm)
    spec_macro = float(_np.mean(per_class_spec))
    # Weighted specificity (by support)
    support = _np.bincount(y_val.values)
    spec_weighted = float(_np.sum(per_class_spec * support) / support.sum()) if support.sum() > 0 else spec_macro
    
    # Pack comprehensive metrics
    metrics = {
        'accuracy': accuracy,
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(prec_weighted),
        'recall_weighted': float(rec_weighted),
        'f1_weighted': float(f1_weighted),
        'specificity_macro': spec_macro,
        'specificity_weighted': spec_weighted,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_val.values, predictions, zero_division=0)
    }
    
    log.info(f"Trial {trial.number}: metrics: f1_weighted={f1_weighted:.4f}, acc={accuracy:.4f}")
    
    # Save comprehensive metrics to disk
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, "metrics.json"), 'w') as fh:
        json.dump(metrics, fh, indent=2)
    
    # Attach metrics to the Optuna trial for later inspection
    trial.set_user_attr('metrics', metrics)
    
    log.info(f"--- Trial {trial.number} Finished: f1_weighted = {f1_weighted:.4f} ---")
    return float(f1_weighted)  # Optimize for weighted F1 instead of accuracy

def main():
    parser = argparse.ArgumentParser(description="Train or tune the QML meta-learner.")
    parser.add_argument('--preds_dir', nargs='+', required=True, help="One or more directories with base learner predictions.")
    parser.add_argument('--indicator_file', type=str, required=True, help="Path to the parquet file with indicator features and labels.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune'], help="Operation mode: 'train' a final model or 'tune' hyperparameters.")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials for tuning.")
    parser.add_argument('--override_steps', type=int, default=None, help="Override the number of training steps from the tuned parameters.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for QML model training steps.")
    parser.add_argument('--skip_cross_validation', action='store_true', help="Skip cross-validation during tuning (use simple train/val split).")
    parser.add_argument('--max_training_time', type=float, default=None, help="Maximum training time in hours (overrides fixed steps). Example: --max_training_time 11")
    parser.add_argument('--checkpoint_frequency', type=int, default=50, help="Save checkpoint every N steps (default: 50)")
    parser.add_argument('--keep_last_n', type=int, default=3, help="Keep last N checkpoints (default: 3)")
    parser.add_argument('--checkpoint_fallback_dir', type=str, default=None, help="Fallback directory for checkpoints if primary is read-only")
    parser.add_argument('--validation_frequency', type=int, default=10, help="Compute validation metrics every N steps (default: 10)")
    parser.add_argument('--use_wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument('--wandb_project', type=str, default=None, help="W&B project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="W&B run name")
    # New CLI args for meta-learner training/tuning
    parser.add_argument('--meta_model_type', type=str, choices=['gated_standard', 'gated_reuploading'], default=None,
                        help="Force meta-learner model type for final training (overrides tuned value). Only gated variants are supported.")
    parser.add_argument('--meta_n_layers', type=int, default=None,
                        help="Force number of layers for meta-learner during final training (overrides tuned value).")
    parser.add_argument('--meta_n_qubits', type=int, default=None,
                        help="Force number of qubits to use for the meta-learner (defaults to n_meta_features).")
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help="Fixed learning rate to use for both tuning and final training (default: 0.5). If passed on the CLI it will override tuned params for final training.")
    args = parser.parse_args()

    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le, indicator_cols = assemble_meta_data(args.preds_dir, args.indicator_file)
    if X_meta_train is None:
        log.critical("Failed to assemble meta-dataset. Exiting.")
        return

    n_classes = len(le.classes_)
    log.info(f"Meta-learner will be trained on {n_classes} classes.")

    # Ensure output directory is writable
    global OUTPUT_DIR
    OUTPUT_DIR = ensure_writable_results_dir(OUTPUT_DIR)
    log.info(f"Using output directory: {OUTPUT_DIR}")
    # Print and save assembled meta-features for inspection
    try:
        train_feats_file = os.path.join(OUTPUT_DIR, 'meta_features_train.csv')
        test_feats_file = os.path.join(OUTPUT_DIR, 'meta_features_test.csv')
        # Save with index (case_id) preserved
        X_meta_train.to_csv(train_feats_file, index=True)
        X_meta_test.to_csv(test_feats_file, index=True)
        log.info(f"Saved assembled meta features to '{train_feats_file}' and '{test_feats_file}'")
        # Log a concise preview (columns and first few rows)
        log.info(f"Meta-train shape: {X_meta_train.shape}; columns: {list(X_meta_train.columns)}")
        try:
            log.info("Meta-train sample:\n" + X_meta_train.head().to_string())
        except Exception:
            # to_string can fail on very large / exotic dtypes; fall back to shape only
            log.info(f"Meta-train preview unavailable; shape={X_meta_train.shape}")
    except Exception as e:
        log.error(f"Failed to save or print assembled meta-features: {e}")
    if args.mode == 'tune':
        log.info(f"--- Starting Hyperparameter Tuning for Meta-Learner ({args.n_trials} trials) ---")

        # Split training data for validation during tuning (stratified)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_meta_train, y_meta_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_meta_train
        )

        study_name = 'qml_metalearner_tuning'

        # Ensure database file is writable
        writable_journal_path = ensure_writable_db(TUNING_JOURNAL_FILE)
        log.info(f"Using journal file: {writable_journal_path}")

        storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=writable_journal_path))
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)

        # Use a fixed learning rate for tuning if provided via CLI; objective will read args.learning_rate
        study.optimize(lambda t: objective(t, X_train_split, y_train_split, X_val_split, y_val_split, n_classes, args, indicator_cols), n_trials=args.n_trials)

        log.info("--- Tuning Complete ---")
        log.info(f"Best hyperparameters found: {study.best_params}")
        log.info(f"Best value (weighted F1): {study.best_value:.4f}")

        # Save best parameters (learning_rate isn't included if fixed via CLI)
        params_file = os.path.join(OUTPUT_DIR, 'best_metalearner_params.json')
        with open(params_file, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        log.info(f"Saved best meta-learner parameters to '{params_file}'")

    elif args.mode == 'train':
        log.info("--- Training Final Meta-Learner ---")
        params_path = os.path.join(OUTPUT_DIR, 'best_metalearner_params.json')
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
            log.info(f"Loaded best parameters from '{params_path}': {json.dumps(params, indent=2)}")
        except FileNotFoundError:
            log.warning(f"Best parameter file not found at '{params_path}'. Using default parameters.")
            # Define sensible defaults if tuning was skipped
            params = {'qml_model': 'reuploading', 'n_layers': 3, 'learning_rate': 0.5, 'steps': 100}

        # Override steps if provided via command line
        if args.override_steps:
            params['steps'] = args.override_steps
            log.info(f"Overriding training steps with: {args.override_steps}")
    # Do NOT scale meta-features: base-learner outputs are probabilities in [0,1]

        # Allow CLI overrides for final training params
        if args.meta_model_type:
            # CLI must specify gated variants; accept legacy names too and map them
            params['qml_model'] = args.meta_model_type
            log.info(f"Overriding qml_model with CLI value: {args.meta_model_type}")
        if args.meta_n_layers:
            params['n_layers'] = args.meta_n_layers
            log.info(f"Overriding n_layers with CLI value: {args.meta_n_layers}")
        # Only override tuned params if the learning_rate was explicitly provided on the CLI
        if '--learning_rate' in sys.argv:
            params['learning_rate'] = float(args.learning_rate)
            log.info(f"Overriding learning_rate with CLI value: {params['learning_rate']}")

        # Prepare model with loaded or default parameters
        # Normalize/Map qml_model to gated variants if necessary
        if not str(params.get('qml_model', '')).startswith('gated'):
            # map legacy values to gated variants
            if params.get('qml_model') == 'standard':
                params['qml_model'] = 'gated_standard'
            else:
                # default map reuploading (or any other) -> gated_reuploading
                params['qml_model'] = 'gated_reuploading'

        # Determine effective n_qubits. For gated variants we only need qubits for base preds
        if params['qml_model'].startswith('gated'):
            n_base = X_meta_train.shape[1] - len(indicator_cols)
            n_qubits = args.meta_n_qubits if args.meta_n_qubits is not None else n_base
        else:
            n_qubits = args.meta_n_qubits if args.meta_n_qubits is not None else X_meta_train.shape[1]
        checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints_metalearner') if args.max_training_time else None
        model_params = {
            'n_qubits': n_qubits,
            'n_layers': params['n_layers'],
            'learning_rate': params['learning_rate'],
            'steps': params['steps'],
            'n_classes': n_classes,
            'verbose': args.verbose,
            'checkpoint_dir': checkpoint_dir,
            'checkpoint_fallback_dir': args.checkpoint_fallback_dir,
            'checkpoint_frequency': args.checkpoint_frequency,
            'keep_last_n': args.keep_last_n,
            'max_training_time': args.max_training_time,
            'validation_frequency': args.validation_frequency,
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project,
            # Construct a clear W&B run name for the final training run
            'wandb_run_name': args.wandb_run_name or f"meta_train_{params.get('qml_model','model')}_q{n_qubits}_l{params['n_layers']}_lr{params['learning_rate']:.4g}"
        }

        # Instantiate only gated variants
        if params['qml_model'] == 'gated_standard':
            final_model = GatedMulticlassQuantumClassifierDR(**model_params)
        else:  # gated_reuploading
            final_model = GatedMulticlassQuantumClassifierDataReuploadingDR(**model_params)

        log.info(f"Training final {params['qml_model']} model with parameters: {json.dumps(model_params, indent=2)}")
        # Fit the gated model: always split into base preds and indicator mask
        base_cols = [c for c in X_meta_train.columns if c not in indicator_cols]
        X_base = X_meta_train[base_cols].values
        X_mask = _build_mask_from_indicators(X_meta_train, base_cols, indicator_cols)
        log.info(f"Final training: X_base shape={X_base.shape}, X_mask shape={X_mask.shape}")
        if X_base.shape != X_mask.shape:
            raise ValueError(f"Shape mismatch before final fit: base_preds {X_base.shape} vs mask {X_mask.shape}")
        final_model.fit((X_base, X_mask), y_meta_train.values)

        # Log best weights step if available
        if hasattr(final_model, 'best_step') and hasattr(final_model, 'best_loss'):
            log.info(f"  - Best weights were obtained at step {final_model.best_step} with loss: {final_model.best_loss:.4f}")

        # Save the trained model
        model_path = os.path.join(OUTPUT_DIR, 'metalearner_model.joblib')
        joblib.dump(final_model, model_path)
        log.info(f"Final meta-learner model saved to '{model_path}'")

        # Note: we do not save a scaler for the meta-learner because inputs are
        # already probabilities (0..1) from base learners plus indicator features.

        # Evaluate and save final predictions (no scaler applied)
        log.info("--- Evaluating on Test Set ---")
        base_cols = [c for c in X_meta_test.columns if c not in indicator_cols]
        X_base_test = X_meta_test[base_cols].values
        X_mask_test = _build_mask_from_indicators(X_meta_test, base_cols, indicator_cols)
        log.info(f"Final evaluation: X_base_test shape={X_base_test.shape}, X_mask_test shape={X_mask_test.shape}")
        if X_base_test.shape != X_mask_test.shape:
            raise ValueError(f"Shape mismatch before final predict: base_preds {X_base_test.shape} vs mask {X_mask_test.shape}")
        test_predictions = final_model.predict((X_base_test, X_mask_test))
        test_accuracy = accuracy_score(y_meta_test.values, test_predictions)
        log.info(f"Final Test Accuracy: {test_accuracy:.4f}")

        # Generate and print classification report
        report = classification_report(y_meta_test.values, test_predictions, labels=list(range(n_classes)), target_names=le.classes_)
        log.info("Classification Report:\n" + report)

        # Save predictions to a file
        preds_df = pd.DataFrame({
            'case_id': X_meta_test.index,
            'true_class': le.inverse_transform(y_meta_test.values),
            'predicted_class': le.inverse_transform(test_predictions)
        })
        preds_file = os.path.join(OUTPUT_DIR, 'final_predictions.csv')
        preds_df.to_csv(preds_file, index=False)
        log.info(f"Final predictions saved to '{preds_file}'")

if __name__ == "__main__":
    main()
