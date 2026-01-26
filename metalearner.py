import pandas as pd
import os
import argparse
import optuna
import joblib
import json
import shutil
import random
from sklearn.model_selection import train_test_split
# No feature scaling required for meta-learner (base learner outputs are probabilities)
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix
)
from utils.metrics_utils import compute_metrics
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


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    _np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not required for QML scripts
    log.info(f"Random seed set to {seed}")

# Environment-configurable directories
ENCODER_DIR = os.environ.get('ENCODER_DIR', 'master_label_encoder')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')
TUNING_JOURNAL_FILE = os.environ.get('TUNING_JOURNAL_FILE', 'tuning_journal.log')
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))

# Initialize random seeds for reproducibility
set_seed(RANDOM_STATE)


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
        return None, None, None, None, None, None

    # Load indicator features and encode labels using the master encoder
    try:
        indicators = pd.read_parquet(indicator_file)
        indicators.set_index('case_id', inplace=True)
    except FileNotFoundError:
        log.error(f"Indicator file not found at {indicator_file}")
        return None, None, None, None, None, None

    labels_categorical = indicators['class']
    labels = pd.Series(le.transform(labels_categorical), index=labels_categorical.index)
    indicators = indicators.drop(columns=['class'])
    # The indicator file encodes missingness: 1 == data missing for that base learner.
    # Convert to inclusion masks (1 == present) so downstream gated QML meta-learners
    # can multiply base predictions by this mask, avoiding direct indicator encoding.
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
        return None, None, None, None, None, None

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
            # To avoid collinearity given that each base learner's class
            # probabilities sum to 1, drop one class-column per base-learner
            # (e.g. the last sorted class) to make the feature matrix full-rank
            def _drop_one_per_base(pred_cols_list, encoder_last_class=None):
                """Return (keep_cols, dropped_map) where dropped_map maps datatype -> dropped_column

                We prefer to drop the column that corresponds to the encoder's last class
                (encoder_last_class) for each base learner. If that class isn't present
                among the base-learner's columns, fall back to dropping the last
                lexicographic column to keep behavior deterministic.
                """
                by_dt = {}
                for c in pred_cols_list:
                    rem = c[len(pred_prefix):] if c.startswith(pred_prefix) else c
                    dt = rem.split('_', 1)[0]
                    by_dt.setdefault(dt, []).append(c)
                keep = []
                dropped_map = {}
                for dt, cols in by_dt.items():
                    cols_sorted = sorted(cols)
                    if len(cols_sorted) <= 1:
                        # nothing to drop
                        keep.extend(cols_sorted)
                        dropped_map[dt] = None
                        continue

                    # Try to find the column corresponding to encoder_last_class
                    chosen_drop = None
                    if encoder_last_class is not None:
                        # construct candidate column name(s) that may match class label
                        # columns are like 'pred_{datatype}_{classname}'
                        for c in cols_sorted:
                            rem = c[len(pred_prefix):] if c.startswith(pred_prefix) else c
                            parts = rem.split('_')
                            # class label may contain underscores; compare suffix join
                            cls_part = '_'.join(parts[1:]) if len(parts) > 1 else ''
                            if str(cls_part) == str(encoder_last_class):
                                chosen_drop = c
                                break
                    if chosen_drop is None:
                        # Fallback: drop last lexicographic column
                        chosen_drop = cols_sorted[-1]

                    dropped_map[dt] = chosen_drop
                    for c in cols_sorted:
                        if c != chosen_drop:
                            keep.append(c)

                return keep, dropped_map

            # Use encoder's last class as preferred drop candidate
            encoder_last = None
            try:
                encoder_last = le.classes_[-1]
            except Exception:
                encoder_last = None

            reduced_pred_cols, dropped_map = _drop_one_per_base(pred_cols, encoder_last)
            dropped_cols = [v for v in dropped_map.values() if v]
            if dropped_cols:
                log.info(f"Dropped one redundant class column per base-learner: {dropped_cols}")
            X_meta_train_preds = X_meta_train_preds[reduced_pred_cols]

            # Persist mapping for traceability
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                mapping_path = os.path.join(OUTPUT_DIR, 'dropped_pred_columns.json')
                with open(mapping_path, 'w') as mf:
                    json.dump(dropped_map, mf, indent=2)
                log.info(f"Saved dropped-columns mapping to {mapping_path}")
            except Exception as e:
                log.warning(f"Failed to persist dropped-columns mapping to {OUTPUT_DIR}: {e}")
    if not X_meta_test_preds.empty:
        pred_cols_test = [c for c in X_meta_test_preds.columns if str(c).startswith(pred_prefix)]
        if pred_cols_test:
            # Apply the same column-dropping decision determined on the
            # training set so feature sets align. Ensure test set has the
            # identical columns and order; fill missing columns with zeros.
            if 'reduced_pred_cols' in locals():
                try:
                    # Reindex test columns to match reduced_pred_cols order; missing cols -> NaN
                    X_meta_test_preds = X_meta_test_preds.reindex(columns=reduced_pred_cols)
                    # Replace missing predictions with 0.0 so the feature shapes align
                    X_meta_test_preds = X_meta_test_preds.fillna(0.0)
                except Exception:
                    # If reindexing fails for any reason, fall back to intersection
                    reduced_test_cols = [c for c in reduced_pred_cols if c in X_meta_test_preds.columns]
                    X_meta_test_preds = X_meta_test_preds[reduced_test_cols]
            else:
                # Fallback: if training-side reduction wasn't computed, just keep test preds as-is
                X_meta_test_preds = X_meta_test_preds[pred_cols_test]

    # Join with indicator features. Use inner join to keep only aligned samples
    # that have both predictions and indicator rows. Ensure indices are named
    # consistently and unique.
    if not X_meta_train_preds.empty:
        X_meta_train_preds.index.name = 'case_id'
    if not X_meta_test_preds.empty:
        X_meta_test_preds.index.name = 'case_id'

    original_train_count = len(X_meta_train_preds) if not X_meta_train_preds.empty else 0
    original_test_count = len(X_meta_test_preds) if not X_meta_test_preds.empty else 0
    X_meta_train = X_meta_train_preds.join(indicators, how='inner') if not X_meta_train_preds.empty else pd.DataFrame()
    X_meta_test = X_meta_test_preds.join(indicators, how='inner') if not X_meta_test_preds.empty else pd.DataFrame()
    if original_train_count > 0 and len(X_meta_train) < int(0.9 * original_train_count):
        pct = (len(X_meta_train) / original_train_count) * 100.0
        log.warning(f"Meta-join alignment reduced training samples significantly: {len(X_meta_train)}/{original_train_count} ({pct:.1f}%). Check case_id consistency across files.")

    # Drop any rows with missing values after the join (conservative)
    if not X_meta_train.empty:
        X_meta_train = X_meta_train.dropna()
    if not X_meta_test.empty:
        X_meta_test = X_meta_test.dropna()

    # Validate resulting datasets
    if X_meta_train.empty:
        log.error("Assembled meta-training set is empty after joining predictions and indicators. Check input files and indices (case_id).")
        return None, None, None, None, None, None

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
    # Sanity-check: ensure that for the assembled training set the number of
    # base prediction columns matches the mask that would be constructed from
    # the indicators. This detects misalignment early and provides a clear
    # error message rather than letting downstream gated models fail.
    try:
        base_cols_train = [c for c in X_meta_train.columns if c not in indicator_cols]
        mask_check = _build_mask_from_indicators(X_meta_train, base_cols_train, indicator_cols)
        if mask_check.shape != (len(X_meta_train), len(base_cols_train)):
            raise ValueError(f"Assembled meta-train base_preds shape { (len(X_meta_train), len(base_cols_train)) } does not match mask shape {mask_check.shape}")
    except Exception as e:
        log.error(f"Meta-data assembly sanity check failed: {e}")
        # Propagate the error since downstream training would fail; caller may catch
        raise

    # Persist final feature column list for traceability (helps map model inputs)
    try:
        feats_path = os.path.join(OUTPUT_DIR, 'meta_train_feature_columns.json')
        with open(feats_path, 'w') as fh:
            json.dump({'base_prediction_columns': base_cols_train, 'indicator_columns': indicator_cols}, fh, indent=2)
        log.info(f"Saved meta-train feature columns to {feats_path}")
    except Exception:
        # Non-fatal; continue
        pass

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
    }
    # Use CLI-provided learning_rate (default 0.5) for tuning — we do not sample LR in Optuna
    params['learning_rate'] = float(args.learning_rate)
    params['steps'] = 100  # Fixed number of steps for tuning
    log.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=2)}")

    # NOTE: Do not scale meta-features — base learner outputs are probabilities [0,1]
    # Keep DataFrame form so we can split base preds and indicators when using the gated model.
    X_train_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    X_val_df = X_val if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val)

    # Determine n_qubits: for gated variants we only need qubits for base-learner outputs
    if params['qml_model'].startswith('gated'):
        if indicator_cols is None:
            raise ValueError("indicator_cols must be provided when using the gated meta-learner")
        base_cols = [c for c in X_train_df.columns if c not in indicator_cols]
        n_qubits_effective = len(base_cols)
    else:
        n_qubits_effective = X_train_df.shape[1]

    # Allow Optuna to search n_layers within CLI-provided bounds (clamped to sensible defaults)
    try:
        min_layers = int(args.minlayers) if getattr(args, 'minlayers', None) is not None else 3
    except Exception:
        min_layers = 3
    try:
        max_layers = int(args.maxlayers) if getattr(args, 'maxlayers', None) is not None else 6
    except Exception:
        max_layers = 6

    # Clamp and sanitize
    if min_layers < 1:
        min_layers = 1
    if max_layers < min_layers:
        # Fallback to sensible defaults if user provided inconsistent bounds
        min_layers = 3
        max_layers = 6

    # Let Optuna sample n_layers within [min_layers, max_layers]
    params['n_layers'] = trial.suggest_int('n_layers', min_layers, max_layers)

    # Construct a clear W&B run name for the meta-learner (no trial suffix by default)
    wandb_name = None
    if args.use_wandb:
        # Use DataFrame shape if available for clarity
        n_meta_feats = X_train_df.shape[1]
        wandb_name = f"meta_{params['qml_model']}_q{n_qubits_effective}_l{params['n_layers']}_lr{params['learning_rate']:.4g}"
    if args.wandb_run_name:
        wandb_name = args.wandb_run_name

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

    # Log meta-feature composition and chosen qubit count for traceability
    try:
        log.info(f"Trial {trial.number}: total_meta_features={X_train_df.shape[1]}")
        log.info(f"Trial {trial.number}: indicator_cols ({len(indicator_cols)}): {indicator_cols}")
        log.info(f"Trial {trial.number}: base_cols ({len(base_cols)}): {base_cols}")
        log.info(f"Trial {trial.number}: computed n_qubits_effective={n_qubits_effective}; model_params['n_qubits']={model_params.get('n_qubits')}")
    except Exception:
        # non-fatal logging failure should not stop tuning
        pass

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
    parser = argparse.ArgumentParser(
        description="Train or tune the QML meta-learner for ensemble stacking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Tune hyperparameters
  python metalearner.py --mode tune --preds_dir base_learner_outputs_app1_standard \
      --indicator_file final_processed_datasets/indicator_features.parquet --n_trials 50
  
  # Train final model with tuned parameters
  python metalearner.py --mode train --preds_dir base_learner_outputs_app1_standard \
      --indicator_file final_processed_datasets/indicator_features.parquet
  
  # Train with multiple prediction directories
  python metalearner.py --mode train \
      --preds_dir dir1 dir2 dir3 \
      --indicator_file indicator_features.parquet
        """)
    
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
    mode_args.add_argument('--skip_cross_validation', action='store_true',
                          help='Use train/val split instead of CV during tuning')
    
    # Model configuration (override tuned values for training)
    model_args = parser.add_argument_group('model parameters (override tuned values in train mode)')
    model_args.add_argument('--meta_model_type', type=str, 
                           choices=['gated_standard', 'gated_reuploading'], default=None,
                           help='Force meta-learner type (only gated variants supported)')
    model_args.add_argument('--meta_n_qubits', type=int, default=None,
                           help='Force number of qubits (default: num_meta_features)')
    model_args.add_argument('--meta_n_layers', type=int, default=None,
                           help='Force number of circuit layers')
    model_args.add_argument('--learning_rate', type=float, default=0.5,
                           help='Learning rate for training (default: 0.5)')
    model_args.add_argument('--steps', type=int, default=None,
                           help='Training steps (overrides tuned value)')
    
    # Tuning configuration
    tuning_args = parser.add_argument_group('tuning parameters')
    tuning_args.add_argument('--minlayers', type=int, default=3,
                            help='Min layers for tuning search space (default: 3)')
    tuning_args.add_argument('--maxlayers', type=int, default=6,
                            help='Max layers for tuning search space (default: 6)')
    
    # Training configuration
    train_args = parser.add_argument_group('training configuration')
    train_args.add_argument('--max_training_time', type=float, default=None,
                           help='Max training hours (overrides --steps)')
    train_args.add_argument('--validation_frequency', type=int, default=10,
                           help='Validation frequency in steps (default: 10)')
    train_args.add_argument('--validation_frac', type=float, default=0.1,
                           help='Fraction of training data for validation during QML training (default: 0.1)')
    
    # Checkpointing
    checkpoint_args = parser.add_argument_group('checkpointing')
    checkpoint_args.add_argument('--checkpoint_frequency', type=int, default=50,
                                help='Checkpoint frequency in steps (default: 50)')
    checkpoint_args.add_argument('--keep_last_n', type=int, default=3,
                                help='Checkpoints to keep (default: 3)')
    checkpoint_args.add_argument('--checkpoint_fallback_dir', type=str, default=None,
                                help='Alternative checkpoint directory')
    checkpoint_args.add_argument('--resume', type=str, default=None,
                                choices=['best', 'latest', 'auto'],
                                help='Resume from checkpoint: best (best validation), latest (most recent), auto (try best, fallback to latest)')
    
    # Logging
    log_args = parser.add_argument_group('logging')
    log_args.add_argument('--verbose', action='store_true',
                         help='Enable detailed training logs')
    log_args.add_argument('--use_wandb', action='store_true',
                         help='Enable W&B experiment tracking')
    log_args.add_argument('--wandb_project', type=str, default=None,
                         help='W&B project name')
    log_args.add_argument('--wandb_run_name', type=str, default=None,
                         help='W&B run name')
    
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
        if args.steps:
            params['steps'] = args.steps
            log.info(f"Using training steps from CLI: {args.steps}")
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
        checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints_metalearner') if (args.max_training_time or args.resume) else None
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
            'validation_frac': args.validation_frac,
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project,
            # Construct a clear W&B run name for the final training run
            'wandb_run_name': args.wandb_run_name or f"meta_train_{params.get('qml_model','model')}_q{n_qubits}_l{params['n_layers']}_lr{params['learning_rate']:.4g}",
            'resume': args.resume
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
        model_path = os.path.join(OUTPUT_DIR, 'meta_learner_final.joblib')
        joblib.dump(final_model, model_path)
        log.info(f"Final meta-learner model saved to '{model_path}'")
        
        # Save column order for inference (inference.py expects this file)
        columns_path = os.path.join(OUTPUT_DIR, 'meta_learner_columns.json')
        all_columns = list(X_meta_train.columns)  # base predictions + indicators in training order
        with open(columns_path, 'w') as fh:
            json.dump(all_columns, fh, indent=2)
        log.info(f"Saved meta-learner column order to '{columns_path}'")

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

        # Compute comprehensive metrics
        metrics = compute_metrics(y_meta_test.values, test_predictions, n_classes)
        log.info("Comprehensive Metrics:")
        log.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        log.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        log.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        log.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        log.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        log.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        log.info(f"  Specificity (macro): {metrics['specificity_macro']:.4f}")
        log.info(f"  Specificity (weighted): {metrics['specificity_weighted']:.4f}")

        # Generate and print classification report
        report = classification_report(y_meta_test.values, test_predictions, labels=list(range(n_classes)), target_names=le.classes_)
        log.info("Classification Report:\n" + report)

        # Save confusion matrix
        cm = confusion_matrix(y_meta_test.values, test_predictions, labels=list(range(n_classes)))
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.csv')
        cm_df.to_csv(cm_path)
        log.info(f"Saved confusion matrix to {cm_path}")

        # Normalized confusion matrix (per-row / true-class)
        with _np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = _np.divide(cm, row_sums, where=(row_sums != 0))
        cmn_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
        cmn_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.csv')
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
        metrics_path = os.path.join(OUTPUT_DIR, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        log.info(f"Saved comprehensive metrics to {metrics_path}")

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
