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
from umap import UMAP
from utils.masked_transformers import MaskedTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.metrics_utils import compute_metrics

# Import the centralized logger
from logging_utils import log

# Import the corrected multiclass model with the improved "DR" naming
from qml_models import MulticlassQuantumClassifierDR

# --- Configuration ---
# Directories (configurable via environment variables)
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'base_learner_outputs_app1_standard')
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
    # fallback
    return MinMaxScaler()

# --- Load the master label encoder before starting ---
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
    description="Train Approach 1 (DRE) base learners with Standard QML circuits",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Examples:
  # Train all modalities with tuned parameters
  python dre_standard.py
  
  # Train specific modalities only
  python dre_standard.py --datatypes GeneExpr CNV
  
  # Override tuned parameters
  python dre_standard.py --n_qbits 8 --n_layers 4 --steps 200
  
  # Generate only OOF predictions for meta-learner
  python dre_standard.py --cv_only
    """)

# Data selection
data_args = parser.add_argument_group('data selection')
data_args.add_argument('--datatypes', nargs='+', type=str, default=None,
                      help='Modalities to train (default: all). Example: --datatypes GeneExpr CNV Prot')

# Pretrained features (for integration with performance extensions)
feature_args = parser.add_argument_group('pretrained features')
feature_args.add_argument('--use_pretrained_features', action='store_true',
                         help='Use pretrained embeddings instead of PCA/UMAP preprocessing')
feature_args.add_argument('--pretrained_features_dir', type=str, default=None,
                         help='Directory containing pretrained feature .npy files (e.g., GeneExpr_embeddings.npy)')
feature_args.add_argument('--dim_reducer', type=str, default='umap', choices=['pca', 'umap'],
                         help='Dimensionality reducer for pretrained features: pca or umap (default: umap)')

# Model parameters (override tuned values)
model_args = parser.add_argument_group('model parameters (override tuned values)')
model_args.add_argument('--n_qbits', type=int, default=None,
                       help='Number of qubits (default: from tuning or num_classes)')
model_args.add_argument('--n_layers', type=int, default=None,
                       help='Number of circuit layers (default: from tuning or 3)')
model_args.add_argument('--steps', type=int, default=None,
                       help='Training steps (default: from tuning or 100)')
model_args.add_argument('--scaler', type=str, default=None,
                       help="Scaler: 's'=Standard, 'm'=MinMax, 'r'=Robust (default: from tuning)")
model_args.add_argument('--skip_tuning', action='store_true',
                       help='Ignore tuned parameters, use CLI args or defaults')

# Training mode
mode_args = parser.add_argument_group('training mode (mutually exclusive)')
mode_args.add_argument('--skip_cross_validation', action='store_true',
                      help='Skip CV, train only on full training set')
mode_args.add_argument('--cv_only', action='store_true',
                      help='Generate only OOF predictions (no final model training)')

# Training configuration
train_args = parser.add_argument_group('training configuration')
train_args.add_argument('--max_training_time', type=float, default=None,
                       help='Maximum training hours (overrides --steps). Example: 11.5')
train_args.add_argument('--validation_frequency', type=int, default=10,
                       help='Validation metric frequency in steps (default: 10)')
train_args.add_argument('--validation_frac', type=float, default=0.1,
                       help='Fraction of training data for validation during QML training (default: 0.1)')

# Checkpointing
checkpoint_args = parser.add_argument_group('checkpointing')
checkpoint_args.add_argument('--checkpoint_frequency', type=int, default=50,
                            help='Checkpoint save frequency in steps (default: 50)')
checkpoint_args.add_argument('--keep_last_n', type=int, default=3,
                            help='Number of recent checkpoints to keep (default: 3)')
checkpoint_args.add_argument('--checkpoint_fallback_dir', type=str, default=None,
                            help='Alternative checkpoint directory if default is read-only')
checkpoint_args.add_argument('--resume', type=str, default=None,
                            choices=['best', 'latest', 'auto'],
                            help='Resume from checkpoint: best, latest, or auto (default: None)')

# Logging
log_args = parser.add_argument_group('logging')
log_args.add_argument('--verbose', action='store_true',
                     help='Enable detailed training logs')
log_args.add_argument('--use_wandb', action='store_true',
                     help='Enable Weights & Biases experiment tracking')
log_args.add_argument('--wandb_project', type=str, default=None,
                     help='W&B project name')
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
    # --- Find and Load the Tuned Hyperparameters (if not skipping) ---
    config = {}
    param_file_found = None
    
    if not args.skip_tuning:
        # Make the search more specific to avoid ambiguity
        search_pattern = f"_{data_type}_app1_"
        if os.path.exists(TUNING_RESULTS_DIR):
            for filename in os.listdir(TUNING_RESULTS_DIR):
                if search_pattern in filename and "_standard" in filename:
                    param_file_found = os.path.join(TUNING_RESULTS_DIR, filename)
                    break
        
        if param_file_found:
            log.info(f"--- Training Base Learner for: {data_type} (using params from {os.path.basename(param_file_found)}) ---")
            with open(param_file_found, 'r') as f:
                config = json.load(f)
            log.info(f"Loaded parameters: {json.dumps(config, indent=2)}")
        else:
            log.warning(f"No tuned parameter file found for {data_type} (standard, app1). Using CLI args or defaults.")
    else:
        log.info(f"--- Training Base Learner for: {data_type} (skipping tuned params, using CLI args or defaults) ---")

    # --- Set parameters from CLI arguments or use defaults ---
    # Default values if not specified
    if args.steps is not None:
        config['steps'] = args.steps
        log.info(f"Using steps from CLI: {args.steps}")
    elif 'steps' not in config:
        config['steps'] = 100  # default
        log.info(f"Using default steps: {config['steps']}")
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = get_scaler(config.get('scaler', 'MinMax'))

    # --- Check for pretrained features ---
    use_pretrained = args.use_pretrained_features and args.pretrained_features_dir is not None
    X_train_pretrained = None
    X_test_pretrained = None
    
    if use_pretrained:
        # Load pretrained embeddings
        pretrained_file = os.path.join(args.pretrained_features_dir, f'{data_type}_embeddings.npy')
        if os.path.exists(pretrained_file):
            log.info(f"  - Loading pretrained features from {pretrained_file}")
            embeddings = np.load(pretrained_file)
            
            # Also load labels to align indices
            labels_file = os.path.join(args.pretrained_features_dir, 'labels.npy')
            if os.path.exists(labels_file):
                # Embeddings should be in same order as original data
                # Split using same indices
                n_train = len(X_train)
                n_test = len(X_test)
                
                # Re-create the split indices
                all_indices = np.arange(len(embeddings))
                train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y.values)
                
                X_train_pretrained = embeddings[train_idx]
                X_test_pretrained = embeddings[test_idx]
                
                log.info(f"  - Pretrained features loaded: train={X_train_pretrained.shape}, test={X_test_pretrained.shape}")
            else:
                log.warning(f"  - labels.npy not found in {args.pretrained_features_dir}, falling back to standard pipeline")
                use_pretrained = False
        else:
            log.warning(f"  - Pretrained file not found: {pretrained_file}, falling back to standard pipeline")
            use_pretrained = False

    # --- Build the appropriate pipeline using TUNED params ---
    
    # Create checkpoint directory for this data type (needed for max_training_time OR resume)
    checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoints_{data_type}') if (args.max_training_time or args.resume) else None
    
    # Determine n_qubits based on pretrained features or config
    if use_pretrained:
        # For pretrained features, n_qubits = embedding dimension (typically 256)
        # We'll use PCA/UMAP to reduce to n_qubits if embedding dim > n_qubits
        embed_dim = X_train_pretrained.shape[1]
        n_qubits_final = min(config['n_qubits'], embed_dim)
        log.info(f"  - Using pretrained features pipeline (embed_dim={embed_dim}, n_qubits={n_qubits_final}, reducer={args.dim_reducer})")
        
        # Select dimensionality reducer based on argument
        if args.dim_reducer == 'umap':
            dim_reducer = UMAP(n_components=n_qubits_final, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
            log.info(f"  - Using UMAP for dimensionality reduction ({embed_dim} -> {n_qubits_final})")
        else:
            dim_reducer = PCA(n_components=n_qubits_final)
            log.info(f"  - Using PCA for dimensionality reduction ({embed_dim} -> {n_qubits_final})")
        
        pipeline = Pipeline([
            ('scaler', scaler),  # Just scale, no imputation needed
            ('dim_reducer', dim_reducer),  # Reduce to n_qubits using PCA or UMAP
            ('qml', MulticlassQuantumClassifierDR(
                n_qubits=n_qubits_final, 
                n_layers=config['n_layers'], 
                steps=config['steps'], 
                n_classes=n_classes, 
                verbose=args.verbose,
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
                wandb_run_name=args.wandb_run_name or f'dre_standard_{data_type}_pretrained'
            ))
        ])
        
        # Replace X_train/X_test with pretrained versions
        X_train = pd.DataFrame(X_train_pretrained, index=X_train.index)
        X_test = pd.DataFrame(X_test_pretrained, index=X_test.index)
    else:
        log.info("  - Using standard pipeline for all data types...")
        
        pipeline = Pipeline([
            ('imputer', MaskedTransformer(SimpleImputer(strategy='median'), fallback='raise')),
            ('scaler', MaskedTransformer(scaler, fallback='raise')),
            ('pca', MaskedTransformer(PCA(n_components=config['n_qubits']), fallback='raise')),
            ('qml', MulticlassQuantumClassifierDR(
                n_qubits=config['n_qubits'], 
                n_layers=config['n_layers'], 
                steps=config['steps'], 
                n_classes=n_classes, 
                verbose=args.verbose,
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
                wandb_run_name=args.wandb_run_name or f'dre_standard_{data_type}'
            ))
        ])
        
    # --- Generate and Save Multiclass Predictions ---
    if not args.skip_cross_validation:
        log.info("  - Generating OOF predictions with cross_val_predict...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        # Pass the unfitted pipeline to cross_val_predict. It handles fitting for each fold internally.
        oof_preds = cross_val_predict(pipeline, X_train, y_train, cv=skf, method='predict_proba', n_jobs=-1)
        oof_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
        pd.DataFrame(oof_preds, index=X_train.index, columns=oof_cols).to_csv(os.path.join(OUTPUT_DIR, f'train_oof_preds_{data_type}.csv'))
        log.info("  - OOF predictions generated and saved.")
    else:
        log.info("  - Skipping cross-validation as requested.")

    # If cv_only is set, skip final training and move to next data type
    if args.cv_only:
        log.info("  - Skipping final training as --cv_only was specified.")
        log.info(f"--- Completed OOF prediction generation for {data_type} ---")
        continue

    # --- Train Final Model on Full Training Data and Predict on Test Set ---
    log.info("  - Fitting final pipeline on the full training set...")
    pipeline.fit(X_train, y_train)
    
    # Log best weights step if available
    qml_model = pipeline.named_steps['qml']
    if hasattr(qml_model, 'best_step') and hasattr(qml_model, 'best_loss'):
        log.info(f"  - Best weights were obtained at step {qml_model.best_step} with loss: {qml_model.best_loss:.4f}")
    
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
        
        # Compute comprehensive metrics
        metrics = compute_metrics(y_test, y_test_pred, n_classes)
        log.info(f"Comprehensive Metrics for {data_type}:")
        log.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        log.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        log.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        log.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        log.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        log.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        log.info(f"  Specificity (macro): {metrics['specificity_macro']:.4f}")
        log.info(f"  Specificity (weighted): {metrics['specificity_weighted']:.4f}")
        
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
