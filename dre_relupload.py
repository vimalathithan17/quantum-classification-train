import pandas as pd
import os
import joblib
import json
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from utils.masked_transformers import MaskedTransformer
from sklearn.pipeline import Pipeline
from umap import UMAP
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.metrics_utils import compute_metrics

# Import the centralized logger
from logging_utils import log

# Import the corrected multiclass model with the improved "DR" naming
from qml_models import MulticlassQuantumClassifierDataReuploadingDR


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
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'base_learner_outputs_app1_reuploading')
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
    log.info(f"Master label encoder loaded successfully. Found {n_classes} classes: {list(le.classes_)}")
except FileNotFoundError:
    log.critical(f"Master label encoder not found at '{label_encoder_path}'. Please run the 'create_master_label_encoder.py' script first.")
    exit()

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="Train Approach 1 (DRE) base learners with Data-Reuploading QML circuits",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Examples:
  # Train all modalities with tuned parameters
  python dre_relupload.py
  
  # Train specific modalities with custom parameters
  python dre_relupload.py --datatypes miRNA Prot --n_qbits 8 --n_layers 4
  
  # Generate only OOF predictions
  python dre_relupload.py --cv_only
    """)

data_args = parser.add_argument_group('data selection')
data_args.add_argument('--datatypes', nargs='+', type=str, default=None,
                      help='Modalities to train (default: all). Example: --datatypes GeneExpr CNV')

# Pretrained features (for integration with performance extensions)
feature_args = parser.add_argument_group('pretrained features')
feature_args.add_argument('--use_pretrained_features', action='store_true',
                         help='Use pretrained embeddings instead of PCA/UMAP preprocessing')
feature_args.add_argument('--pretrained_features_dir', type=str, default=None,
                         help='Directory containing pretrained feature .npy files (e.g., GeneExpr_embeddings.npy)')
feature_args.add_argument('--dim_reducer', type=str, default='umap', choices=['pca', 'umap'],
                         help='Dimensionality reducer for pretrained features: pca or umap (default: umap)')

model_args = parser.add_argument_group('model parameters (override tuned values)')
model_args.add_argument('--n_qbits', type=int, default=None,
                       help='Number of qubits (default: from tuning or num_classes)')
model_args.add_argument('--n_layers', type=int, default=None,
                       help='Number of circuit layers (default: from tuning or 3)')
model_args.add_argument('--steps', type=int, default=None,
                       help='Training steps (default: from tuning or 100)')
model_args.add_argument('--scaler', type=str, default=None,
                       help="Scaler: 's'=Standard, 'm'=MinMax, 'r'=Robust")
model_args.add_argument('--skip_tuning', action='store_true',
                       help='Ignore tuned parameters, use CLI args or defaults')
model_args.add_argument('--weight_decay', type=float, default=1e-3,
                       help='L2 regularization (weight decay) for QML models (default: 1e-3, 0 to disable)')

mode_args = parser.add_argument_group('training mode (mutually exclusive)')
mode_args.add_argument('--skip_cross_validation', action='store_true',
                      help='Skip CV, train only on full training set')
mode_args.add_argument('--cv_only', action='store_true',
                      help='Generate only OOF predictions (no final model)')

train_args = parser.add_argument_group('training configuration')
train_args.add_argument('--max_training_time', type=float, default=None,
                       help='Maximum training hours (overrides --steps)')
train_args.add_argument('--validation_frequency', type=int, default=25,
                       help='Validation frequency in steps (default: 25)')
train_args.add_argument('--validation_frac', type=float, default=0.2,
                       help='Fraction of training data for validation during QML training (default: 0.2)')
train_args.add_argument('--patience', type=int, default=25,
                       help='Early stopping patience in steps (default: 25, 0 to disable)')

checkpoint_args = parser.add_argument_group('checkpointing')
checkpoint_args.add_argument('--checkpoint_frequency', type=int, default=50,
                            help='Checkpoint frequency in steps (default: 50)')
checkpoint_args.add_argument('--keep_last_n', type=int, default=3,
                            help='Checkpoints to keep (default: 3)')
checkpoint_args.add_argument('--checkpoint_fallback_dir', type=str, default=None,
                            help='Alternative checkpoint directory')
checkpoint_args.add_argument('--resume', type=str, default=None,
                            choices=['best', 'latest', 'auto'],
                            help='Resume from checkpoint: best, latest, or auto (default: None)')

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
    use_pretrained = args.use_pretrained_features and args.pretrained_features_dir is not None
    pretrained_has_split = False  # Track if pretrained features have proper train/test split
    
    if use_pretrained:
        # Check for properly split pretrained features (no leakage)
        train_emb_file = os.path.join(args.pretrained_features_dir, f'{data_type}_train_embeddings.npy')
        test_emb_file = os.path.join(args.pretrained_features_dir, f'{data_type}_test_embeddings.npy')
        train_labels_file = os.path.join(args.pretrained_features_dir, 'train_labels.npy')
        test_labels_file = os.path.join(args.pretrained_features_dir, 'test_labels.npy')
        train_case_ids_file = os.path.join(args.pretrained_features_dir, 'train_case_ids.npy')
        test_case_ids_file = os.path.join(args.pretrained_features_dir, 'test_case_ids.npy')
        
        # Also check old format (combined file)
        pretrained_file = os.path.join(args.pretrained_features_dir, f'{data_type}_embeddings.npy')
        labels_file = os.path.join(args.pretrained_features_dir, 'labels.npy')
        case_ids_file = os.path.join(args.pretrained_features_dir, 'case_ids.npy')
        
        if os.path.exists(train_emb_file) and os.path.exists(test_emb_file):
            # ✓ Properly split pretrained features - use directly without additional splitting
            log.info(f"  - Loading SPLIT pretrained features (no leakage)")
            pretrained_has_split = True
            
            train_embeddings = np.load(train_emb_file)
            test_embeddings = np.load(test_emb_file)
            train_labels_raw = np.load(train_labels_file, allow_pickle=True)
            test_labels_raw = np.load(test_labels_file, allow_pickle=True)
            train_case_ids = np.load(train_case_ids_file, allow_pickle=True)
            test_case_ids = np.load(test_case_ids_file, allow_pickle=True)
            
            # Convert to DataFrame/Series for compatibility
            X_train = pd.DataFrame(train_embeddings, index=train_case_ids)
            X_test = pd.DataFrame(test_embeddings, index=test_case_ids)
            y_train = pd.Series(le.transform(train_labels_raw), index=train_case_ids)
            y_test = pd.Series(le.transform(test_labels_raw), index=test_case_ids)
            
            log.info(f"  - Loaded train: {X_train.shape[0]} samples, {X_train.shape[1]} embed_dim")
            log.info(f"  - Loaded test:  {X_test.shape[0]} samples, {X_test.shape[1]} embed_dim")
            
        elif os.path.exists(pretrained_file) and os.path.exists(labels_file) and os.path.exists(case_ids_file):
            # ⚠️ Old format without split - warn about leakage
            log.warning(f"  - ⚠️ LEAKAGE WARNING: Using combined pretrained features!")
            log.warning(f"  - The encoder may have seen test samples during pretraining.")
            log.warning(f"  - For proper evaluation, re-run pretrain_contrastive.py with --test_size 0.2")
            log.warning(f"  - Then re-run extract_pretrained_features.py")
            
            embeddings = np.load(pretrained_file)
            labels_raw = np.load(labels_file, allow_pickle=True)
            case_ids = np.load(case_ids_file, allow_pickle=True)
            
            # Convert to DataFrame/Series for compatibility
            X = pd.DataFrame(embeddings, index=case_ids)
            y_categorical = pd.Series(labels_raw, index=case_ids)
            
            # Sort by case_id for consistent ordering
            X = X.sort_index()
            y_categorical = y_categorical.sort_index()
            
            # Encode labels
            y = pd.Series(le.transform(y_categorical), index=y_categorical.index)
            
            log.info(f"  - Loaded pretrained features: {X.shape[0]} samples, {X.shape[1]} embed_dim")
        else:
            log.warning(f"  - Pretrained files not found in {args.pretrained_features_dir}, falling back to parquet")
            use_pretrained = False
    
    if not use_pretrained:
        # Original flow: load from parquet
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
    
    # Deterministic train/test split (skip if pretrained features already split)
    if not pretrained_has_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    # Log data split information
    total_samples = len(X_train) + len(X_test)
    n_features = X_train.shape[1]
    log.info(f"  - Data split: Total={total_samples} samples, Train={len(X_train)} ({100*len(X_train)/total_samples:.1f}%), Test={len(X_test)} ({100*len(X_test)/total_samples:.1f}%)")
    log.info(f"  - Features: {n_features} {'pretrained embed_dim' if use_pretrained else 'raw features'}")
    train_class_counts = y_train.value_counts().sort_index()
    test_class_counts = y_test.value_counts().sort_index()
    log.info(f"  - Train class distribution: {dict(train_class_counts)}")
    log.info(f"  - Test class distribution: {dict(test_class_counts)}")
    
    scaler_obj = get_scaler(config.get('scaler', 'MinMax'))

    # --- Build the appropriate pipeline using TUNED params ---
    
    # Create checkpoint directory for this data type (needed for max_training_time OR resume)
    checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoints_{data_type}') if (args.max_training_time or args.resume) else None
    
    if use_pretrained:
        embed_dim = X_train.shape[1]
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
            ('scaler', scaler_obj),
            ('dim_reducer', dim_reducer),
            ('qml', MulticlassQuantumClassifierDataReuploadingDR(
                n_qubits=n_qubits_final, 
                n_layers=config['n_layers'], 
                steps=config['steps'], 
                n_classes=n_classes, 
                verbose=args.verbose,
                weight_decay=args.weight_decay,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fallback_dir=args.checkpoint_fallback_dir,
                checkpoint_frequency=args.checkpoint_frequency,
                keep_last_n=args.keep_last_n,
                max_training_time=args.max_training_time,
                validation_frequency=args.validation_frequency,
                validation_frac=args.validation_frac,
                patience=args.patience if args.patience > 0 else None,
                resume=args.resume,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name or f'dre_relupload_{data_type}_pretrained'
            ))
        ])
    else:
        log.info("  - Using standard pipeline for all data types...")
        
        pipeline = Pipeline([
            ('imputer', MaskedTransformer(SimpleImputer(strategy='median'), fallback='raise')),
            ('scaler', MaskedTransformer(scaler_obj, fallback='raise')),
            ('pca', MaskedTransformer(PCA(n_components=config['n_qubits']), fallback='raise')),
            ('qml', MulticlassQuantumClassifierDataReuploadingDR(
                n_qubits=config['n_qubits'], 
                n_layers=config['n_layers'], 
                steps=config['steps'], 
                n_classes=n_classes, 
                verbose=args.verbose,
                weight_decay=args.weight_decay,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fallback_dir=args.checkpoint_fallback_dir,
                checkpoint_frequency=args.checkpoint_frequency,
                keep_last_n=args.keep_last_n,
                max_training_time=args.max_training_time,
                validation_frequency=args.validation_frequency,
                validation_frac=args.validation_frac,
                patience=args.patience if args.patience > 0 else None,
                resume=args.resume,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name or f'dre_relupload_{data_type}'
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
    if hasattr(qml_model, 'best_step') and hasattr(qml_model, 'best_metric'):
        log.info(f"  - Best weights were obtained at step {qml_model.best_step} with metric ({getattr(qml_model, 'selection_metric', 'composite')}): {qml_model.best_metric:.4f}")
    
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, f'pipeline_{data_type}.joblib'))
    log.info(f"  - Saved final pipeline for {data_type}.")
    
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
        'dim_reducer': args.dim_reducer if use_pretrained else 'pca',
        'feature_dim': X_train.shape[1]
    }
    config_path = os.path.join(OUTPUT_DIR, f'preprocessing_config_{data_type}.json')
    with open(config_path, 'w') as f:
        json.dump(preprocessing_config, f, indent=2)
    log.info(f"  - Saved preprocessing config to {config_path}")

    # --- Classification report and confusion matrix on the hold-out test set ---
    try:
        # Load and prepare global_test set
        global_test_dir = os.environ.get('GLOBAL_TEST_DIR', 'data/global_test')
        X_test_eval, y_test_eval = None, None
        
        if use_pretrained:
            if pretrained_has_split:
                # If we're using properly split pre-extracted features, X_test already contains the global test set.
                X_test_eval, y_test_eval = X_test, y_test
                log.info(f"Using previously loaded split test features as the global test set.")
            else:
                test_features_dir = args.pretrained_features_dir.replace('train', 'test') if 'train' in args.pretrained_features_dir else os.path.join(global_test_dir, 'features')
                test_emb_file = os.path.join(test_features_dir, f'{data_type}_embeddings.npy')
                test_labels_file = os.path.join(test_features_dir, 'labels.npy')
                test_case_ids_file = os.path.join(test_features_dir, 'case_ids.npy')
                
                log.info(f"Loading pretrained test features from {test_features_dir}")
                try:
                    import numpy as np
                    import pandas as pd
                    embeddings_test = np.load(test_emb_file)
                    labels_raw_test = np.load(test_labels_file, allow_pickle=True)
                    case_ids_test = np.load(test_case_ids_file, allow_pickle=True)
                    
                    # Use same case_ids encoding and index
                    X_test_eval = pd.DataFrame(embeddings_test, index=case_ids_test)
                    y_categorical_test = pd.Series(labels_raw_test, index=case_ids_test)
                    
                    # Sort indices
                    X_test_eval = X_test_eval.sort_index()
                    y_categorical_test = y_categorical_test.sort_index()
                    
                    # Transform labels using the encoder fitted on train
                    y_test_eval = pd.Series(le.transform(y_categorical_test), index=y_categorical_test.index)
                except Exception as e:
                    log.warning(f"Could not load pretrained test features: {e}. Falling back to internal validation set.")
                    X_test_eval, y_test_eval = X_test, y_test
        else:
            file_path_test = os.path.join(global_test_dir, f'data_{data_type}_.parquet')
            log.info(f"Loading raw test features from {file_path_test}")
            try:
                import pandas as pd
                df_test = safe_load_parquet(file_path_test)
                if df_test is not None:
                    # Depending on module, handle ordering
                    id_col_local = ID_COL if 'ID_COL' in globals() or 'ID_COL' in locals() else 'case_id'
                    df_test = df_test.sort_values(id_col_local).set_index(id_col_local)
                    
                    if 'FEATURE_COLS' in globals() and globals()['FEATURE_COLS']:
                        X_test_eval = df_test[FEATURE_COLS]
                    else:
                        lbl_col_local = LABEL_COL_NAME if 'LABEL_COL_NAME' in globals() or 'LABEL_COL_NAME' in locals() else 'label'
                        exclude_cols = [lbl_col_local]
                        feature_cols_test = [c for c in df_test.columns if c not in exclude_cols]
                        X_test_eval = df_test[feature_cols_test]
                        
                    y_categorical_test = df_test[lbl_col_local]
                    y_test_eval = pd.Series(le.transform(y_categorical_test), index=y_categorical_test.index)
            except Exception as e:
                log.warning(f"Could not load raw test dataframe: {e}. Falling back to internal validation set.")
                X_test_eval, y_test_eval = X_test, y_test
                
        if X_test_eval is None or y_test_eval is None:
            log.warning("Test dataset could not be evaluated properly. Using internal validation set.")
            X_test_eval, y_test_eval = X_test, y_test
            
        y_test_pred = pipeline.predict(X_test_eval)
        
        # Save predictions based strictly on the global test set evaluations for MetaLearner
        test_preds = pipeline.predict_proba(X_test_eval)
        test_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
        pd.DataFrame(test_preds, index=X_test_eval.index, columns=test_cols).to_csv(os.path.join(OUTPUT_DIR, f'test_preds_{data_type}.csv'))
        log.info(f"Saved true global test predictions for {data_type} to test_preds_{data_type}.csv")

        acc = accuracy_score(y_test_eval, y_test_pred)
        log.info(f"Test Accuracy for {data_type}: {acc:.4f}")
        
        # Compute comprehensive metrics
        metrics = compute_metrics(y_test_eval, y_test_pred, n_classes)
        log.info(f"Comprehensive Metrics for {data_type}:")
        log.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        log.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        log.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        log.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        log.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        log.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        log.info(f"  Specificity (macro): {metrics['specificity_macro']:.4f}")
        log.info(f"  Specificity (weighted): {metrics['specificity_weighted']:.4f}")
        
        report = classification_report(y_test_eval, y_test_pred, labels=list(range(n_classes)), target_names=le.classes_)
        log.info(f"Classification Report for {data_type}:\n{report}")

        # Confusion matrix (raw)
        cm = confusion_matrix(y_test_eval, y_test_pred, labels=list(range(n_classes)))
        log.info(f"Confusion Matrix for {data_type} (rows=true, cols=pred):\n{cm}")

        # Save confusion matrix as CSV with class labels
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}.csv')
        cm_df.to_csv(cm_path)
        log.info(f"Saved confusion matrix to {cm_path}")

        # Normalized confusion matrix (per-row / true-class)
        import numpy as np
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=(row_sums != 0))
        cmn_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
        cmn_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}_normalized.csv')
        cmn_df.to_csv(cmn_path)
        log.info(f"Saved normalized confusion matrix to {cmn_path}")
        
        # Save comprehensive metrics to JSON
        import json
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

        # Log final test metrics to wandb
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'test/accuracy': float(metrics['accuracy']),
                    'test/f1_macro': float(metrics['f1_macro']),
                    'test/f1_weighted': float(metrics['f1_weighted']),
                    'test/precision_macro': float(metrics['precision_macro']),
                    'test/recall_macro': float(metrics['recall_macro']),
                    'test/specificity_macro': float(metrics['specificity_macro']),
                })
        except Exception as e:
            log.warning(f"Failed to log test metrics to wandb: {e}")
            
    except Exception as e:
        log.warning(f"Could not compute classification report or test on global_test for {data_type}: {e}")

    except Exception as e:
        log.warning(f"Could not compute classification report or confusion matrix for {data_type}: {e}")
