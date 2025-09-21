import pandas as pd
import os
import joblib
import json
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_load_parquet(file_path):
    """Loads a parquet file with increased thrift limits."""
    limit = 1 * 1024**3
    return pd.read_parquet(
        file_path,
        thrift_string_size_limit=limit,
        thrift_container_size_limit=limit
    )

def get_scaler(scaler_name):
    """Returns a scaler object from a string name."""
    if scaler_name == 'MinMax': return MinMaxScaler()
    if scaler_name == 'Standard': return StandardScaler()
    if scaler_name == 'Robust': return RobustScaler()

# --- Load the master label encoder ---
try:
    label_encoder_path = os.path.join(ENCODER_DIR, 'label_encoder.joblib')
    le = joblib.load(label_encoder_path)
    n_classes = len(le.classes_)
    print(f"Master label encoder loaded successfully. Found {n_classes} classes: {list(le.classes_)}")
except FileNotFoundError:
    print(f"FATAL ERROR: Master label encoder not found at '{label_encoder_path}'. Please run the 'create_master_encoder.py' script first.")
    exit()

# --- Main Training Loop ---
for data_type in DATA_TYPES_TO_TRAIN:
    # --- Find and Load Tuned Hyperparameters ---
    param_file_found = None
    for filename in os.listdir(TUNING_RESULTS_DIR):
        if f"_{data_type}_" in filename and "_app1_" in filename and "_reuploading" in filename:
            param_file_found = os.path.join(TUNING_RESULTS_DIR, filename)
            break
    
    if not param_file_found:
        print(f"\n--- No tuned parameter file found for {data_type} with re-uploading. Skipping. ---")
        continue

    print(f"\n--- Training Base Learner for: {data_type} (Data Re-uploading) ---")
    print(f"    (using params from {os.path.basename(param_file_found)})")
    with open(param_file_found, 'r') as f:
        config = json.load(f)

    # --- Load Data and Encode Labels ---
    file_path = os.path.join(SOURCE_DIR, f'data_{data_type}_.parquet')
    df = safe_load_parquet(file_path)
    X = df.drop(columns=[ID_COL, LABEL_COL])
    y_categorical = df[LABEL_COL]
    
    # Use the master encoder to transform labels
    y = le.transform(y_categorical)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = get_scaler(config.get('scaler', 'MinMax'))

    # --- Build the appropriate pipeline using TUNED params ---
    if data_type == 'Meth':
        print("  - Using advanced pipeline for Meth data...")
        selector = SelectFromModel(
            lgb.LGBMClassifier(random_state=42, n_jobs=-1), 
            max_features=config['select_features']
        )
        pipeline = Pipeline([
            ('selector', selector),
            ('imputer', KNNImputer(n_neighbors=config['n_neighbors'])),
            ('scaler', scaler),
            ('pca', PCA(n_components=config['n_qubits'])),
            ('qml', MulticlassQuantumClassifierDataReuploadingDR(n_qubits=config['n_qubits'], n_layers=config['n_layers'], steps=config['steps'], n_classes=n_classes))
        ])
    elif data_type == 'SNV':
        print("  - Using simplified pipeline for SNV data...")
        n_snv_features = X_train.shape[1]
        pipeline = Pipeline([
            ('scaler', scaler),
            ('qml', MulticlassQuantumClassifierDataReuploadingDR(n_qubits=n_snv_features, n_layers=config['n_layers'], steps=config['steps'], n_classes=n_classes))
        ])
    else: # For CNV, GeneExpr, miRNA, Prot
        print("  - Using standard pipeline for dense data...")
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=config.get('n_neighbors', 5))),
            ('scaler', scaler),
            ('pca', PCA(n_components=config['n_qubits'])),
            ('qml', MulticlassQuantumClassifierDataReuploadingDR(n_qubits=config['n_qubits'], n_layers=config['n_layers'], steps=config['steps'], n_classes=n_classes))
        ])
        
    print("  - Fitting pipeline on the full training set...")
    pipeline.fit(X_train, y_train)

    # --- Generate and Save Multiclass Predictions ---
    print("  - Generating predictions...")
    oof_preds = cross_val_predict(pipeline, X_train, y_train, cv=3, method='predict_proba', n_jobs=-1)
    oof_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(oof_preds, index=X_train.index, columns=oof_cols).to_csv(os.path.join(OUTPUT_DIR, f'train_oof_preds_{data_type}.csv'))
    
    test_preds = pipeline.predict_proba(X_test)
    test_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(test_preds, index=X_test.index, columns=test_cols).to_csv(os.path.join(OUTPUT_DIR, f'test_preds_{data_type}.csv'))
    
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, f'pipeline_{data_type}.joblib'))
    print(f"  - Saved predictions and final pipeline for {data_type}.")

    # --- Classification report and confusion matrix on the hold-out test set ---
    try:
        y_test_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        print(f"  - Test Accuracy for {data_type}: {acc:.4f}")
        print(f"  - Classification Report for {data_type}:")
        print(classification_report(y_test, y_test_pred, target_names=le.classes_))

        # Confusion matrix (raw)
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"  - Confusion Matrix for {data_type} (rows=true, cols=pred):")
        print(cm)

        # Save confusion matrix as CSV with class labels
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}.csv')
        cm_df.to_csv(cm_path)
        print(f"  - Saved confusion matrix to {cm_path}")

        # Normalized confusion matrix (per-row / true-class)
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=(row_sums != 0))
        cmn_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
        cmn_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}_normalized.csv')
        cmn_df.to_csv(cmn_path)
        print(f"  - Saved normalized confusion matrix to {cmn_path}")
    except Exception as e:
        print(f"Warning: Could not compute classification report or confusion matrix for {data_type}: {e}")
