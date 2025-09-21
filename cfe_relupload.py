import pandas as pd
import os
import joblib
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    print(f"Master label encoder loaded. Found {n_classes} classes: {list(le.classes_)}")
except FileNotFoundError:
    print(f"FATAL ERROR: Master label encoder not found at '{label_encoder_path}'.")
    print("Please run the 'create_master_encoder.py' script first.")
    exit()

# --- Main Training Loop ---
for data_type in DATA_TYPES_TO_TRAIN:
    # --- Find and Load Tuned Hyperparameters ---
    param_file_found = None
    search_pattern = f"_{data_type}_app2_"
    for filename in os.listdir(TUNING_RESULTS_DIR):
        if search_pattern in filename and "_reuploading" in filename:
            param_file_found = os.path.join(TUNING_RESULTS_DIR, filename)
            break
    
    if not param_file_found:
        print(f"\n--- No tuned parameter file found for {data_type} (Approach 2, Re-uploading). Skipping. ---")
        continue

    print(f"\n--- Training Base Learner for: {data_type} (Approach 2, Re-uploading) ---")
    with open(param_file_found, 'r') as f:
        config = json.load(f)

    # --- Load Data and Encode Labels ---
    file_path = os.path.join(SOURCE_DIR, f'data_{data_type}_.parquet')
    df = safe_load_parquet(file_path)
    X = df.drop(columns=[ID_COL, LABEL_COL])
    y_categorical = df[LABEL_COL]
    y = le.transform(y_categorical)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    y_train, y_test = pd.Series(y_train, index=X_train.index), pd.Series(y_test, index=X_test.index)

    n_features = config['n_qubits']

    if data_type == 'Meth':
        print("  - Using advanced LightGBM feature selection for Meth data...")
        # Use a more robust imputer for the selector
        imputer_for_fs = KNNImputer(n_neighbors=5)
        X_train_imputed = imputer_for_fs.fit_transform(X_train)
        
        # Use the powerful embedded method for final feature selection
        selector = SelectFromModel(
            lgb.LGBMClassifier(random_state=42), 
            max_features=n_features # Use the tuned number of features
        ).fit(X_train_imputed, y_train)
        
        final_selected_cols = X_train.columns[selector.get_support()]
        # The rest of the script will now use these intelligently selected columns
    else:
        # For all other data types, we will select features inside the CV loop as before
        final_selected_cols = None

    # --- Generate Out-of-Fold Predictions ---
    print("  - Generating out-of-fold predictions...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_train), n_classes))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"    - Processing Fold {fold + 1}/3...")
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        if final_selected_cols is not None: # We already selected features for Meth
            selected_cols = final_selected_cols
        else: # Use fast selection for other data types
            imputer_for_fs = SimpleImputer(strategy='median')
            X_train_fold_imputed = imputer_for_fs.fit_transform(X_train_fold)
            selector = SelectKBest(f_classif, k=n_features).fit(X_train_fold_imputed, y_train_fold)
            selected_cols = X_train_fold.columns[selector.get_support()]
        
        X_train_fold_selected = X_train_fold[selected_cols]
        X_val_fold_selected = X_val_fold[selected_cols]

        is_missing_train = X_train_fold_selected.isnull().astype(int).values
        X_train_filled = X_train_fold_selected.fillna(0.0).values
        is_missing_val = X_val_fold_selected.isnull().astype(int).values
        X_val_filled = X_val_fold_selected.fillna(0.0).values
        
        scaler = get_scaler(config.get('scaler', 'MinMax'))
        scaler.fit(X_train_filled)
        X_train_scaled = scaler.transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        
        model = ConditionalMulticlassQuantumClassifierDataReuploadingFS(
            n_qubits=n_features, n_layers=config['n_layers'], 
            steps=config['steps'], n_classes=n_classes
        )
        model.fit((X_train_scaled, is_missing_train), y_train_fold.values)
        val_preds = model.predict_proba((X_val_scaled, is_missing_val))
        oof_preds[val_idx] = val_preds

    oof_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(oof_preds, index=X_train.index, columns=oof_cols).to_csv(os.path.join(OUTPUT_DIR, f'train_oof_preds_{data_type}.csv'))
    print("  - Saved out-of-fold training predictions.")
    
    # --- Train Final Model on Full Training Data ---
    print("  - Training final model on full training data...")
    imputer_for_fs = SimpleImputer(strategy='median')
    X_train_imputed = imputer_for_fs.fit_transform(X_train)
    final_selector = SelectKBest(f_classif, k=n_features).fit(X_train_imputed, y_train)
    final_selected_cols = X_train.columns[final_selector.get_support()]

    X_train_selected = X_train[final_selected_cols]
    is_missing_train = X_train_selected.isnull().astype(int).values
    X_train_filled = X_train_selected.fillna(0.0).values
    final_scaler = get_scaler(config.get('scaler', 'MinMax'))
    X_train_scaled = final_scaler.fit_transform(X_train_filled)
    
    final_model = ConditionalMulticlassQuantumClassifierDataReuploadingFS(
        n_qubits=n_features, n_layers=config['n_layers'], 
        steps=config['steps'], n_classes=n_classes
    )
    final_model.fit((X_train_scaled, is_missing_train), y_train.values)

    # --- Generate Predictions on Test Set ---
    print("  - Generating predictions on the hold-out test set...")
    X_test_selected = X_test[final_selected_cols]
    is_missing_test = X_test_selected.isnull().astype(int).values
    X_test_filled = X_test_selected.fillna(0.0).values
    X_test_scaled = final_scaler.transform(X_test_filled)

    test_preds = final_model.predict_proba((X_test_scaled, is_missing_test))
    test_cols = [f"pred_{data_type}_{cls}" for cls in le.classes_]
    pd.DataFrame(test_preds, index=X_test.index, columns=test_cols).to_csv(os.path.join(OUTPUT_DIR, f'test_preds_{data_type}.csv'))
    print("  - Saved test predictions.")

    # --- Save all components for inference ---
    joblib.dump(final_selector, os.path.join(OUTPUT_DIR, f'selector_{data_type}.joblib'))
    joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, f'scaler_{data_type}.joblib'))
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, f'qml_model_{data_type}.joblib'))
    print(f"  - Saved final selector, scaler, and QML model for {data_type}.")

    # --- Classification report on the hold-out test set ---
    try:
        test_preds_labels = np.argmax(test_preds, axis=1)
        acc = accuracy_score(y_test, test_preds_labels)
        print(f"  - Test Accuracy for {data_type}: {acc:.4f}")
        print(f"  - Classification Report for {data_type}:")
        print(classification_report(y_test, test_preds_labels, target_names=le.classes_))

        # Confusion matrix (raw)
        cm = confusion_matrix(y_test, test_preds_labels)
        print(f"  - Confusion Matrix for {data_type} (rows=true, cols=pred):")
        print(cm)

        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}.csv')
        cm_df.to_csv(cm_path)
        print(f"  - Saved confusion matrix to {cm_path}")

        # Normalized confusion matrix
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=(row_sums != 0))
        cmn_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
        cmn_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{data_type}_normalized.csv')
        cmn_df.to_csv(cmn_path)
        print(f"  - Saved normalized confusion matrix to {cmn_path}")
    except Exception as e:
        print(f"Warning: Could not compute classification report or confusion matrix for {data_type}: {e}")
