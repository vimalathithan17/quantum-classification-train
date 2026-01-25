# inference.py

import pandas as pd
import os
import joblib
import json
import numpy as np
import argparse

# Import the centralized logger
from logging_utils import log

# --- Configuration ---
ID_COL = 'case_id'
LABEL_COL = 'class'
DATA_TYPES = ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']

def safe_load_parquet(file_path):
    """Safely loads a parquet file, returning None if it doesn't exist."""
    if not os.path.exists(file_path):
        log.warning(f"Data file not found at {file_path}. Treating as entirely missing.")
        return None
    limit = 1 * 1024**3
    try:
        return pd.read_parquet(file_path, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
    except Exception as e:
        log.error(f"Error loading {file_path}: {e}")
        return None

def make_single_prediction(model_dir, new_patient_data_dir):
    """
    Makes a final prediction for a new patient using a curated directory of mixed-approach models.
    """
    # --- Load Global Components ---
    try:
        le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        meta_learner = joblib.load(os.path.join(model_dir, 'meta_learner_final.joblib'))
        with open(os.path.join(model_dir, 'meta_learner_columns.json'), 'r') as f:
            meta_columns_order = json.load(f)
        n_classes = len(le.classes_)
        log.info("Successfully loaded global components (encoder, meta-learner, column order).")
    except FileNotFoundError as e:
        log.critical(f"A required global file is missing from '{model_dir}'. Details: {e}")
        return None

    base_predictions_list = []
    is_missing_flags = {}

    log.info("--- Generating predictions from base learners... ---")
    for data_type in DATA_TYPES:
        log.info(f"  - Processing {data_type}...")
        patient_df = safe_load_parquet(os.path.join(new_patient_data_dir, f'data_{data_type}_.parquet'))
        
        # --- Check for missing data first ---
        if patient_df is None:
            is_missing_flags[f'is_missing_{data_type}'] = 1
            # Strategy: zeros by default for gated/meta models; override via env
            strategy = os.environ.get('INFERENCE_NEUTRAL_PROBAS', 'zeros').strip().lower()
            if strategy == 'uniform':
                neutral_proba = np.full((1, n_classes), 1 / n_classes)
            else:
                neutral_proba = np.zeros((1, n_classes))
            pred_df = pd.DataFrame(neutral_proba, columns=[f"pred_{data_type}_{cls}" for cls in le.classes_])
        else:
            is_missing_flags[f'is_missing_{data_type}'] = 0
            X_new = patient_df.drop(columns=[ID_COL, LABEL_COL], errors='ignore')
            
            # --- Auto-detect the model type (Approach 1 vs Approach 2) ---
            pipeline_path = os.path.join(model_dir, f'pipeline_{data_type}.joblib')
            # For Approach 2, we check for the selected features file.
            features_path = os.path.join(model_dir, f'selected_features_{data_type}.joblib')

            if os.path.exists(pipeline_path):
                # --- Approach 1 Logic (Pipeline-based) ---
                log.info(f"    - Found Approach 1 pipeline for {data_type}.")
                pipeline = joblib.load(pipeline_path)
                prediction_proba = pipeline.predict_proba(X_new)
            elif os.path.exists(features_path):
                # --- Approach 2 Logic (Component-based) ---
                log.info(f"    - Found Approach 2 components for {data_type}.")
                # Scaler may be None (sentinel) for conditional models which do
                # not require scaling. Load if present; treat missing or None
                # as a no-op (skip scaling).
                scaler_path = os.path.join(model_dir, f'scaler_{data_type}.joblib')
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                else:
                    scaler = None
                qml_model = joblib.load(os.path.join(model_dir, f'qml_model_{data_type}.joblib'))
                selected_cols = joblib.load(features_path)
                
                # Ensure all selected columns are present in the new data
                if not all(col in X_new.columns for col in selected_cols):
                    log.critical(f"New data for {data_type} is missing some columns required by the model.")
                    missing_in_df = list(set(selected_cols) - set(X_new.columns))
                    log.critical(f"Columns missing in input data: {missing_in_df}")
                    return None

                X_new_selected = X_new[selected_cols]
                
                is_missing_mask = X_new_selected.isnull().astype(int).values
                X_filled = X_new_selected.fillna(0.0).values
                # If scaler is None (or missing), do not scale — pass filled arrays
                # directly. Otherwise, apply transform.
                if scaler is None:
                    log.info(f"No scaler found for {data_type} (scaler is None). Skipping scaling for conditional model input.")
                    X_scaled = X_filled
                else:
                    X_scaled = scaler.transform(X_filled)

                prediction_proba = qml_model.predict_proba((X_scaled, is_missing_mask))
            else:
                log.critical(f"No model files found for {data_type} in '{model_dir}'. Cannot proceed.")
                return None
            
            pred_df = pd.DataFrame(prediction_proba, columns=[f"pred_{data_type}_{cls}" for cls in le.classes_])

        base_predictions_list.append(pred_df)

    # --- Assemble Meta-Features ---
    meta_features_df = pd.concat(base_predictions_list, axis=1)
    for flag, value in is_missing_flags.items():
        meta_features_df[flag] = value

    # --- Make the Final Prediction ---
    log.info("--- Making final prediction with meta-learner... ---")
    # Ensure columns are in the exact order the meta-learner was trained on
    # Reindex to required columns; fill missing features with 0.0 to avoid failure
    missing_cols = [c for c in meta_columns_order if c not in meta_features_df.columns]
    if missing_cols:
        log.warning(f"Meta features missing expected columns; filling with zeros: {missing_cols}")
    meta_features_df = meta_features_df.reindex(columns=meta_columns_order, fill_value=0.0)
    
    # Separate base prediction columns from indicator columns for the gated meta-learner.
    # Indicators follow pattern 'is_missing_{datatype}'; base predictions are all others.
    indicator_cols = [c for c in meta_columns_order if c.startswith('is_missing_')]
    base_cols = [c for c in meta_columns_order if c not in indicator_cols]
    
    X_base = meta_features_df[base_cols].values
    
    # Build a mask matching the shape of X_base. For each base prediction column,
    # find the corresponding indicator (inverted: indicator=1 -> missing -> mask=0).
    # Default to 1.0 (present) if no indicator found.
    mask_columns = []
    indicator_map = {ic.replace('is_missing_', '').lower(): ic for ic in indicator_cols}
    for col in base_cols:
        # Extract datatype from pattern 'pred_{datatype}_{class...}'
        parts = col.split('_')
        if len(parts) >= 2:
            datatype = parts[1].lower()
        else:
            datatype = col.lower()
        
        ind_col = indicator_map.get(datatype)
        if ind_col is not None and ind_col in meta_features_df.columns:
            # is_missing=1 means data was missing -> mask=0 (exclude)
            # is_missing=0 means data was present -> mask=1 (include)
            mask_columns.append(1.0 - meta_features_df[ind_col].values)
        else:
            # Default to present (include)
            mask_columns.append(np.ones(len(meta_features_df)))
    
    X_mask = np.column_stack(mask_columns) if mask_columns else np.ones_like(X_base)
    
    log.info(f"Meta-learner input: X_base shape={X_base.shape}, X_mask shape={X_mask.shape}")
    
    final_prediction_encoded = meta_learner.predict((X_base, X_mask))
    final_prediction_label = le.inverse_transform(final_prediction_encoded)
    
    return final_prediction_label[0]

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a new patient's multimodal data using trained QML ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python inference.py --model_dir trained_models/ --patient_data_dir patient_001/
  
Description:
  Loads trained QML base learners and meta-learner to predict patient class.
  Expects model_dir to contain:
    - Base learner models (*.pkl)
    - Meta-learner (metalearner.pkl)
    - Label encoder (master_label_encoder.pkl)
  
  Expects patient_data_dir to contain parquet files for all modalities:
    - CNV_app1_pca.parquet, CNV_app2_pca.parquet
    - clinical.parquet
    - transcriptomics_pca.parquet
        """)
    
    # Required arguments
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--model_dir', type=str, required=True,
                              help='Directory with trained models and label encoder')
    required_args.add_argument('--patient_data_dir', type=str, required=True,
                              help="Directory with patient's multimodal parquet files")
    
    args = parser.parse_args()

    log.info(f"Starting inference for patient data in '{args.patient_data_dir}' using models from '{args.model_dir}'.")
    
    prediction = make_single_prediction(args.model_dir, args.patient_data_dir)
    
    if prediction:
        log.info("\n--- Final Prediction ---")
        log.info(f"The predicted class for the patient is: {prediction}")
        log.info("------------------------")

if __name__ == "__main__":
    main()
