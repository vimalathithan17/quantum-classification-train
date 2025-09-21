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
            neutral_proba = np.full((1, n_classes), 1 / n_classes)
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
                scaler = joblib.load(os.path.join(model_dir, f'scaler_{data_type}.joblib'))
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
    try:
        meta_features_df = meta_features_df[meta_columns_order]
    except KeyError as e:
        log.critical(f"A required column is missing for the meta-learner. Details: {e}")
        log.critical(f"Required columns: {meta_columns_order}")
        log.critical(f"Available columns: {meta_features_df.columns.tolist()}")
        return None
    
    final_prediction_encoded = meta_learner.predict(meta_features_df.values)
    final_prediction_label = le.inverse_transform(final_prediction_encoded)
    
    return final_prediction_label[0]

def main():
    parser = argparse.ArgumentParser(description="Run inference on a new patient's data.")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing all trained models and components.")
    parser.add_argument('--patient_data_dir', type=str, required=True, help="Directory containing the new patient's parquet files.")
    args = parser.parse_args()

    log.info(f"Starting inference for patient data in '{args.patient_data_dir}' using models from '{args.model_dir}'.")
    
    prediction = make_single_prediction(args.model_dir, args.patient_data_dir)
    
    if prediction:
        log.info(f"\n--- Final Prediction ---")
        log.info(f"The predicted class for the patient is: {prediction}")
        log.info("------------------------")

if __name__ == "__main__":
    main()
