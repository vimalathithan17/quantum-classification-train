# inference.py

import pandas as pd
import os
import joblib
import json
import numpy as np
import argparse

# --- Configuration ---
ID_COL = 'case_id'
LABEL_COL = 'class'
DATA_TYPES = ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']

def safe_load_parquet(file_path):
    """Safely loads a parquet file, returning None if it doesn't exist."""
    if not os.path.exists(file_path):
        print(f"  - WARNING: Data file not found at {file_path}. Treating as entirely missing.")
        return None
    limit = 1 * 1024**3
    return pd.read_parquet(file_path, thrift_string_size_limit=limit, thrift_container_size_limit=limit)

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
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required global file is missing from '{model_dir}'. Details: {e}")
        return None

    base_predictions_list = []
    is_missing_flags = {}

    print("--- Generating predictions from base learners... ---")
    for data_type in DATA_TYPES:
        print(f"  - Processing {data_type}...")
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
            selector_path = os.path.join(model_dir, f'selector_{data_type}.joblib')

            if os.path.exists(pipeline_path):
                # --- Approach 1 Logic ---
                # print(f"    - Found Approach 1 pipeline.")
                pipeline = joblib.load(pipeline_path)
                prediction_proba = pipeline.predict_proba(X_new)
            elif os.path.exists(selector_path):
                # --- Approach 2 Logic ---
                # print(f"    - Found Approach 2 components.")
                scaler = joblib.load(os.path.join(model_dir, f'scaler_{data_type}.joblib'))
                qml_model = joblib.load(os.path.join(model_dir, f'qml_model_{data_type}.joblib'))
                selector = joblib.load(selector_path)
                
                selected_cols = X_new.columns[selector.get_support()]
                X_new_selected = X_new[selected_cols]
                
                is_missing_mask = X_new_selected.isnull().astype(int).values
                X_filled = X_new_selected.fillna(0.0).values
                X_scaled = scaler.transform(X_filled)
                
                prediction_proba = qml_model.predict_proba((X_scaled, is_missing_mask))
            else:
                print(f"    - FATAL ERROR: No model files found for {data_type} in '{model_dir}'.")
                return None
            
            pred_df = pd.DataFrame(prediction_proba, columns=[f"pred_{data_type}_{cls}" for cls in le.classes_])

        base_predictions_list.append(pred_df)

    # --- Assemble Meta-Features ---
    meta_features_df = pd.concat(base_predictions_list, axis=1)
    for flag, value in is_missing_flags.items():
        meta_features_df[flag] = value

    # --- Make the Final Prediction ---
    print("\n--- Making final prediction with meta-learner... ---")
    # Ensure columns are in the exact order the meta-learner was trained on
    meta_features_df = meta_features_df[meta_columns_order]
    
    final_prediction_encoded = meta_learner.predict(meta_features_df.values)
    final_prediction_label = le.inverse_transform(final_prediction_encoded)
    
    return final_prediction_label[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified inference script for the QML stacking ensemble.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the curated final model directory.")
    parser.add_argument('--patient_data_dir', type=str, required=True, help="Path to the directory with the new patient's data files.")
    args = parser.parse_args()

    prediction = make_single_prediction(args.model_dir, args.patient_data_dir)
    
    if prediction is not None:
        print("\n--- INFERENCE COMPLETE ---")
        print(f"Final Predicted Class: {prediction}")
