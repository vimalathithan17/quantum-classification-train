import pandas as pd
import os
import argparse
import optuna
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import both DR model types for experimentation
from qml_models import MulticlassQuantumClassifierDR, MulticlassQuantumClassifierDataReuploadingDR

def assemble_meta_data(preds_dirs, indicator_file, encoder_dir):
    """Loads and combines base learner predictions from multiple directories."""
    print(f"--- Assembling data from: {preds_dirs} ---")

    # Load the master label encoder
    try:
        encoder_path = os.path.join(encoder_dir, 'label_encoder.joblib')
        le = joblib.load(encoder_path)
        print(f"Master label encoder loaded from '{encoder_path}'")
    except FileNotFoundError:
        print(f"FATAL ERROR: Master label encoder not found in '{encoder_dir}'.")
        print("Please run the 'create_master_encoder.py' script first.")
        return None, None, None, None, None

    # Load indicator features and encode labels using the master encoder
    try:
        indicators = pd.read_parquet(indicator_file)
        indicators.set_index('case_id', inplace=True)
    except FileNotFoundError:
        print(f"Error: Indicator file not found at {indicator_file}")
        return None, None, None, None, None

    labels_categorical = indicators['class']
    labels = pd.Series(le.transform(labels_categorical), index=labels_categorical.index)
    indicators = indicators.drop(columns=['class'])

    oof_preds_list = []
    test_preds_list = []

    # Loop through each provided prediction directory
    for preds_dir in preds_dirs:
        print(f"  - Loading predictions from '{preds_dir}'...")
        oof_files = [f for f in os.listdir(preds_dir) if f.startswith('train_oof_preds_')]
        test_files = [f for f in os.listdir(preds_dir) if f.startswith('test_preds_')]
        
        for f in oof_files:
            oof_preds_list.append(pd.read_csv(os.path.join(preds_dir, f), index_col=0))
        for f in test_files:
            test_preds_list.append(pd.read_csv(os.path.join(preds_dir, f), index_col=0))
    
    if not oof_preds_list:
        print("Error: No out-of-fold prediction files found in the provided directories.")
        return None, None, None, None, None

    # Concatenate all found predictions
    X_meta_train_preds = pd.concat(oof_preds_list, axis=1)
    X_meta_test_preds = pd.concat(test_preds_list, axis=1)

    # Join with indicator features
    X_meta_train = X_meta_train_preds.join(indicators).dropna()
    X_meta_test = X_meta_test_preds.join(indicators).dropna()
    
    # Align labels with the final set of samples
    y_meta_train = labels.loc[X_meta_train.index]
    y_meta_test = labels.loc[X_meta_test.index]

    print(f"Meta-training data shape: {X_meta_train.shape}")
    print(f"Meta-test data shape: {X_meta_test.shape}")
    
    return X_meta_train, y_meta_train, X_meta_test, y_meta_test, le

def objective(trial, X_train, y_train, X_val, y_val, n_classes):
    """Defines one trial for tuning the meta-learner."""
    qml_model_type = trial.suggest_categorical('qml_model', ['standard', 'reuploading'])
    n_qubits = X_train.shape[1]
    n_layers = trial.suggest_int('n_layers', 1, 6)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    steps = trial.suggest_int('steps', 50, 150, step=25)
    
    model_params = {
        'n_qubits': n_qubits, 'n_layers': n_layers, 
        'learning_rate': learning_rate, 'steps': steps, 'n_classes': n_classes
    }
    
    if qml_model_type == 'standard':
        model = MulticlassQuantumClassifierDR(**model_params)
    else: # reuploading
        model = MulticlassQuantumClassifierDataReuploadingDR(**model_params)
    
    model.fit(X_train.values, y_train.values)
    predictions = model.predict(X_val.values)
    return accuracy_score(y_val.values, predictions)

def main():
    parser = argparse.ArgumentParser(description="Train or tune the QML meta-learner.")
    parser.add_argument('--preds_dir', nargs='+', required=True, help="One or more directories with base learner predictions.")
    parser.add_argument('--indicator_file', type=str, default='indicator_features.parquet', help="Path to the indicator features file.")
    parser.add_argument('--encoder_dir', type=str, default='master_label_encoder', help="Directory containing the master label_encoder.joblib")
    parser.add_argument('--tune', action='store_true', help="If set, run hyperparameter tuning.")
    parser.add_argument('--params_file', type=str, default='meta_learner_best_params.json', help="File to save/load best hyperparameters.")
    args = parser.parse_args()

    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le = assemble_meta_data(
        args.preds_dir, args.indicator_file, args.encoder_dir
    )
    if X_meta_train is None: return

    n_classes = len(le.classes_)

    if args.tune:
        print("\n--- Running Hyperparameter Tuning for Meta-Learner ---")
        X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
            X_meta_train, y_meta_train, test_size=0.3, random_state=42, stratify=y_meta_train
        )
        study = optuna.create_study(direction='maximize', study_name='qml_meta_learner_tuning')
        study.optimize(lambda t: objective(t, X_tune_train, y_tune_train, X_tune_val, y_tune_val, n_classes), n_trials=50)
        
        print("\n--- Tuning Complete ---")
        print("Best hyperparameters for meta-learner:", study.best_params)
        with open(args.params_file, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"Saved best parameters to '{args.params_file}'")

    else:
        print("\n--- Training Final Quantum Meta-Learner ---")
        best_params = {}
        try:
            with open(args.params_file, 'r') as f:
                best_params = json.load(f)
            print(f"Loaded best parameters from '{args.params_file}': {best_params}")
        except FileNotFoundError:
            print(f"Warning: Parameter file '{args.params_file}' not found. Using default parameters.")

        n_meta_features = X_meta_train.shape[1]
        
        model_params = {
            'n_qubits': n_meta_features, 'n_classes': n_classes,
            'n_layers': best_params.get('n_layers', 3),
            'learning_rate': best_params.get('learning_rate', 0.01),
            'steps': best_params.get('steps', 100)
        }
        
        if best_params.get('qml_model', 'standard') == 'standard':
            print("  - Using Standard QML Model.")
            meta_learner = MulticlassQuantumClassifierDR(**model_params)
        else:
            print("  - Using Data Re-uploading QML Model.")
            meta_learner = MulticlassQuantumClassifierDataReuploadingDR(**model_params)

        meta_learner.fit(X_meta_train.values, y_meta_train.values)
        
        final_predictions_encoded = meta_learner.predict(X_meta_test.values)
        
        print("\n--- Stacking Ensemble Results ---")
        print("Classification Report on Test Set:")
        print(classification_report(y_meta_test.values, final_predictions_encoded, target_names=le.classes_))
        
        # --- SAVE THE FINAL MODEL AND COLUMN ORDER ---
        joblib.dump(meta_learner, 'meta_learner_final.joblib')
        
        # Save the exact order of columns the meta-learner was trained on
        with open('meta_learner_columns.json', 'w') as f:
            json.dump(list(X_meta_train.columns), f)
            
        print("\nSaved final meta-learner model to 'meta_learner_final.joblib'")
        print("Saved meta-learner column order to 'meta_learner_columns.json'")
        

if __name__ == "__main__":
    main()
