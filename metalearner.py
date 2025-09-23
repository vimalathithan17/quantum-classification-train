import pandas as pd
import os
import argparse
import optuna
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

# Import the centralized logger
from logging_utils import log

# Import both DR model types for experimentation
from qml_models import MulticlassQuantumClassifierDR, MulticlassQuantumClassifierDataReuploadingDR

# Environment-configurable directories
ENCODER_DIR = os.environ.get('ENCODER_DIR', 'master_label_encoder')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')
TUNING_JOURNAL_FILE = os.environ.get('TUNING_JOURNAL_FILE', 'tuning_journal.log')


def get_scaler(scaler_name):
    """Returns a scaler object from a string name."""
    if scaler_name == 'MinMax': return MinMaxScaler()
    if scaler_name == 'Standard': return StandardScaler()
    if scaler_name == 'Robust': return RobustScaler()


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

            for f in oof_files:
                oof_preds_list.append(pd.read_csv(os.path.join(preds_dir, f), index_col=0))
            for f in test_files:
                test_preds_list.append(pd.read_csv(os.path.join(preds_dir, f), index_col=0))
        except FileNotFoundError:
            log.error(f"Prediction directory not found: '{preds_dir}'. Skipping.")
            continue
    
    if not oof_preds_list:
        log.error("No out-of-fold prediction files found in any of the provided directories.")
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

    log.info(f"Meta-training data shape: {X_meta_train.shape}")
    log.info(f"Meta-test data shape: {X_meta_test.shape}")
    
    return X_meta_train, y_meta_train, X_meta_test, y_meta_test, le

def objective(trial, X_train, y_train, X_val, y_val, n_classes, args, scaler_options):
    """Defines one trial for tuning the meta-learner."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Log suggested parameters
    params = {
        'scaler': trial.suggest_categorical('scaler', scaler_options),
        'qml_model': trial.suggest_categorical('qml_model', ['standard', 'reuploading']),
        'n_layers': trial.suggest_int('n_layers', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    }
    params['steps'] = 100  # Fixed number of steps for tuning
    log.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=2)}")

    # Create and fit the scaler
    scaler = get_scaler(params['scaler'])
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model_params = {
        'n_qubits': X_train.shape[1], 
        'n_layers': params['n_layers'], 
        'learning_rate': params['learning_rate'], 
        'steps': params['steps'], 
        'n_classes': n_classes,
        'verbose': args.verbose
    }
    
    if params['qml_model'] == 'standard':
        model = MulticlassQuantumClassifierDR(**model_params)
    else: # reuploading
        model = MulticlassQuantumClassifierDataReuploadingDR(**model_params)
    
    log.info(f"Trial {trial.number}: Training {params['qml_model']} model...")
    model.fit(X_train_scaled, y_train.values)
    
    log.info(f"Trial {trial.number}: Evaluating...")
    predictions = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val.values, predictions)
    
    log.info(f"--- Trial {trial.number} Finished: Accuracy = {accuracy:.4f} ---")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train or tune the QML meta-learner.")
    parser.add_argument('--preds_dir', nargs='+', required=True, help="One or more directories with base learner predictions.")
    parser.add_argument('--indicator_file', type=str, required=True, help="Path to the parquet file with indicator features and labels.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune'], help="Operation mode: 'train' a final model or 'tune' hyperparameters.")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials for tuning.")
    parser.add_argument('--override_steps', type=int, default=None, help="Override the number of training steps from the tuned parameters.")
    parser.add_argument('--scalers', type=str, default='smr', help="String indicating which scalers to try (s: Standard, m: MinMax, r: Robust). E.g., 'sm' for Standard and MinMax.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for QML model training steps.")
    args = parser.parse_args()

    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le = assemble_meta_data(args.preds_dir, args.indicator_file)
    if X_meta_train is None:
        log.critical("Failed to assemble meta-dataset. Exiting.")
        return

    n_classes = len(le.classes_)
    log.info(f"Meta-learner will be trained on {n_classes} classes.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.mode == 'tune':
        log.info(f"--- Starting Hyperparameter Tuning for Meta-Learner ({args.n_trials} trials) ---")
        
        # --- Scaler selection logic ---
        scaler_map = {'s': 'Standard', 'm': 'MinMax', 'r': 'Robust'}
        scaler_options = [scaler_map[char] for char in args.scalers if char in scaler_map]
        if not scaler_options:
            log.error("No valid scalers specified. Use 's', 'm', or 'r'. Exiting.")
            return
        log.info(f"Using scalers: {scaler_options}")

        # Split training data for validation during tuning
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_meta_train, y_meta_train, test_size=0.25, random_state=42, stratify=y_meta_train
        )
        
        study_name = 'qml_metalearner_tuning'
        storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=TUNING_JOURNAL_FILE))
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
        
        study.optimize(lambda t: objective(t, X_train_split, y_train_split, X_val_split, y_val_split, n_classes, args, scaler_options), n_trials=args.n_trials)

        log.info("--- Tuning Complete ---")
        log.info(f"Best hyperparameters found: {study.best_params}")
        
        # Save best parameters
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
            params = {'scaler': 'Standard', 'qml_model': 'reuploading', 'n_layers': 3, 'learning_rate': 0.05, 'steps': 100}

        # Override steps if provided via command line
        if args.override_steps:
            params['steps'] = args.override_steps
            log.info(f"Overriding training steps with: {args.override_steps}")

        # Create and fit the scaler on the full training data
        scaler = get_scaler(params.get('scaler', 'Standard'))
        log.info(f"Using scaler: {params.get('scaler', 'Standard')}")
        X_meta_train_scaled = scaler.fit_transform(X_meta_train)

        # Prepare model with loaded or default parameters
        model_params = {
            'n_qubits': X_meta_train.shape[1],
            'n_layers': params['n_layers'],
            'learning_rate': params['learning_rate'],
            'steps': params['steps'],
            'n_classes': n_classes,
            'verbose': args.verbose
        }
        
        if params['qml_model'] == 'standard':
            final_model = MulticlassQuantumClassifierDR(**model_params)
        else:
            final_model = MulticlassQuantumClassifierDataReuploadingDR(**model_params)

        log.info(f"Training final {params['qml_model']} model with parameters: {json.dumps(model_params, indent=2)}")
        final_model.fit(X_meta_train_scaled, y_meta_train.values)
        
        # Save the trained model and scaler
        model_path = os.path.join(OUTPUT_DIR, 'metalearner_model.joblib')
        joblib.dump(final_model, model_path)
        log.info(f"Final meta-learner model saved to '{model_path}'")
        
        scaler_path = os.path.join(OUTPUT_DIR, 'metalearner_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        log.info(f"Final meta-learner scaler saved to '{scaler_path}'")

        # Evaluate and save final predictions
        log.info("--- Evaluating on Test Set ---")
        X_meta_test_scaled = scaler.transform(X_meta_test)
        test_predictions = final_model.predict(X_meta_test_scaled)
        test_accuracy = accuracy_score(y_meta_test.values, test_predictions)
        log.info(f"Final Test Accuracy: {test_accuracy:.4f}")

        # Generate and print classification report
        report = classification_report(y_meta_test.values, test_predictions, target_names=le.classes_)
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
