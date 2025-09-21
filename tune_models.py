# Final Universal Tuning Script (`tune_models.py`)
import pandas as pd
import numpy as np
import optuna
import argparse
import os
import itertools
import json
import lightgbm as lgb
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from umap import UMAP

# Import the centralized logger
from logging_utils import log

# Ensure your corrected, multiclass 'qml_models.py' is in the same directory
from qml_models import (
    MulticlassQuantumClassifierDR, 
    MulticlassQuantumClassifierDataReuploadingDR,
    ConditionalMulticlassQuantumClassifierFS,
    ConditionalMulticlassQuantumClassifierDataReuploadingFS
)

# Directories configurable via environment
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
TUNING_RESULTS_DIR = os.environ.get('TUNING_RESULTS_DIR', 'tuning_results')
TUNING_JOURNAL_FILE = os.environ.get('TUNING_JOURNAL_FILE', 'tuning_journal.log')


def safe_load_parquet(file_path):
    """Loads a parquet file with increased thrift limits, returning None on failure."""
    if not os.path.exists(file_path):
        log.error(f"File not found at {file_path}")
        return None
    limit = 1 * 1024**3
    try:
        return pd.read_parquet(
            file_path,
            thrift_string_size_limit=limit,
            thrift_container_size_limit=limit
        )
    except Exception as e:
        log.error(f"Error loading {file_path}: {e}")
        return None

def get_scaler(scaler_name):
    """Returns a scaler object from a string name."""
    if scaler_name == 'MinMax': return MinMaxScaler()
    if scaler_name == 'Standard': return StandardScaler()
    if scaler_name == 'Robust': return RobustScaler()

def objective(trial, args, X, y, n_classes):
    """Defines one trial with Stratified K-Fold for a given pipeline configuration."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Log suggested parameters
    params = {
        'scaler': trial.suggest_categorical('scaler', ['MinMax', 'Standard', 'Robust']),
        'steps': trial.suggest_int('steps', 50, 100, step=25),
        'n_qubits': trial.suggest_int('n_qubits', n_classes, 12),
        'n_layers': trial.suggest_int('n_layers', 1, 5)
    }

    if args.approach == 1 and args.datatype == 'Meth':
        params['select_features'] = trial.suggest_int('select_features', 500, 2000)
        params['n_neighbors'] = trial.suggest_int('n_neighbors', 3, 9)

    log.info(f"Trial {trial.number} Parameters: {json.dumps(params, indent=2)}")

    scaler = get_scaler(params['scaler'])
    steps = params['steps']
    n_qubits = params['n_qubits']
    n_layers = params['n_layers']
    
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    if args.approach == 1:
        steps_list = []
        if args.datatype == 'Meth':
            select_features = params['select_features']
            n_neighbors = params['n_neighbors']
            selector = SelectFromModel(lgb.LGBMClassifier(random_state=42), max_features=select_features)
            steps_list.extend([('selector', selector), ('imputer', KNNImputer(n_neighbors=n_neighbors))])
        else:
            steps_list.append(('imputer', KNNImputer(n_neighbors=5)))
        
        steps_list.append(('scaler', scaler))

        if args.dim_reducer == 'pca':
            steps_list.append(('dim_reducer', PCA(n_components=n_qubits)))
        else:
            steps_list.append(('dim_reducer', UMAP(n_components=n_qubits, random_state=42)))

        if args.qml_model == 'standard':
            qml_model = MulticlassQuantumClassifierDR(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose)
        else: # reuploading
            qml_model = MulticlassQuantumClassifierDataReuploadingDR(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose)
            
        pipeline = Pipeline(steps_list + [('qml', qml_model)])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Starting training...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_val, y_val)
            scores.append(score)
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Completed with score {score:.4f}")

    elif args.approach == 2:
        if args.qml_model == 'standard':
            qml_model = ConditionalMulticlassQuantumClassifierFS(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose)
        else: # reuploading
            qml_model = ConditionalMulticlassQuantumClassifierDataReuploadingFS(n_qubits=n_qubits, n_layers=n_layers, steps=steps, n_classes=n_classes, verbose=args.verbose)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Starting training...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Perform fold-specific feature selection and preprocessing
            temp_imputer = SimpleImputer(strategy='median')
            X_train_imputed = temp_imputer.fit_transform(X_train)
            selector = SelectKBest(f_classif, k=n_qubits).fit(X_train_imputed, y_train)
            selected_cols = X_train.columns[selector.get_support()]
            
            X_train_selected = X_train[selected_cols]
            X_val_selected = X_val[selected_cols]

            # Prepare the data tuple (mask, fill, and scale) correctly
            is_missing_train = X_train_selected.isnull().astype(int).values
            X_train_filled = X_train_selected.fillna(0.0).values
            is_missing_val = X_val_selected.isnull().astype(int).values
            X_val_filled = X_val_selected.fillna(0.0).values
            
            scaler.fit(X_train_filled)
            X_train_scaled = scaler.transform(X_train_filled)
            X_val_scaled = scaler.transform(X_val_filled)

            qml_model.fit((X_train_scaled, is_missing_train), y_train.values)
            score = qml_model.score((X_val_scaled, is_missing_val), y_val.values)
            scores.append(score)
            log.info(f"Trial {trial.number}, Fold {fold+1}/{n_splits}: Completed with score {score:.4f}")

    average_accuracy = np.mean(scores)
    log.info(f"--- Trial {trial.number} Finished: Average Accuracy = {average_accuracy:.4f} ---")
    return average_accuracy

def main():
    parser = argparse.ArgumentParser(description="Universal QML tuning framework for multiclass problems.")
    parser.add_argument('--datatype', type=str, required=True, help="Data type (e.g., CNV, Meth)")
    parser.add_argument('--approach', type=int, required=True, choices=[1, 2], help="1: Classical+QML, 2: Conditional QML")
    parser.add_argument('--dim_reducer', type=str, default='pca', choices=['pca', 'umap'], help="For Approach 1: PCA or UMAP")
    parser.add_argument('--qml_model', type=str, default='standard', choices=['standard', 'reuploading'], help="QML circuit type")
    parser.add_argument('--n_trials', type=int, default=30, help="Number of Optuna trials for random search")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for QML model training steps.")
    args = parser.parse_args()

    log.info(f"Starting hyperparameter tuning with arguments: {args}")

    df = safe_load_parquet(os.path.join(SOURCE_DIR, f'data_{args.datatype}_.parquet'))
    if df is None: 
        log.error("Failed to load data, exiting.")
        return

    X = df.drop(columns=['case_id', 'class'])
    y_categorical = df['class']

    # Perform label encoding for the multiclass target
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_categorical), index=y_categorical.index)
    n_classes = len(le.classes_)
    log.info(f"Detected {n_classes} classes: {list(le.classes_)}")

    study_name = f'multiclass_qml_tuning_{args.datatype}_app{args.approach}_{args.dim_reducer}_{args.qml_model}'
    log.info(f"Using study name: {study_name}")
    log.info(f"Using journal file: {TUNING_JOURNAL_FILE}")

    storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=TUNING_JOURNAL_FILE))
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)
    
    study.optimize(lambda t: objective(t, args, X, y, n_classes), n_trials=args.n_trials)

    log.info("--- Hyperparameter Tuning Complete ---")
    log.info(f"Best hyperparameters found: {study.best_params}")
    
    os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)
    params_file = os.path.join(TUNING_RESULTS_DIR, f'best_params_{study_name}.json')
    with open(params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    log.info(f"Saved best parameters to '{params_file}'")

if __name__ == "__main__":
    main()
