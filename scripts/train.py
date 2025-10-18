#!/usr/bin/env python
"""
Basic training script for quantum classifiers with enhanced features.

This script provides:
- Stratified 80/20 train/test split
- Resume functionality (auto/latest/best)
- Checkpoint management
- Metrics logging and visualization
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import log
from qml_models import MulticlassQuantumClassifierDR


def get_scaler(scaler_name):
    """Returns a scaler object from a string name."""
    if not scaler_name:
        return StandardScaler()
    s = scaler_name.strip().lower()
    if s in ('m', 'minmax', 'min_max', 'minmaxscaler'):
        return MinMaxScaler()
    if s in ('s', 'standard', 'standardscaler'):
        return StandardScaler()
    if s in ('r', 'robust', 'robustscaler'):
        return RobustScaler()
    return StandardScaler()


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


def main():
    parser = argparse.ArgumentParser(description="Train quantum classifier with enhanced features")
    parser.add_argument('--data_path', type=str, required=True, help="Path to parquet data file")
    parser.add_argument('--output_dir', type=str, default='./training_output', 
                       help="Output directory for models and checkpoints")
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help="Checkpoint directory (default: output_dir/checkpoints)")
    parser.add_argument('--resume_mode', type=str, default=None, 
                       choices=['auto', 'latest', 'best', None],
                       help="Resume mode: auto (default), latest, best, or None for fresh start")
    parser.add_argument('--n_qubits', type=int, default=8, help="Number of qubits")
    parser.add_argument('--n_layers', type=int, default=3, help="Number of layers")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--steps', type=int, default=100, help="Training steps")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size (not used yet)")
    parser.add_argument('--test_size', type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument('--scaler', type=str, default='standard', 
                       help="Scaler: 's' (Standard), 'm' (MinMax), 'r' (Robust)")
    parser.add_argument('--pca_components', type=int, default=None,
                       help="Number of PCA components (default: n_qubits)")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")
    parser.add_argument('--checkpoint_frequency', type=int, default=10, 
                       help="Save checkpoint every N steps")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log.info(f"Training quantum classifier")
    log.info(f"Data: {args.data_path}")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Checkpoints: {checkpoint_dir}")
    
    # Load data
    log.info("Loading data...")
    df = safe_load_parquet(args.data_path)
    if df is None:
        log.error("Failed to load data")
        return 1
    
    # Separate features and labels
    if 'class' not in df.columns:
        log.error("Data must have a 'class' column")
        return 1
    
    y = df['class'].values
    
    # Remove ID and class columns
    feature_cols = [col for col in df.columns if col not in ['case_id', 'class']]
    X = df[feature_cols].values
    
    log.info(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    log.info(f"Number of classes: {n_classes}")
    
    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, 
        stratify=y_encoded, random_state=args.random_seed
    )
    
    log.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Preprocessing
    log.info("Preprocessing data...")
    scaler = get_scaler(args.scaler)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA dimensionality reduction
    n_components = args.pca_components or args.n_qubits
    log.info(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=args.random_seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Pad if needed
    if X_train_pca.shape[1] < args.n_qubits:
        padding = np.zeros((X_train_pca.shape[0], args.n_qubits - X_train_pca.shape[1]))
        X_train_pca = np.hstack([X_train_pca, padding])
        padding_test = np.zeros((X_test_pca.shape[0], args.n_qubits - X_test_pca.shape[1]))
        X_test_pca = np.hstack([X_test_pca, padding_test])
    
    log.info(f"PCA output shape: {X_train_pca.shape}")
    
    # Initialize model
    log.info("Initializing quantum classifier...")
    model = MulticlassQuantumClassifierDR(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=n_classes,
        learning_rate=args.learning_rate,
        steps=args.steps,
        verbose=args.verbose,
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency
    )
    
    # Train model (with resume if specified)
    log.info(f"Training model (resume mode: {args.resume_mode})...")
    model.fit(X_train_pca, y_train, resume=args.resume_mode)
    
    # Evaluate on test set
    log.info("Evaluating on test set...")
    y_pred = model.predict(X_test_pca)
    accuracy = np.mean(y_pred == y_test)
    log.info(f"Test accuracy: {accuracy:.4f}")
    
    # Save model and preprocessing artifacts
    log.info("Saving model and artifacts...")
    model_path = os.path.join(args.output_dir, 'qml_model.joblib')
    joblib.dump(model, model_path)
    
    scaler_path = os.path.join(args.output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    pca_path = os.path.join(args.output_dir, 'pca.joblib')
    joblib.dump(pca, pca_path)
    
    le_path = os.path.join(args.output_dir, 'label_encoder.joblib')
    joblib.dump(le, le_path)
    
    # Save test predictions
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'correct': y_test == y_pred
    })
    results_path = os.path.join(args.output_dir, 'test_predictions.csv')
    results_df.to_csv(results_path, index=False)
    
    log.info(f"Training complete! Model saved to {model_path}")
    log.info(f"Test predictions saved to {results_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
