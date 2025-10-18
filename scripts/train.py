#!/usr/bin/env python
"""
Unified training script for quantum classifiers with all enhancements.
Supports: checkpointing, resume, metrics logging, validation splits, stratified splits.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qml_models import MulticlassQuantumClassifierDR
from logging_utils import log


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


def main():
    parser = argparse.ArgumentParser(description="Unified training script for quantum classifiers")
    
    # Data arguments
    parser.add_argument('--data_file', type=str, required=True, help='Path to input parquet data file')
    parser.add_argument('--output_dir', type=str, default='trained_models', help='Output directory for models')
    
    # Model arguments
    parser.add_argument('--n_qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of ansatz layers')
    parser.add_argument('--hidden_size', type=int, default=16, help='Hidden size for readout head')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu'], help='Activation function')
    
    # Training arguments
    parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (currently not used)')
    parser.add_argument('--scaler', type=str, default='minmax', help='Scaler type: s/m/r')
    
    # Split arguments
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2 for 80/20 split)')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size from training set')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--resume_mode', type=str, default='auto', 
                       choices=['auto', 'latest', 'best', 'none'], 
                       help='Resume mode: auto/latest/best/none')
    parser.add_argument('--checkpoint_frequency', type=int, default=10, help='Checkpoint save frequency')
    
    # Metric arguments
    parser.add_argument('--metric', type=str, default='f1_weighted', 
                       help='Selection metric for best model')
    
    # Other arguments
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set checkpoint dir if not provided
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log.info("="*80)
    log.info("Unified Quantum Classifier Training Script")
    log.info("="*80)
    log.info(f"Data file: {args.data_file}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Checkpoint directory: {args.checkpoint_dir}")
    log.info(f"Resume mode: {args.resume_mode}")
    log.info(f"Selection metric: {args.metric}")
    
    # Load data
    log.info("Loading data...")
    df = pd.read_parquet(args.data_file)
    
    # Separate features and labels
    if 'class' not in df.columns:
        log.error("Data must contain 'class' column")
        return 1
    
    # Drop ID column if present
    if 'case_id' in df.columns:
        df = df.drop(columns=['case_id'])
    
    X = df.drop(columns=['class']).values
    y_raw = df['class'].values
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    
    log.info(f"Data shape: {X.shape}")
    log.info(f"Number of classes: {n_classes}")
    log.info(f"Classes: {list(le.classes_)}")
    
    # Stratified train/test split (80/20 default)
    log.info(f"Splitting data: {1-args.test_size:.0%} train, {args.test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    
    # Optional validation split from training set
    if args.val_size > 0:
        log.info(f"Creating validation set: {args.val_size:.0%} of training set")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=args.val_size, stratify=y_train, random_state=args.random_state
        )
    else:
        X_val, y_val = None, None
    
    # Scale features
    log.info(f"Scaling features with {args.scaler} scaler...")
    scaler = get_scaler(args.scaler)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if X_val is not None:
        X_val = scaler.transform(X_val)
    
    # Truncate or pad features to match n_qubits
    if X_train.shape[1] > args.n_qubits:
        log.info(f"Truncating features from {X_train.shape[1]} to {args.n_qubits}")
        X_train = X_train[:, :args.n_qubits]
        X_test = X_test[:, :args.n_qubits]
        if X_val is not None:
            X_val = X_val[:, :args.n_qubits]
    elif X_train.shape[1] < args.n_qubits:
        log.info(f"Padding features from {X_train.shape[1]} to {args.n_qubits}")
        pad_width = args.n_qubits - X_train.shape[1]
        X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant')
        if X_val is not None:
            X_val = np.pad(X_val, ((0, 0), (0, pad_width)), mode='constant')
    
    # Create classifier
    log.info("Creating quantum classifier...")
    clf = MulticlassQuantumClassifierDR(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=n_classes,
        learning_rate=args.learning_rate,
        steps=args.steps,
        verbose=args.verbose,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        hidden_size=args.hidden_size,
        activation=args.activation,
        resume_mode=args.resume_mode
    )
    
    # Train
    log.info("Training classifier...")
    clf.fit(X_train, y_train)
    
    # Evaluate on training set
    log.info("\nEvaluating on training set...")
    y_train_pred = clf.predict(X_train)
    train_acc = np.mean(y_train_pred == y_train)
    log.info(f"Training accuracy: {train_acc:.4f}")
    
    # Evaluate on validation set if present
    if X_val is not None:
        log.info("\nEvaluating on validation set...")
        y_val_pred = clf.predict(X_val)
        val_acc = np.mean(y_val_pred == y_val)
        log.info(f"Validation accuracy: {val_acc:.4f}")
        log.info("\nValidation Classification Report:")
        print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    
    # Evaluate on test set
    log.info("\nEvaluating on test set...")
    y_test_pred = clf.predict(X_test)
    test_acc = np.mean(y_test_pred == y_test)
    log.info(f"Test accuracy: {test_acc:.4f}")
    log.info("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    
    # Save model and artifacts
    log.info(f"\nSaving model to {args.output_dir}...")
    model_path = os.path.join(args.output_dir, 'model.joblib')
    scaler_path = os.path.join(args.output_dir, 'scaler.joblib')
    encoder_path = os.path.join(args.output_dir, 'label_encoder.joblib')
    
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    
    log.info(f"Model saved to: {model_path}")
    log.info(f"Scaler saved to: {scaler_path}")
    log.info(f"Label encoder saved to: {encoder_path}")
    
    # Save predictions
    test_preds_df = pd.DataFrame({
        'y_true': le.inverse_transform(y_test),
        'y_pred': le.inverse_transform(y_test_pred)
    })
    test_preds_path = os.path.join(args.output_dir, 'test_predictions.csv')
    test_preds_df.to_csv(test_preds_path, index=False)
    log.info(f"Test predictions saved to: {test_preds_path}")
    
    log.info("\nTraining complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
