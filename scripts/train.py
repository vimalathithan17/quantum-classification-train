#!/usr/bin/env python
"""
Training script for quantum classification models with checkpointing and metrics logging.

Supports training any classifier with:
- 80/20 stratified train/test split by default
- Custom Adam optimizer with state serialization
- Resume from checkpoints ('auto', 'latest', 'best')
- Per-epoch metrics logging (accuracy, precision, recall, F1, specificity, confusion matrix)
- CSV and PNG outputs for metrics
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import log
from qml_models import (
    MulticlassQuantumClassifierDR,
    MulticlassQuantumClassifierDataReuploadingDR,
    ConditionalMulticlassQuantumClassifierFS,
    ConditionalMulticlassQuantumClassifierDataReuploadingFS
)


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


def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load data from parquet file and perform stratified train/test split.
    
    Args:
        data_path: Path to parquet file
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    log.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_parquet(data_path)
    
    # Separate features and labels
    if 'case_id' in df.columns:
        df = df.drop(columns=['case_id'])
    
    X = df.drop(columns=['class'])
    y_categorical = df['class']
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_categorical)
    
    log.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    log.info(f"Classes: {list(le.classes_)}")
    
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    log.info(f"Train set: {X_train.shape[0]} samples")
    log.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, le


def create_classifier(classifier_type, n_qubits, n_layers, n_classes, learning_rate, 
                     steps, hidden_size, checkpoint_dir, verbose, use_classical_readout=True):
    """
    Factory function to create a classifier instance.
    
    Args:
        classifier_type: Type of classifier ('standard', 'reuploading', 'conditional', 'conditional_reuploading')
        n_qubits: Number of qubits
        n_layers: Number of circuit layers
        n_classes: Number of classes
        learning_rate: Learning rate
        steps: Number of training steps
        hidden_size: Hidden layer size for classical readout
        checkpoint_dir: Directory for checkpoints
        verbose: Verbose logging flag
        use_classical_readout: Whether to use classical readout layer
    
    Returns:
        Classifier instance
    """
    common_params = {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'n_classes': n_classes,
        'learning_rate': learning_rate,
        'steps': steps,
        'verbose': verbose,
        'checkpoint_dir': checkpoint_dir,
        'checkpoint_frequency': 10,
        'keep_last_n': 3
    }
    
    # Add classical readout params for DR models
    if classifier_type in ['standard', 'reuploading']:
        common_params['hidden_size'] = hidden_size
        common_params['use_classical_readout'] = use_classical_readout
    
    if classifier_type == 'standard':
        return MulticlassQuantumClassifierDR(**common_params)
    elif classifier_type == 'reuploading':
        return MulticlassQuantumClassifierDataReuploadingDR(**common_params)
    elif classifier_type == 'conditional':
        return ConditionalMulticlassQuantumClassifierFS(**common_params)
    elif classifier_type == 'conditional_reuploading':
        return ConditionalMulticlassQuantumClassifierDataReuploadingFS(**common_params)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Train quantum classification model with checkpointing and metrics logging"
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to parquet data file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for test set (default: 0.2)')
    
    # Classifier arguments
    parser.add_argument('--classifier', type=str, default='standard',
                       choices=['standard', 'reuploading', 'conditional', 'conditional_reuploading'],
                       help='Type of classifier to train')
    parser.add_argument('--n_qubits', type=int, default=8,
                       help='Number of qubits (default: 8)')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of circuit layers (default: 3)')
    parser.add_argument('--hidden_size', type=int, default=16,
                       help='Hidden layer size for classical readout (default: 16)')
    parser.add_argument('--use_classical_readout', action='store_true', default=True,
                       help='Use classical MLP readout layer (default: True)')
    parser.add_argument('--no_classical_readout', action='store_false', dest='use_classical_readout',
                       help='Disable classical readout layer')
    
    # Training arguments
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01,
                       dest='learning_rate', help='Learning rate (default: 0.01)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of training steps (default: 100)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (default: None, full batch)')
    
    # Preprocessing arguments
    parser.add_argument('--scaler', type=str, default='minmax',
                       choices=['minmax', 'standard', 'robust', 'm', 's', 'r'],
                       help='Scaler type (default: minmax)')
    
    # Checkpoint and resume arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume_mode', type=str, default=None,
                       choices=['auto', 'latest', 'best'],
                       help='Resume mode: auto (latest if exists), latest, best, or None')
    parser.add_argument('--selection_metric', type=str, default='f1_weighted',
                       help='Metric for best model selection (default: f1_weighted)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--output_dir', type=str, default='./training_output',
                       help='Directory to save final model and results')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data(
        args.data_path, test_size=args.test_size, random_state=args.seed
    )
    
    n_classes = len(le.classes_)
    
    # Preprocess features
    log.info("Preprocessing features...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = get_scaler(args.scaler)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create classifier
    log.info(f"Creating {args.classifier} classifier...")
    model = create_classifier(
        classifier_type=args.classifier,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=n_classes,
        learning_rate=args.learning_rate,
        steps=args.steps,
        hidden_size=args.hidden_size,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose,
        use_classical_readout=args.use_classical_readout
    )
    
    # Train model
    log.info("Starting training...")
    model.fit(
        X_train_scaled, y_train,
        resume=args.resume_mode,
        selection_metric=args.selection_metric,
        validation_frac=0.1,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set
    log.info("Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Save model and artifacts
    log.info("Saving model and artifacts...")
    model_path = os.path.join(args.output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    log.info(f"Model saved to {model_path}")
    
    scaler_path = os.path.join(args.output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved to {scaler_path}")
    
    imputer_path = os.path.join(args.output_dir, 'imputer.joblib')
    joblib.dump(imputer, imputer_path)
    log.info(f"Imputer saved to {imputer_path}")
    
    le_path = os.path.join(args.output_dir, 'label_encoder.joblib')
    joblib.dump(le, le_path)
    log.info(f"Label encoder saved to {le_path}")
    
    # Save test predictions
    test_preds_df = pd.DataFrame({
        'true_label': le.inverse_transform(y_test),
        'predicted_label': le.inverse_transform(y_pred)
    })
    for i, class_name in enumerate(le.classes_):
        test_preds_df[f'prob_{class_name}'] = y_pred_proba[:, i]
    
    preds_path = os.path.join(args.output_dir, 'test_predictions.csv')
    test_preds_df.to_csv(preds_path, index=False)
    log.info(f"Test predictions saved to {preds_path}")
    
    # Compute and log test metrics
    from sklearn.metrics import accuracy_score, classification_report
    test_accuracy = accuracy_score(y_test, y_pred)
    log.info(f"\n=== Test Set Performance ===")
    log.info(f"Test Accuracy: {test_accuracy:.4f}")
    log.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    log.info(f"\n=== Training Complete ===")
    log.info(f"Outputs saved to: {args.output_dir}")
    log.info(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
