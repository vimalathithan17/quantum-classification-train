"""
Utilities for computing and logging detailed metrics for classification tasks.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, List, Optional
import pandas as pd


def compute_per_class_specificity(y_true: np.ndarray, y_pred: np.ndarray, 
                                   n_classes: int) -> Dict[int, float]:
    """
    Compute specificity (true negative rate) for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes
        
    Returns:
        Dictionary mapping class index to specificity
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    specificities = {}
    for i in range(n_classes):
        # True negatives: sum of all cells except row i and column i
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        # False positives: sum of column i except cell (i, i)
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities[i] = specificity
    
    return specificities


def compute_epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None,
                         loss: Optional[float] = None,
                         n_classes: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for one epoch.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        loss: Training/validation loss (optional)
        n_classes: Number of classes (auto-detected if None)
        
    Returns:
        Dictionary with all computed metrics
    """
    if n_classes is None:
        n_classes = len(np.unique(y_true))
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision (macro and weighted)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Recall (macro and weighted)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # F1 (macro and weighted)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class specificity
    specificities = compute_per_class_specificity(y_true, y_pred, n_classes)
    for class_idx, spec in specificities.items():
        metrics[f'specificity_class_{class_idx}'] = spec
    
    # Confusion matrix (flattened for CSV storage)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    for i in range(n_classes):
        for j in range(n_classes):
            metrics[f'cm_{i}_{j}'] = cm[i, j]
    
    # Loss (if provided)
    if loss is not None:
        metrics['loss'] = loss
    
    return metrics


def format_metrics_for_logging(metrics: Dict[str, Any], 
                               prefix: str = "") -> str:
    """
    Format metrics dictionary for console logging.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for log message (e.g., "Train: " or "Val: ")
        
    Returns:
        Formatted string
    """
    # Select key metrics for display
    key_metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 
                   'recall_weighted', 'loss']
    
    parts = []
    for key in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
    
    return prefix + ", ".join(parts)


def save_confusion_matrix_plot(cm: np.ndarray, 
                               filepath: str,
                               class_names: Optional[List[str]] = None,
                               title: str = "Confusion Matrix") -> None:
    """
    Save confusion matrix as a PNG plot.
    
    Args:
        cm: Confusion matrix (n_classes x n_classes)
        filepath: Output file path
        class_names: List of class names (optional)
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def save_metrics_plots(epoch_history: List[Dict[str, Any]], 
                      checkpoint_dir: str,
                      metrics_to_plot: Optional[List[str]] = None) -> None:
    """
    Save plots for selected metrics over training epochs.
    
    Args:
        epoch_history: List of epoch metric dictionaries
        checkpoint_dir: Directory to save plots
        metrics_to_plot: List of metric names to plot (if None, plots key metrics)
    """
    import matplotlib.pyplot as plt
    import os
    
    if not epoch_history:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Default metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = ['loss', 'accuracy', 'f1_weighted', 'precision_weighted', 
                          'recall_weighted', 'f1_macro']
    
    df = pd.DataFrame(epoch_history)
    
    # Filter to metrics that exist in the data
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if not available_metrics:
        return
    
    # Create separate plots for loss and other metrics
    if 'loss' in available_metrics:
        plt.figure(figsize=(10, 6))
        if 'epoch' in df.columns:
            plt.plot(df['epoch'], df['loss'], marker='o', label='Loss')
            plt.xlabel('Epoch')
        else:
            plt.plot(df['loss'], marker='o', label='Loss')
            plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'), dpi=150)
        plt.close()
    
    # Plot other metrics together
    other_metrics = [m for m in available_metrics if m != 'loss']
    if other_metrics:
        plt.figure(figsize=(12, 6))
        for metric in other_metrics:
            if 'epoch' in df.columns:
                plt.plot(df['epoch'], df[metric], marker='o', label=metric)
            else:
                plt.plot(df[metric], marker='o', label=metric)
        
        plt.xlabel('Epoch' if 'epoch' in df.columns else 'Step')
        plt.ylabel('Metric Value')
        plt.title('Training Metrics')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, 'metrics_plot.png'), dpi=150)
        plt.close()


def get_selection_metric(metrics: Dict[str, Any], 
                        metric_name: str = 'f1_weighted') -> float:
    """
    Get the specified selection metric from metrics dictionary.
    
    Args:
        metrics: Dictionary of computed metrics
        metric_name: Name of metric to use for selection (default: 'f1_weighted')
        
    Returns:
        Metric value (higher is better, so negate loss if using loss)
    """
    if metric_name.lower() == 'loss':
        # For loss, lower is better, so return negative
        return -metrics.get('loss', float('inf'))
    else:
        # For other metrics, higher is better
        return metrics.get(metric_name, 0.0)
