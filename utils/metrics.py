"""
Utility functions for metrics logging and evaluation.
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


def compute_classification_metrics(y_true, y_pred, n_classes):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes
    
    Returns:
        dict: Dictionary of metric values
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, f1 with macro and weighted averaging
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    metrics['confusion_matrix'] = cm
    
    # Specificity (per class)
    specificities = []
    for i in range(n_classes):
        # True negatives: correctly predicted as not class i
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        # False positives: incorrectly predicted as class i
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    metrics['specificity_macro'] = np.mean(specificities)
    
    return metrics


def save_metrics_to_csv(metrics_history, output_path):
    """
    Save metrics history to CSV file.
    
    Args:
        metrics_history: List of dicts, each containing epoch metrics
        output_path: Path to save CSV file
    """
    # Filter out confusion matrices for CSV (too large)
    metrics_for_csv = []
    for m in metrics_history:
        m_filtered = {k: v for k, v in m.items() if k != 'confusion_matrix'}
        metrics_for_csv.append(m_filtered)
    
    df = pd.DataFrame(metrics_for_csv)
    df.to_csv(output_path, index=False)


def plot_metrics(metrics_history, output_dir):
    """
    Generate and save plots of training metrics.
    
    Args:
        metrics_history: List of dicts, each containing epoch metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = [m['epoch'] for m in metrics_history]
    
    # Plot 1: Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'train_loss' in metrics_history[0]:
        ax1.plot(epochs, [m['train_loss'] for m in metrics_history], label='Train Loss', marker='o')
    if 'val_loss' in metrics_history[0]:
        ax1.plot(epochs, [m['val_loss'] for m in metrics_history], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if 'train_accuracy' in metrics_history[0]:
        ax2.plot(epochs, [m['train_accuracy'] for m in metrics_history], label='Train Acc', marker='o')
    if 'val_accuracy' in metrics_history[0]:
        ax2.plot(epochs, [m['val_accuracy'] for m in metrics_history], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_accuracy.png'), dpi=150)
    plt.close()
    
    # Plot 2: F1 Scores
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if 'val_f1_macro' in metrics_history[0]:
        ax.plot(epochs, [m['val_f1_macro'] for m in metrics_history], 
                label='Val F1 Macro', marker='o')
    if 'val_f1_weighted' in metrics_history[0]:
        ax.plot(epochs, [m['val_f1_weighted'] for m in metrics_history], 
                label='Val F1 Weighted', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_scores.png'), dpi=150)
    plt.close()
    
    # Plot 3: Confusion Matrix (last epoch)
    if metrics_history and 'confusion_matrix' in metrics_history[-1]:
        cm = metrics_history[-1]['confusion_matrix']
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set_title('Confusion Matrix (Last Epoch)')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()


def plot_confusion_matrix_standalone(cm, output_path, title='Confusion Matrix'):
    """
    Save a standalone confusion matrix plot.
    
    Args:
        cm: Confusion matrix numpy array
        output_path: Path to save the PNG file
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
