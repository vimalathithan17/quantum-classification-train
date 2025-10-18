"""Helper utilities for metrics computation and visualization."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import pandas as pd
import os


def compute_specificity(y_true, y_pred, n_classes):
    """
    Compute specificity per class.
    
    Specificity for class i = TN_i / (TN_i + FP_i)
    where TN_i is true negatives and FP_i is false positives for class i.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specificities = []
    
    for i in range(n_classes):
        # For class i, TN = sum of all elements not in row i or column i
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        # FP = sum of column i minus the diagonal element
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        if (tn + fp) > 0:
            spec = tn / (tn + fp)
        else:
            spec = 0.0
        specificities.append(spec)
    
    return np.array(specificities)


def compute_metrics(y_true, y_pred, n_classes):
    """
    Compute comprehensive metrics for classification.
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 (macro and weighted)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision_macro'] = prec_macro
    metrics['precision_weighted'] = prec_weighted
    metrics['recall_macro'] = rec_macro
    metrics['recall_weighted'] = rec_weighted
    metrics['f1_macro'] = f1_macro
    metrics['f1_weighted'] = f1_weighted
    
    # Specificity
    specificities = compute_specificity(y_true, y_pred, n_classes)
    metrics['specificity_macro'] = np.mean(specificities)
    metrics['specificity_weighted'] = np.average(
        specificities,
        weights=np.bincount(y_true, minlength=n_classes)
    )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    return metrics


def save_metrics_to_csv(history, checkpoint_dir):
    """
    Save training history to CSV file.
    
    Args:
        history: Dictionary containing training history
        checkpoint_dir: Directory to save CSV file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build dataframe from history
    rows = []
    for i in range(len(history.get('train_loss', []))):
        row = {'step': i}
        
        # Add all available metrics
        for key, values in history.items():
            if i < len(values):
                row[key] = values[i]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(checkpoint_dir, 'history.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_training_curves(history, checkpoint_dir):
    """
    Create and save training visualization plots.
    
    Args:
        history: Dictionary containing training history
        checkpoint_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Plot loss curves
        if 'train_loss' in history and len(history['train_loss']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
            if 'val_loss' in history and len(history['val_loss']) > 0:
                plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(checkpoint_dir, 'loss.png'), dpi=100)
            plt.close()
        
        # Plot accuracy curves
        if 'val_acc' in history and len(history['val_acc']) > 0:
            plt.figure(figsize=(10, 6))
            if 'train_acc' in history:
                plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
            plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(checkpoint_dir, 'accuracy.png'), dpi=100)
            plt.close()
        
        # Plot F1 scores
        if 'val_f1_weighted' in history and len(history['val_f1_weighted']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(history['val_f1_weighted'], label='Weighted F1', linewidth=2)
            if 'val_f1_macro' in history:
                plt.plot(history['val_f1_macro'], label='Macro F1', linewidth=2, alpha=0.7)
            plt.xlabel('Step')
            plt.ylabel('F1 Score')
            plt.title('Validation F1 Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(checkpoint_dir, 'f1_scores.png'), dpi=100)
            plt.close()
        
        # Plot precision and recall
        if 'val_prec_weighted' in history and len(history['val_prec_weighted']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(history['val_prec_weighted'], label='Precision (weighted)', linewidth=2)
            if 'val_rec_weighted' in history:
                plt.plot(history['val_rec_weighted'], label='Recall (weighted)', linewidth=2)
            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.title('Validation Precision and Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(checkpoint_dir, 'precision_recall.png'), dpi=100)
            plt.close()
            
    except ImportError:
        # matplotlib not available, skip plotting
        pass
    except Exception as e:
        # Log error but don't fail
        print(f"Warning: Failed to create plots: {e}")
