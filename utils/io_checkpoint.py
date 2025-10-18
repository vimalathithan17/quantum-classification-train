"""
Checkpoint I/O utilities for saving and loading training checkpoints.
Supports saving quantum parameters, classical parameters, optimizer state, RNG state, and metadata.
"""
import joblib
import os
import numpy as np


def save_checkpoint(path, data_dict):
    """
    Save a checkpoint to disk using joblib.
    
    Args:
        path (str): Path to save the checkpoint
        data_dict (dict): Dictionary containing checkpoint data with keys like:
            - 'quantum_params': quantum circuit parameters
            - 'classical_params': classical readout layer parameters (optional)
            - 'optimizer_state': optimizer state dict from optimizer.get_state()
            - 'rng_state': random number generator state
            - 'step': current training step
            - 'best_val_metric': best validation metric value
            - 'epoch': current epoch (optional)
            - 'metadata': additional metadata (optional)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    joblib.dump(data_dict, path)


def load_checkpoint(path):
    """
    Load a checkpoint from disk.
    
    Args:
        path (str): Path to the checkpoint file
    
    Returns:
        dict: Dictionary containing checkpoint data
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return joblib.load(path)


def save_best_checkpoint(checkpoint_dir, data_dict, metric_name='best_val_metric'):
    """
    Save the best checkpoint based on validation metric.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints
        data_dict (dict): Checkpoint data dictionary
        metric_name (str): Name of the metric used for tracking best model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, 'best_weights.joblib')
    save_checkpoint(path, data_dict)


def save_latest_checkpoint(checkpoint_dir, data_dict, step):
    """
    Save the latest checkpoint at a specific step.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints
        data_dict (dict): Checkpoint data dictionary
        step (int): Current training step
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.joblib')
    save_checkpoint(path, data_dict)


def get_latest_checkpoint_path(checkpoint_dir):
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir (str): Directory to search for checkpoints
    
    Returns:
        str or None: Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('checkpoint_step_') and f.endswith('.joblib')]
    
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find the latest
    steps = []
    for f in checkpoint_files:
        try:
            step = int(f.replace('checkpoint_step_', '').replace('.joblib', ''))
            steps.append((step, f))
        except ValueError:
            continue
    
    if not steps:
        return None
    
    steps.sort(reverse=True)
    return os.path.join(checkpoint_dir, steps[0][1])


def get_best_checkpoint_path(checkpoint_dir):
    """
    Get path to the best checkpoint.
    
    Args:
        checkpoint_dir (str): Directory to search for checkpoints
    
    Returns:
        str or None: Path to best checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    best_path = os.path.join(checkpoint_dir, 'best_weights.joblib')
    return best_path if os.path.exists(best_path) else None
