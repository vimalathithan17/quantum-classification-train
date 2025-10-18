"""
Standardized checkpoint save/load utilities for quantum classifiers.
Handles quantum params, classical params, optimizer state, and metadata.
"""
import os
import joblib
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def save_checkpoint(
    checkpoint_dir: str,
    quantum_params: Any,
    classical_params: Optional[Dict[str, Any]] = None,
    optimizer_state: Optional[Dict[str, Any]] = None,
    step: int = 0,
    loss: float = float('inf'),
    best_val_metric: Optional[float] = None,
    rng_state: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    checkpoint_name: Optional[str] = None
) -> str:
    """
    Save a training checkpoint with all state information.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        quantum_params: Quantum circuit parameters (weights)
        classical_params: Classical readout head parameters (optional)
        optimizer_state: Optimizer state dict from get_state() (optional)
        step: Current training step
        loss: Current loss value
        best_val_metric: Best validation metric value (optional)
        rng_state: Random number generator state (optional)
        metadata: Additional metadata dict (optional)
        is_best: Whether this is the best checkpoint
        checkpoint_name: Custom checkpoint name (default: 'checkpoint_step_{step}.joblib')
        
    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'quantum_params': quantum_params,
        'classical_params': classical_params,
        'optimizer_state': optimizer_state,
        'step': step,
        'loss': loss,
        'best_val_metric': best_val_metric,
        'rng_state': rng_state,
        'metadata': metadata or {}
    }
    
    # Determine filename
    if checkpoint_name:
        filename = checkpoint_name
    elif is_best:
        filename = 'best_checkpoint.joblib'
    else:
        filename = f'checkpoint_step_{step}.joblib'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Save checkpoint
    joblib.dump(checkpoint_data, checkpoint_path)
    
    # Also save 'latest' symlink or copy
    if not is_best:
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.joblib')
        joblib.dump(checkpoint_data, latest_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_dir: str,
    checkpoint_type: str = 'auto'
) -> Optional[Dict[str, Any]]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_type: Type of checkpoint to load:
            - 'auto': Load best if exists, else latest, else None
            - 'latest': Load latest checkpoint
            - 'best': Load best checkpoint
            - specific filename: Load that checkpoint file
            
    Returns:
        Dictionary with checkpoint data or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_path = None
    
    if checkpoint_type == 'auto':
        # Try best first, then latest
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.joblib')
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.joblib')
        
        if os.path.exists(best_path):
            checkpoint_path = best_path
        elif os.path.exists(latest_path):
            checkpoint_path = latest_path
        else:
            return None
            
    elif checkpoint_type == 'latest':
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.joblib')
        
    elif checkpoint_type == 'best':
        checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.joblib')
        
    else:
        # Custom filename
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_type)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            return joblib.load(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            return None
    
    return None


def save_epoch_history(
    checkpoint_dir: str,
    epoch_history: list,
    filename: str = 'epoch_history.csv'
) -> None:
    """
    Save epoch history to CSV file.
    
    Args:
        checkpoint_dir: Directory to save history
        epoch_history: List of dicts with epoch metrics
        filename: CSV filename (default: 'epoch_history.csv')
    """
    import pandas as pd
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    
    df = pd.DataFrame(epoch_history)
    df.to_csv(filepath, index=False)


def load_epoch_history(
    checkpoint_dir: str,
    filename: str = 'epoch_history.csv'
) -> Optional[list]:
    """
    Load epoch history from CSV file.
    
    Args:
        checkpoint_dir: Directory containing history
        filename: CSV filename (default: 'epoch_history.csv')
        
    Returns:
        List of dicts with epoch metrics or None if not found
    """
    import pandas as pd
    
    filepath = os.path.join(checkpoint_dir, filename)
    
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading epoch history from {filepath}: {e}")
            return None
    
    return None


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True
) -> None:
    """
    Remove old checkpoints, keeping only the last N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to always keep best checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all checkpoint files (excluding best and latest)
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_step_') and filename.endswith('.joblib'):
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoint_files.append((filepath, os.path.getmtime(filepath)))
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old checkpoints
    for filepath, _ in checkpoint_files[keep_last_n:]:
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing old checkpoint {filepath}: {e}")


def get_checkpoint_info(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Get information about available checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Dictionary with checkpoint information
    """
    info = {
        'checkpoint_dir': checkpoint_dir,
        'has_best': False,
        'has_latest': False,
        'num_checkpoints': 0,
        'checkpoint_files': []
    }
    
    if not os.path.exists(checkpoint_dir):
        return info
    
    best_path = os.path.join(checkpoint_dir, 'best_checkpoint.joblib')
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.joblib')
    
    info['has_best'] = os.path.exists(best_path)
    info['has_latest'] = os.path.exists(latest_path)
    
    # Count checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_step_') and filename.endswith('.joblib'):
            checkpoint_files.append(filename)
    
    info['num_checkpoints'] = len(checkpoint_files)
    info['checkpoint_files'] = sorted(checkpoint_files)
    
    return info
