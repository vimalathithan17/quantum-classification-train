"""
Checkpoint I/O utilities for saving and loading model state.

This module provides standardized checkpoint save/load functions that include:
- Model parameters (quantum and classical)
- Optimizer state
- RNG state
- Metadata and validation metrics
"""
import os
import joblib
import numpy as np
from datetime import datetime


def save_checkpoint(checkpoint_path, model_params, classical_params=None, 
                   optimizer_state=None, rng_state=None, metadata=None,
                   metrics=None):
    """
    Save a complete checkpoint with all model state and metadata.
    
    Args:
        checkpoint_path (str): Path where checkpoint will be saved
        model_params (dict): Dictionary of model parameters (e.g., quantum weights)
        classical_params (dict, optional): Classical readout head parameters
        optimizer_state (dict, optional): Optimizer state from get_state()
        rng_state (dict, optional): Random number generator state
        metadata (dict, optional): Additional metadata (epoch, step, etc.)
        metrics (dict, optional): Validation metrics for this checkpoint
    """
    checkpoint = {
        'model_params': model_params,
        'classical_params': classical_params,
        'optimizer_state': optimizer_state,
        'rng_state': rng_state,
        'metadata': metadata or {},
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
    
    # Save with joblib
    joblib.dump(checkpoint, checkpoint_path)
    

def load_checkpoint(checkpoint_path):
    """
    Load a checkpoint from disk.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        dict: Checkpoint dictionary with all saved state
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = joblib.load(checkpoint_path)
    return checkpoint


def save_best_and_latest_checkpoints(checkpoint_dir, model_params, classical_params=None,
                                     optimizer_state=None, rng_state=None, 
                                     step=0, epoch=0, loss=None, metrics=None,
                                     is_best=False):
    """
    Save both latest and best checkpoints.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints
        model_params (dict): Model parameters
        classical_params (dict, optional): Classical parameters
        optimizer_state (dict, optional): Optimizer state
        rng_state (dict, optional): RNG state
        step (int): Current training step
        epoch (int): Current epoch
        loss (float, optional): Current loss value
        metrics (dict, optional): Current metrics
        is_best (bool): Whether this is the best checkpoint so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    metadata = {
        'step': step,
        'epoch': epoch,
        'loss': loss,
    }
    
    # Always save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.joblib')
    save_checkpoint(latest_path, model_params, classical_params, 
                   optimizer_state, rng_state, metadata, metrics)
    
    # Save best checkpoint if this is the best
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.joblib')
        save_checkpoint(best_path, model_params, classical_params,
                       optimizer_state, rng_state, metadata, metrics)


def get_rng_state():
    """
    Get the current state of all random number generators.
    
    Returns:
        dict: RNG states for numpy
    """
    return {
        'numpy': np.random.get_state()
    }


def set_rng_state(rng_state):
    """
    Restore random number generator states.
    
    Args:
        rng_state (dict): RNG state dictionary from get_rng_state()
    """
    if rng_state and 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir (str): Directory to search
        
    Returns:
        str or None: Path to latest checkpoint, or None if not found
    """
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.joblib')
    if os.path.exists(latest_path):
        return latest_path
    return None


def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint in a directory.
    
    Args:
        checkpoint_dir (str): Directory to search
        
    Returns:
        str or None: Path to best checkpoint, or None if not found
    """
    best_path = os.path.join(checkpoint_dir, 'checkpoint_best.joblib')
    if os.path.exists(best_path):
        return best_path
    return None
