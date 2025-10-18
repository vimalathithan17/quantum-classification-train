"""Checkpoint save/load utilities for quantum machine learning models."""
import os
import joblib
import glob
from typing import Dict, Any, Optional


def save_checkpoint(path: str, data_dict: Dict[str, Any], compress: bool = True) -> None:
    """
    Save checkpoint to disk.
    
    Args:
        path: File path to save checkpoint
        data_dict: Dictionary containing checkpoint data
        compress: Whether to use compression (default True)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    compress_level = 3 if compress else 0
    joblib.dump(data_dict, path, compress=compress_level)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load checkpoint from disk.
    
    Args:
        path: File path to load checkpoint from
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return joblib.load(path)


def save_best_checkpoint(checkpoint_dir: str, data_dict: Dict[str, Any]) -> str:
    """
    Save best model checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        data_dict: Dictionary containing checkpoint data
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, 'best_weights.joblib')
    save_checkpoint(path, data_dict, compress=True)
    return path


def save_periodic_checkpoint(checkpoint_dir: str, step: int, data_dict: Dict[str, Any],
                             keep_last_n: Optional[int] = None) -> str:
    """
    Save periodic checkpoint and optionally clean up old checkpoints.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        step: Current training step
        data_dict: Dictionary containing checkpoint data
        keep_last_n: Number of recent checkpoints to keep (None = keep all)
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.joblib')
    save_checkpoint(path, data_dict, compress=True)
    
    # Clean up old checkpoints if requested
    if keep_last_n is not None and keep_last_n > 0:
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.joblib')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        # Sort by step number
        def extract_step(filepath):
            basename = os.path.basename(filepath)
            try:
                # Extract step number from filename
                return int(basename.split('_')[-1].replace('.joblib', ''))
            except (ValueError, IndexError):
                return 0
        
        checkpoint_files.sort(key=extract_step)
        
        # Remove old checkpoints
        if len(checkpoint_files) > keep_last_n:
            for old_checkpoint in checkpoint_files[:-keep_last_n]:
                try:
                    os.remove(old_checkpoint)
                except OSError:
                    pass  # Ignore errors during cleanup
    
    return path


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.joblib')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by step number and return latest
    def extract_step(filepath):
        basename = os.path.basename(filepath)
        try:
            return int(basename.split('_')[-1].replace('.joblib', ''))
        except (ValueError, IndexError):
            return 0
    
    checkpoint_files.sort(key=extract_step)
    return checkpoint_files[-1]


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the best checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory to search for best checkpoint
        
    Returns:
        Path to best checkpoint or None if not found
    """
    best_path = os.path.join(checkpoint_dir, 'best_weights.joblib')
    return best_path if os.path.exists(best_path) else None
