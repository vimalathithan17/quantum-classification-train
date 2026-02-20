#!/usr/bin/env python
"""
Hyperparameter tuning script for multimodal transformer fusion model using Optuna.

This script tunes the transformer fusion model hyperparameters:
- embed_dim: Embedding dimension (64, 128, 256, 512)
- num_heads: Number of attention heads (4, 8, 16)
- num_layers: Number of transformer layers (2-6)
- lr: Learning rate
- batch_size: Training batch size

Usage:
    python examples/tune_transformer_fusion.py --data_dir /path/to/data --n_trials 50
    
    # With CUDA and custom epochs
    python examples/tune_transformer_fusion.py --device cuda --num_epochs 75 --n_trials 100
"""

import os
import argparse
import json
import random
import tempfile
import signal
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from performance_extensions.transformer_fusion import MultimodalFusionClassifier
from logging_utils import log

# Directories configurable via environment
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
TUNING_RESULTS_DIR = os.environ.get('TUNING_RESULTS_DIR', 'transformer_tuning_results')
OPTUNA_DB_PATH = os.environ.get('OPTUNA_DB_PATH', './optuna_transformer_studies.db')
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))

# Global flag for interruption handling
interrupted = False


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info(f"Random seed set to {seed}")


def handle_interruption(signum, frame):
    """Handle interruption signals (SIGINT, SIGTERM) gracefully."""
    global interrupted
    if not interrupted:
        interrupted = True
        log.warning(f"\nReceived interruption signal ({signum}). Finishing current trial...")
        log.warning("Press Ctrl+C again to force exit.")
    else:
        log.error("\nForced interruption. Exiting immediately.")
        sys.exit(1)


def is_db_writable(db_path):
    """Check if a database file is writable."""
    if not os.path.exists(db_path):
        parent_dir = os.path.dirname(db_path) or '.'
        return os.access(parent_dir, os.W_OK)
    return os.access(db_path, os.W_OK)


def ensure_writable_db(db_path):
    """Ensure the database is writable. If read-only, copy to a writable location."""
    if is_db_writable(db_path):
        return db_path
    
    log.warning(f"Database at {db_path} is read-only. Copying to a writable location...")
    
    candidate_paths = [
        os.path.join(os.getcwd(), 'optuna_transformer_working.db'),
        os.path.join(tempfile.gettempdir(), 'optuna_transformer_working.db')
    ]
    
    for candidate in candidate_paths:
        try:
            parent_dir = os.path.dirname(candidate) or '.'
            if os.access(parent_dir, os.W_OK):
                if os.path.exists(db_path):
                    import shutil
                    shutil.copy2(db_path, candidate)
                log.info(f"Using writable database at: {candidate}")
                return candidate
        except Exception:
            continue
    
    log.warning("No writable location found, using in-memory database")
    return ":memory:"


def load_multiomics_data(data_dir, modalities=None):
    """Load multi-omics data from parquet files."""
    data_dir = Path(data_dir)
    
    if modalities is None:
        modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot']
    
    data = {}
    modality_dims = {}
    labels = None
    case_ids = None
    
    for modality in modalities:
        file_path = data_dir / f"data_{modality}_.parquet"
        
        if file_path.exists():
            log.info(f"Loading {modality} from {file_path}")
            df = pd.read_parquet(file_path)
            
            if 'case_id' in df.columns:
                df = df.sort_values('case_id')
                if case_ids is None:
                    case_ids = df['case_id'].values
            
            METADATA_COLS = {'class', 'case_id'}
            feature_cols = [col for col in df.columns if col not in METADATA_COLS]
            features = df[feature_cols].values.astype(np.float32)
            
            # Handle NaN values with median imputation
            col_medians = np.nanmedian(features, axis=0)
            nan_mask = np.isnan(features)
            for col_idx in range(features.shape[1]):
                features[nan_mask[:, col_idx], col_idx] = col_medians[col_idx]
            
            data[modality] = features
            modality_dims[modality] = features.shape[1]
            
            if labels is None and 'class' in df.columns:
                labels = df['class'].values
            
            log.info(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
    
    # Encode labels
    if labels is not None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(labels)
    
    return data, labels, modality_dims


class MultiOmicsDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for multi-omics data."""
    
    def __init__(self, data_dict, labels):
        self.data = {k: torch.from_numpy(v).float() for k, v in data_dict.items()}
        self.labels = torch.from_numpy(labels).long()
        self.n_samples = len(labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sample_data = {k: v[idx] for k, v in self.data.items()}
        return sample_data, self.labels[idx]


def create_dataloader(data, labels, batch_size, shuffle=True):
    """Create PyTorch dataloader from numpy arrays."""
    dataset = MultiOmicsDataset(data, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )


def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_data, batch_labels in dataloader:
        for modality in batch_data:
            if batch_data[modality] is not None:
                batch_data[modality] = batch_data[modality].to(device)
        batch_labels = batch_labels.to(device)
        
        logits, _ = model(batch_data)
        loss = criterion(logits, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            for modality in batch_data:
                if batch_data[modality] is not None:
                    batch_data[modality] = batch_data[modality].to(device)
            batch_labels = batch_labels.to(device)
            
            logits, _ = model(batch_data)
            loss = criterion(logits, batch_labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def objective(trial, args, data, labels, modality_dims, n_classes, device):
    """Optuna objective function for transformer fusion tuning."""
    global interrupted
    
    if interrupted:
        raise optuna.TrialPruned()
    
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Hyperparameters to tune
    embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256, 512])
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Ensure embed_dim is divisible by num_heads
    while embed_dim % num_heads != 0:
        num_heads = num_heads // 2
        if num_heads < 2:
            num_heads = 2
            break
    
    log.info(f"Trial {trial.number} params: embed_dim={embed_dim}, num_heads={num_heads}, "
             f"num_layers={num_layers}, lr={lr:.6f}, batch_size={batch_size}, dropout={dropout:.2f}")
    
    # K-Fold cross-validation
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if interrupted:
            raise optuna.TrialPruned()
        
        log.info(f"  Fold {fold + 1}/{n_splits}")
        
        # Split data
        train_data = {k: v[train_idx] for k, v in data.items()}
        val_data = {k: v[val_idx] for k, v in data.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Create dataloaders
        train_loader = create_dataloader(train_data, train_labels, batch_size, shuffle=True)
        val_loader = create_dataloader(val_data, val_labels, batch_size, shuffle=False)
        
        # Create model
        model = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=n_classes,
            dropout=dropout
        )
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        best_f1 = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(args.num_epochs):
            if interrupted:
                raise optuna.TrialPruned()
            
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                log.info(f"    Early stopping at epoch {epoch + 1}")
                break
            
            # Report intermediate results for pruning
            trial.report(val_f1, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        fold_scores.append(best_f1)
        log.info(f"    Fold {fold + 1} best F1: {best_f1:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    
    trial.set_user_attr('mean_f1', float(mean_f1))
    trial.set_user_attr('std_f1', float(std_f1))
    
    log.info(f"--- Trial {trial.number} Finished: mean_f1 = {mean_f1:.4f} ± {std_f1:.4f} ---")
    
    return mean_f1


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Transformer Fusion model using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Basic tuning with CUDA
  python examples/tune_transformer_fusion.py --device cuda --n_trials 50
  
  # Custom epochs per trial
  python examples/tune_transformer_fusion.py --num_epochs 75 --n_trials 100
  
  # Continue existing study
  python examples/tune_transformer_fusion.py --study_name my_study --n_trials 20
        """)
    
    # Data configuration
    data_args = parser.add_argument_group('data configuration')
    data_args.add_argument('--data_dir', type=str, default=None,
                          help='Directory with parquet files (default: from SOURCE_DIR env)')
    data_args.add_argument('--modalities', type=str, nargs='+', 
                          default=['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot'],
                          help='Modalities to use (default: GeneExpr miRNA Meth CNV Prot)')
    
    # Tuning configuration
    tuning_args = parser.add_argument_group('tuning parameters')
    tuning_args.add_argument('--n_trials', type=int, default=50,
                            help='Number of Optuna trials (default: 50)')
    tuning_args.add_argument('--num_epochs', type=int, default=50,
                            help='Training epochs per trial (default: 50)')
    tuning_args.add_argument('--study_name', type=str, default=None,
                            help='Custom study name (auto-generated if not provided)')
    
    # System configuration
    system_args = parser.add_argument_group('system configuration')
    system_args.add_argument('--device', type=str,
                            default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='Device: cuda or cpu (default: auto-detect)')
    system_args.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    
    # Logging configuration
    log_args = parser.add_argument_group('logging configuration')
    log_args.add_argument('--verbose', action='store_true',
                         help='Enable verbose output')
    log_args.add_argument('--use_wandb', action='store_true',
                         help='Enable Weights & Biases tracking')
    log_args.add_argument('--wandb_project', type=str, default='transformer-fusion-tuning',
                         help='W&B project name')
    log_args.add_argument('--wandb_run_name', type=str, default=None,
                         help='W&B run name')
    
    args = parser.parse_args()
    
    # Setup
    signal.signal(signal.SIGINT, handle_interruption)
    signal.signal(signal.SIGTERM, handle_interruption)
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    log.info(f"Using device: {device}")
    
    # Load data
    data_dir = args.data_dir or SOURCE_DIR
    log.info(f"Loading data from: {data_dir}")
    data, labels, modality_dims = load_multiomics_data(data_dir, args.modalities)
    
    if not data or labels is None:
        log.error("Failed to load data!")
        return
    
    n_classes = len(np.unique(labels))
    log.info(f"Loaded {len(data)} modalities, {len(labels)} samples, {n_classes} classes")
    
    # Study name
    study_name = args.study_name or f'transformer_fusion_tuning_{args.num_epochs}ep'
    log.info(f"Study name: {study_name}")
    
    # Database
    writable_db = ensure_writable_db(OPTUNA_DB_PATH)
    storage = f"sqlite:///{writable_db}" if writable_db != ":memory:" else None
    
    # Initialize wandb if requested
    wandb_callback = None
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or study_name,
                config={'n_trials': args.n_trials, 'num_epochs': args.num_epochs}
            )
            wandb_callback = optuna.integration.WeightsAndBiasesCallback(
                metric_name='mean_f1', wandb_kwargs={'project': args.wandb_project}
            )
        except Exception as e:
            log.warning(f"Failed to initialize wandb: {e}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    existing_trials = len(study.trials)
    log.info(f"Study has {existing_trials} existing trials")
    
    # Run optimization
    callbacks = [wandb_callback] if wandb_callback else []
    
    def interruption_callback(study, trial):
        if interrupted:
            study.stop()
    
    callbacks.append(interruption_callback)
    
    try:
        study.optimize(
            lambda t: objective(t, args, data, labels, modality_dims, n_classes, device),
            n_trials=args.n_trials,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        log.warning("Optimization interrupted by user")
    
    # Results
    log.info("\n" + "="*80)
    log.info("Hyperparameter Tuning Complete")
    log.info("="*80)
    
    if len(study.trials) > 0:
        try:
            best_trial = study.best_trial
            log.info(f"\nBest trial: {best_trial.number}")
            log.info(f"Best F1 score: {best_trial.value:.4f}")
            log.info(f"Best parameters:")
            for key, value in best_trial.params.items():
                log.info(f"  {key}: {value}")
            
            # Save best config
            results_dir = Path(TUNING_RESULTS_DIR)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            best_config = {
                'study_name': study_name,
                'best_trial': best_trial.number,
                'best_f1': best_trial.value,
                'best_params': best_trial.params,
                'n_trials': len(study.trials),
                'num_epochs': args.num_epochs
            }
            
            config_path = results_dir / f'{study_name}_best_config.json'
            with open(config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            log.info(f"\nBest config saved to: {config_path}")
            
        except ValueError as e:
            log.error(f"No completed trials: {e}")
    else:
        log.warning("No trials completed")


if __name__ == "__main__":
    main()
