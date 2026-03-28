#!/usr/bin/env python
"""
Example script for multimodal transformer fusion training.

This script demonstrates Option 1: Multimodal Transformer Fusion
as described in PERFORMANCE_EXTENSIONS.md.

Usage:
    python examples/train_transformer_fusion.py --data_dir /path/to/data --output_dir transformer_models
    
    # With pretrained encoders:
    python examples/train_transformer_fusion.py --pretrained_encoders_dir pretrained_models/contrastive/encoders
"""

import os
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from performance_extensions.transformer_fusion import MultimodalFusionClassifier
from performance_extensions.training_utils import load_pretrained_encoders
from utils.metrics_utils import compute_metrics


def load_pretrained_features(features_dir, modalities=None, label_encoder=None):
    """
    Load pretrained features from extracted embeddings.
    
    Supports two formats:
    1. Split format (preferred, no leakage): {modality}_train_embeddings.npy, {modality}_test_embeddings.npy
    2. Combined format (legacy, warns about leakage): {modality}_embeddings.npy
    
    Args:
        features_dir: Directory containing *_embeddings.npy files
        modalities: List of modality names to load (if None, load all available)
        label_encoder: Pre-fitted LabelEncoder for consistent label mapping
        
    Returns:
        Tuple of (data_dict, labels, modality_dims, case_ids, is_split)
        - If is_split=True: data_dict contains 'train' and 'test' keys with modality sub-dicts
        - If is_split=False: data_dict contains modality keys directly (legacy format)
    """
    features_dir = Path(features_dir)
    
    if modalities is None:
        modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot', 'SNV']
    
    # Check if split files exist (indicates proper train/test split from pretraining)
    first_modality = modalities[0]
    train_file_check = features_dir / f"{first_modality}_train_embeddings.npy"
    test_file_check = features_dir / f"{first_modality}_test_embeddings.npy"
    has_split_files = train_file_check.exists() and test_file_check.exists()
    
    if has_split_files:
        # ✓ Properly split pretrained features - load train/test separately
        print("Loading SPLIT pretrained features (no leakage)")
        
        train_data = {}
        test_data = {}
        modality_dims = {}
        
        # Load train/test labels and case_ids
        train_labels_file = features_dir / 'train_labels.npy'
        test_labels_file = features_dir / 'test_labels.npy'
        train_case_ids_file = features_dir / 'train_case_ids.npy'
        test_case_ids_file = features_dir / 'test_case_ids.npy'
        
        if not train_labels_file.exists() or not test_labels_file.exists():
            raise ValueError(f"Split label files not found in {features_dir}")
        
        train_labels = np.load(train_labels_file, allow_pickle=True)
        test_labels = np.load(test_labels_file, allow_pickle=True)
        train_case_ids = np.load(train_case_ids_file, allow_pickle=True) if train_case_ids_file.exists() else None
        test_case_ids = np.load(test_case_ids_file, allow_pickle=True) if test_case_ids_file.exists() else None
        
        # Encode labels if string
        if train_labels.dtype.kind in ('U', 'S', 'O'):
            if label_encoder is None:
                raise ValueError("String labels found but no label_encoder provided.")
            train_labels = label_encoder.transform(train_labels)
            test_labels = label_encoder.transform(test_labels)
            print(f"Encoded string labels using master encoder: {label_encoder.classes_}")
        train_labels = train_labels.astype(np.int64)
        test_labels = test_labels.astype(np.int64)
        
        print(f"Train labels: {len(train_labels)} samples, classes={np.unique(train_labels)}")
        print(f"Test labels: {len(test_labels)} samples, classes={np.unique(test_labels)}")
        
        # Load embeddings for each modality
        for modality in modalities:
            train_file = features_dir / f"{modality}_train_embeddings.npy"
            test_file = features_dir / f"{modality}_test_embeddings.npy"
            
            if train_file.exists() and test_file.exists():
                print(f"Loading pretrained {modality} (train + test)")
                train_features = np.load(train_file).astype(np.float32)
                test_features = np.load(test_file).astype(np.float32)
                
                # Check for NaN/Inf
                for name, features in [('train', train_features), ('test', test_features)]:
                    nan_count = np.isnan(features).sum()
                    inf_count = np.isinf(features).sum()
                    if nan_count > 0 or inf_count > 0:
                        print(f"  Warning: Found {nan_count} NaN and {inf_count} Inf in {modality} {name}")
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                train_data[modality] = train_features
                test_data[modality] = test_features
                modality_dims[modality] = train_features.shape[1]
                
                print(f"  Train: {train_features.shape[0]} samples, Test: {test_features.shape[0]} samples, dim={train_features.shape[1]}")
            else:
                print(f"Warning: {modality} split files not found, skipping")
        
        if not train_data:
            raise ValueError(f"No split embeddings found in {features_dir}")
        
        # Return with is_split=True
        return {'train': train_data, 'test': test_data}, \
               {'train': train_labels, 'test': test_labels}, \
               modality_dims, \
               {'train': train_case_ids, 'test': test_case_ids}, \
               True  # is_split
    
    else:
        # ⚠️ Old format without split - warn about leakage
        print("⚠️ LEAKAGE WARNING: Using combined pretrained features!")
        print("The encoder may have seen test samples during pretraining.")
        print("For proper evaluation, re-run pretrain_contrastive.py with --test_size 0.2")
        print("Then re-run extract_pretrained_features.py")
        
        data = {}
        modality_dims = {}
        
        # Load case_ids and labels
        case_ids = None
        labels = None
        
        case_ids_file = features_dir / 'case_ids.npy'
        if case_ids_file.exists():
            case_ids = np.load(case_ids_file, allow_pickle=True)
            print(f"Loaded case_ids: {len(case_ids)} samples")
        
        labels_file = features_dir / 'labels.npy'
        if labels_file.exists():
            labels = np.load(labels_file, allow_pickle=True)
            if labels.dtype.kind in ('U', 'S', 'O'):
                if label_encoder is None:
                    raise ValueError("String labels found but no label_encoder provided.")
                labels = label_encoder.transform(labels)
                print(f"Encoded string labels using master encoder: {label_encoder.classes_}")
            labels = labels.astype(np.int64)
            print(f"Loaded labels: {len(labels)} samples, classes={np.unique(labels)}")
        
        # Load embeddings for each modality
        for modality in modalities:
            file_path = features_dir / f"{modality}_embeddings.npy"
            
            if file_path.exists():
                print(f"Loading pretrained {modality} from {file_path}")
                features = np.load(file_path).astype(np.float32)
                
                nan_count = np.isnan(features).sum()
                inf_count = np.isinf(features).sum()
                if nan_count > 0 or inf_count > 0:
                    print(f"  Warning: Found {nan_count} NaN and {inf_count} Inf values in {modality}")
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                data[modality] = features
                modality_dims[modality] = features.shape[1]
                
                print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
            else:
                print(f"Warning: {file_path} not found, skipping {modality}")
        
        if not data:
            raise ValueError(f"No embeddings found in {features_dir}")
        
        # Return with is_split=False
        return data, labels, modality_dims, case_ids, False  # is_split


def load_multiomics_data(data_dir, modalities=None, label_encoder=None):
    """
    Load multi-omics data from parquet files.
    
    Args:
        data_dir: Directory containing data files
        modalities: List of modality names to load (if None, load all available)
        label_encoder: Pre-fitted LabelEncoder for consistent label mapping
        
    Returns:
        Tuple of (data_dict, labels, modality_dims)
    
    Note: Standardization should be done AFTER train/test split to avoid data leakage.
    """
    data_dir = Path(data_dir)
    
    if modalities is None:
        modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot', 'SNV']
    
    data = {}
    modality_dims = {}
    labels = None
    case_ids = None
    
    for modality in modalities:
        file_path = data_dir / f"data_{modality}_.parquet"
        
        if file_path.exists():
            print(f"Loading {modality} from {file_path}")
            df = pd.read_parquet(file_path)
            
            # CRITICAL: Sort by case_id for consistent ordering across all scripts
            if 'case_id' in df.columns:
                df = df.sort_values('case_id')
                # Extract case_ids from first modality
                if case_ids is None:
                    case_ids = df['case_id'].values
            
            # Metadata columns to exclude from features (only case_id and class exist in the data)
            METADATA_COLS = {'class', 'case_id'}
            
            # Extract features (exclude metadata columns)
            feature_cols = [col for col in df.columns if col not in METADATA_COLS]
            features = df[feature_cols].values.astype(np.float32)
            
            # Check for NaN/Inf values and handle them
            nan_count = np.isnan(features).sum()
            inf_count = np.isinf(features).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  Warning: Found {nan_count} NaN and {inf_count} Inf values in {modality}")
                # Replace NaN with column mean, Inf with large finite value
                col_means = np.nanmean(features, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)  # If column is all NaN, use 0
                for col_idx in range(features.shape[1]):
                    mask_nan = np.isnan(features[:, col_idx])
                    mask_inf = np.isinf(features[:, col_idx])
                    features[mask_nan, col_idx] = col_means[col_idx]
                    features[mask_inf, col_idx] = col_means[col_idx]
                print(f"  Replaced NaN/Inf with column means")
            
            # NOTE: Standardization moved to AFTER train/test split to avoid data leakage
            
            data[modality] = features
            modality_dims[modality] = features.shape[1]
            
            # Extract labels (from first modality encountered)
            if labels is None and 'class' in df.columns:
                labels = df['class'].values
                # If labels are strings, encode them to integers using master encoder
                if labels.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
                    if label_encoder is None:
                        raise ValueError("String labels found but no label_encoder provided. "
                                       "Use --label_encoder_path to specify master label encoder.")
                    labels = label_encoder.transform(labels)
                    print(f"  Encoded string labels using master encoder: {label_encoder.classes_}")
                labels = labels.astype(np.int64)
            
            print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
        else:
            print(f"Warning: {file_path} not found, skipping {modality}")
    
    return data, labels, modality_dims


def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=None):
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping (None or 0 to disable)
    
    Returns:
        Tuple of (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    processed_batches = 0
    nan_detected = False
    
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        # Move data to device
        for modality in batch_data:
            if batch_data[modality] is not None:
                batch_data[modality] = batch_data[modality].to(device)
                # Check for NaN/Inf in input data
                if torch.isnan(batch_data[modality]).any() or torch.isinf(batch_data[modality]).any():
                    if not nan_detected:
                        print(f"  Warning: NaN/Inf detected in {modality} input data at batch {batch_idx}")
                        nan_detected = True
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        logits, _ = model(batch_data)
        
        # Check for NaN in logits - skip batch if found
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            if not nan_detected:
                print(f"  Warning: NaN/Inf detected in model outputs at batch {batch_idx}")
                nan_detected = True
            continue
        
        loss = criterion(logits, batch_labels)
        
        # Skip NaN loss to prevent model corruption
        if torch.isnan(loss) or torch.isinf(loss):
            if not nan_detected:
                print(f"  Warning: NaN/Inf loss at batch {batch_idx}, skipping update")
                nan_detected = True
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
        processed_batches += 1
    
    # Handle case where all batches were skipped
    if processed_batches == 0:
        return float('nan'), 0.0
    
    avg_loss = total_loss / processed_batches
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, n_classes=None):
    """Evaluate model with comprehensive metrics.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss criterion
        device: Device to evaluate on
        n_classes: Number of classes (for compute_metrics)
        
    Returns:
        Tuple of (avg_loss, accuracy, all_preds, all_labels, metrics_dict)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    processed_batches = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            # Move data to device
            for modality in batch_data:
                if batch_data[modality] is not None:
                    batch_data[modality] = batch_data[modality].to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            logits, _ = model(batch_data)
            
            # Skip NaN outputs
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                continue
                
            loss = criterion(logits, batch_labels)
            
            # Skip NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            processed_batches += 1
    
    if processed_batches == 0:
        return float('nan'), 0.0, [], [], None
        
    avg_loss = total_loss / processed_batches
    accuracy = 100.0 * accuracy_score(all_labels, all_preds)
    
    # Compute comprehensive metrics if n_classes provided
    metrics_dict = None
    if n_classes is not None:
        metrics_dict = compute_metrics(all_labels, all_preds, n_classes)
    
    return avg_loss, accuracy, all_preds, all_labels, metrics_dict


def create_dataloader(data, labels, batch_size, shuffle=True):
    """Create PyTorch dataloader from numpy arrays."""
    
    class MultiOmicsDataset(torch.utils.data.Dataset):
        def __init__(self, data_dict, labels):
            self.data = {k: torch.from_numpy(v).float() for k, v in data_dict.items()}
            self.labels = torch.from_numpy(labels).long()
            self.n_samples = len(labels)
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            sample_data = {k: v[idx] for k, v in self.data.items()}
            return sample_data, self.labels[idx]
    
    dataset = MultiOmicsDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal transformer fusion model with cross-modal attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Train from scratch on raw data
  python train_transformer_fusion.py --data_dir data --num_epochs 50
  
  # Train with pretrained features (from extract_pretrained_features.py)
  python train_transformer_fusion.py --use_pretrained_features --pretrained_features_dir pretrained_features
  
  # With pretrained encoders (fine-tuning)
  python train_transformer_fusion.py --pretrained_encoders_dir pretrained/encoders --num_epochs 30
  
  # With frozen pretrained encoders (linear probing)
  python train_transformer_fusion.py --pretrained_encoders_dir pretrained/encoders --freeze_encoders
  
  # Select specific modalities
  python train_transformer_fusion.py --modalities GeneExpr miRNA Prot --num_epochs 50
  
  # Large model configuration
  python train_transformer_fusion.py --embed_dim 512 --num_heads 16 --num_layers 8
        """)
    
    # Data configuration
    data_args = parser.add_argument_group('data configuration')
    data_args.add_argument('--data_dir', type=str, default='final_processed_datasets',
                          help='Directory with parquet files (default: final_processed_datasets)')
    data_args.add_argument('--output_dir', type=str, default='transformer_models',
                          help='Output directory for trained model (default: transformer_models)')
    data_args.add_argument('--test_size', type=float, default=0.2,
                          help='Test set fraction (default: 0.2)')
    data_args.add_argument('--val_size', type=float, default=0.15,
                          help='Validation fraction taken from train split (default: 0.15)')
    data_args.add_argument('--no_standardize', action='store_true',
                          help='Disable feature standardization (not recommended)')
    
    # Pretrained features configuration
    feature_args = parser.add_argument_group('pretrained features')
    feature_args.add_argument('--use_pretrained_features', action='store_true',
                             help='Use pretrained embeddings instead of raw parquet data')
    feature_args.add_argument('--pretrained_features_dir', type=str, default=None,
                             help='Directory with *_embeddings.npy files from extract_pretrained_features.py')
    feature_args.add_argument('--modalities', type=str, nargs='+', 
                             default=None,
                             help='Modalities to use (default: all available). E.g., --modalities GeneExpr miRNA Prot')
    
    # Model architecture
    model_args = parser.add_argument_group('model architecture')
    model_args.add_argument('--embed_dim', type=int, default=256,
                           help='Embedding dimension, must be divisible by num_heads (default: 256)')
    model_args.add_argument('--num_heads', type=int, default=8,
                           help='Number of attention heads (default: 8)')
    model_args.add_argument('--num_layers', type=int, default=4,
                           help='Number of transformer layers (default: 4)')
    model_args.add_argument('--dropout', type=float, default=0.2,
                           help='Dropout probability for transformer and head (default: 0.2)')
    model_args.add_argument('--use_cls_token', action='store_true',
                           help='Use CLS-token pooling instead of flattening all modalities')
    model_args.add_argument('--pretrained_encoders_dir', type=str, default=None,
                           help='Directory with pretrained encoders (optional, for transfer learning)')
    model_args.add_argument('--freeze_encoders', action='store_true',
                           help='Freeze pretrained encoders (linear probing instead of fine-tuning)')
    
    # Training configuration
    train_args = parser.add_argument_group('training configuration')
    train_args.add_argument('--batch_size', type=int, default=32,
                           help='Training batch size (default: 32)')
    train_args.add_argument('--num_epochs', type=int, default=50,
                           help='Number of training epochs (default: 50)')
    train_args.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate (default: 0.001)')
    train_args.add_argument('--weight_decay', type=float, default=1e-2,
                           help='AdamW weight decay (default: 0.01)')
    train_args.add_argument('--label_smoothing', type=float, default=0.05,
                           help='Cross-entropy label smoothing (default: 0.05)')
    train_args.add_argument('--patience', type=int, default=10,
                           help='Early stopping patience on validation F1 (default: 10)')
    train_args.add_argument('--lr_patience', type=int, default=4,
                           help='ReduceLROnPlateau patience on validation loss (default: 4)')
    train_args.add_argument('--lr_factor', type=float, default=0.5,
                           help='ReduceLROnPlateau decay factor (default: 0.5)')
    train_args.add_argument('--min_lr', type=float, default=1e-6,
                           help='Minimum learning rate for scheduler (default: 1e-6)')
    train_args.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility (default: 42)')
    train_args.add_argument('--max_grad_norm', type=float, default=1.0,
                           help='Maximum gradient norm for clipping, 0 to disable (default: 1.0)')
    
    # Checkpoint configuration
    checkpoint_args = parser.add_argument_group('checkpoint configuration')
    checkpoint_args.add_argument('--checkpoint_interval', type=int, default=10,
                                help='Save checkpoint every N epochs (default: 10, 0 to disable)')
    checkpoint_args.add_argument('--keep_last_n', type=int, default=3,
                                help='Keep only last N checkpoints (default: 3)')
    checkpoint_args.add_argument('--resume', type=str, default=None,
                                help='Path to checkpoint file to resume training from')
    
    # System configuration
    system_args = parser.add_argument_group('system configuration')
    system_args.add_argument('--device', type=str,
                            default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='Device: cuda or cpu (default: cuda if available)')
    
    # Logging configuration
    log_args = parser.add_argument_group('logging configuration')
    log_args.add_argument('--use_wandb', action='store_true',
                         help='Enable Weights & Biases experiment tracking')
    log_args.add_argument('--wandb_project', type=str, default=None,
                         help='W&B project name')
    log_args.add_argument('--wandb_run_name', type=str, default=None,
                         help='W&B run name')
    
    # Label encoder configuration
    # Check ENCODER_DIR environment variable for consistency with other training scripts
    default_encoder_dir = os.environ.get('ENCODER_DIR', 'master_label_encoder')
    default_encoder_path = os.path.join(default_encoder_dir, 'label_encoder.joblib')
    encoder_args = parser.add_argument_group('label encoder configuration')
    encoder_args.add_argument('--label_encoder_path', type=str, 
                             default=default_encoder_path,
                             help=f'Path to master label encoder (default: {default_encoder_path})')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project or 'transformer-fusion',
                name=args.wandb_run_name or f'transformer_d{args.embed_dim}_h{args.num_heads}_l{args.num_layers}',
                config={
                    'embed_dim': args.embed_dim,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'batch_size': args.batch_size,
                    'num_epochs': args.num_epochs,
                    'lr': args.lr,
                    'seed': args.seed,
                    'max_grad_norm': args.max_grad_norm,
                    'device': args.device,
                    'freeze_encoders': args.freeze_encoders
                },
                reinit=True
            )
            print(f"W&B logging enabled: project={args.wandb_project}, run={wandb_run.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            wandb_run = None
    
    print("="*80)
    print("Multimodal Transformer Fusion Training")
    print("="*80)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Load master label encoder
    label_encoder = None
    label_encoder_path = Path(args.label_encoder_path)
    if label_encoder_path.exists():
        label_encoder = joblib.load(label_encoder_path)
        print(f"Loaded master label encoder from: {label_encoder_path}")
        print(f"  Classes: {list(label_encoder.classes_)}")
    else:
        print(f"Warning: Label encoder not found at {label_encoder_path}")
        print("  Run create_master_label_encoder.py first or provide --label_encoder_path")
    
    # Load data - either from pretrained features or raw parquet
    print("\nLoading data...")
    case_ids = None
    is_pretrained_split = False  # Track if pretrained features are already split
    
    if args.use_pretrained_features and args.pretrained_features_dir:
        print(f"Using pretrained features from: {args.pretrained_features_dir}")
        data, labels, modality_dims, case_ids, is_pretrained_split = load_pretrained_features(
            args.pretrained_features_dir, 
            modalities=args.modalities,
            label_encoder=label_encoder
        )
    else:
        print(f"Loading raw data from: {args.data_dir}")
        data, labels, modality_dims = load_multiomics_data(
            args.data_dir, 
            modalities=args.modalities,
            label_encoder=label_encoder
        )
    
    if not data or labels is None:
        print("Error: No data or labels found!")
        return
    
    print(f"\nLoaded {len(modality_dims)} modalities:")
    for modality, dim in modality_dims.items():
        print(f"  {modality}: {dim} features")
    
    # Handle split vs combined data
    if is_pretrained_split:
        # Data is already split for train/test; carve validation from train only.
        print(f"\nUsing pre-split data (train/test provided)")
        base_train_data = data['train']
        test_data = data['test']
        base_train_labels = labels['train']
        test_labels = labels['test']

        all_labels = np.concatenate([base_train_labels, test_labels])
        num_classes = len(np.unique(all_labels))

        train_indices = np.arange(len(base_train_labels))
        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=base_train_labels
        )
        train_data = {k: v[train_idx] for k, v in base_train_data.items()}
        val_data = {k: v[val_idx] for k, v in base_train_data.items()}
        train_labels = base_train_labels[train_idx]
        val_labels = base_train_labels[val_idx]
    else:
        # Legacy path: split into train/test, then split train into train/val.
        num_classes = len(np.unique(labels))
        print(f"\nNumber of classes: {num_classes}")

        print(f"\nSplitting data (test_size={args.test_size}, val_size={args.val_size}, seed={args.seed})...")
        indices = np.arange(len(labels))
        train_full_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels
        )

        train_full_data = {k: v[train_full_idx] for k, v in data.items()}
        train_full_labels = labels[train_full_idx]

        train_sub_idx, val_sub_idx = train_test_split(
            np.arange(len(train_full_labels)),
            test_size=args.val_size,
            random_state=args.seed,
            stratify=train_full_labels
        )

        train_data = {k: v[train_sub_idx] for k, v in train_full_data.items()}
        val_data = {k: v[val_sub_idx] for k, v in train_full_data.items()}
        train_labels = train_full_labels[train_sub_idx]
        val_labels = train_full_labels[val_sub_idx]
        test_data = {k: v[test_idx] for k, v in data.items()}
        test_labels = labels[test_idx]
    
    # Standardize AFTER split to avoid data leakage (fit on train only)
    if not args.no_standardize and not args.use_pretrained_features:
        from sklearn.preprocessing import StandardScaler
        print("\nStandardizing features (fit on training data only to avoid leakage)...")
        scalers = {}
        for modality in train_data:
            scaler = StandardScaler()
            train_data[modality] = scaler.fit_transform(train_data[modality]).astype(np.float32)
            val_data[modality] = scaler.transform(val_data[modality]).astype(np.float32)
            test_data[modality] = scaler.transform(test_data[modality]).astype(np.float32)
            scalers[modality] = scaler
            print(f"  {modality}: standardized (train fit, val/test transform)")
    
    total_samples = len(train_labels) + len(val_labels) + len(test_labels)
    print(f"\nTraining samples: {len(train_labels)} ({100*len(train_labels)/total_samples:.1f}%)")
    print(f"Validation samples: {len(val_labels)} ({100*len(val_labels)/total_samples:.1f}%)")
    print(f"Test samples: {len(test_labels)} ({100*len(test_labels)/total_samples:.1f}%)")
    
    # Log class distribution
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    print(f"Train class distribution: {dict(zip(unique_train.tolist(), counts_train.tolist()))}")
    print(f"Val class distribution: {dict(zip(unique_val.tolist(), counts_val.tolist()))}")
    print(f"Test class distribution: {dict(zip(unique_test.tolist(), counts_test.tolist()))}")
    
    # Create dataloaders
    train_loader = create_dataloader(train_data, train_labels, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, val_labels, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, test_labels, args.batch_size, shuffle=False)
    
    # Load pretrained encoders if provided
    pretrained_encoders = None
    if args.pretrained_encoders_dir:
        print(f"\nLoading pretrained encoders from {args.pretrained_encoders_dir}...")
        pretrained_encoders, metadata = load_pretrained_encoders(args.pretrained_encoders_dir)
        print(f"Loaded {len(pretrained_encoders)} pretrained encoders")
        
        # Verify embed_dim matches
        if metadata['embed_dim'] != args.embed_dim:
            print(f"Warning: Pretrained embed_dim ({metadata['embed_dim']}) != specified ({args.embed_dim})")
            print(f"Using pretrained embed_dim: {metadata['embed_dim']}")
            args.embed_dim = metadata['embed_dim']
    
    # Create model
    print(f"\nInitializing multimodal transformer...")
    model = MultimodalFusionClassifier(
        modality_dims=modality_dims,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        use_cls_token=args.use_cls_token,
        pretrained_encoders=pretrained_encoders
    )
    
    # Freeze encoders if requested
    if args.freeze_encoders and pretrained_encoders is not None:
        print("Freezing pretrained encoders (linear probing mode)")
        for param in model.encoders.parameters():
            param.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    device = torch.device(args.device)
    model.to(device)

    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.from_numpy(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=args.label_smoothing
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr
    )
    
    print(f"\nUsing device: {device}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1_weighted': [], 'lr': []}
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('val_acc', 0)
            best_val_f1 = checkpoint.get('val_f1_weighted', best_val_f1)
            print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        else:
            print(f"Warning: Checkpoint not found at {resume_path}, starting from scratch")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            max_grad_norm=args.max_grad_norm
        )
        
        # Evaluate on validation split (keeps test set untouched)
        val_loss, val_acc, _, _, val_metrics = evaluate(model, val_loader, criterion, device, n_classes=num_classes)
        val_f1_weighted = val_metrics['f1_weighted'] if val_metrics is not None else 0.0
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_weighted'].append(val_f1_weighted)
        history['lr'].append(current_lr)
        
        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1_weighted': val_f1_weighted,
                'lr': current_lr,
                'best_val_f1_weighted': best_val_f1,
                'best_val_acc': best_val_acc
            })
        
        # Print progress
        print(f"Epoch [{epoch+1}/{args.num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% Val F1(w): {val_f1_weighted:.4f} | LR: {current_lr:.2e}")
        
        # Save best model based on weighted F1, with accuracy as tie-breaker
        improved = (val_f1_weighted > best_val_f1 + 1e-6) or (
            abs(val_f1_weighted - best_val_f1) <= 1e-6 and val_acc > best_val_acc
        )
        if improved:
            best_val_acc = val_acc
            best_val_f1 = val_f1_weighted
            epochs_without_improvement = 0
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            
            # Save config.json separately for extract_transformer_features.py
            config = {
                'modality_dims': modality_dims,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'num_classes': num_classes,
                'dropout': args.dropout,
                'use_cls_token': args.use_cls_token,
                'epoch': epoch,
                'val_f1_weighted': float(val_f1_weighted),
                'val_acc': float(val_acc),
                'seed': args.seed,
                'test_size': args.test_size,
                'val_size': args.val_size,
                'weight_decay': args.weight_decay,
                'label_smoothing': args.label_smoothing,
                'patience': args.patience,
                'max_grad_norm': args.max_grad_norm,
                'label_encoder_path': str(args.label_encoder_path),
                'label_classes': list(label_encoder.classes_) if label_encoder else None
            }
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1} (patience={args.patience})")
            break
        
        # Periodic checkpointing
        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1_weighted': val_f1_weighted,
                'val_acc': val_acc,
                'config': {
                    'modality_dims': modality_dims,
                    'embed_dim': args.embed_dim,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'num_classes': num_classes
                }
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Keep only last N checkpoints
            if args.keep_last_n > 0:
                checkpoints = sorted(output_dir.glob('checkpoint_epoch_*.pt'))
                while len(checkpoints) > args.keep_last_n:
                    checkpoints[0].unlink()
                    checkpoints = checkpoints[1:]
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80 + "\n")

    best_model_path = Path(args.output_dir) / 'best_model.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best validation checkpoint from: {best_model_path}")
    else:
        print("Warning: best_model.pt not found; evaluating last epoch weights")
    
    val_loss, val_acc, val_preds, val_labels, metrics_dict = evaluate(
        model, test_loader, criterion, device, n_classes=num_classes
    )
    
    print(f"Test Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {val_acc:.2f}%")
    
    # Print comprehensive metrics
    if metrics_dict is not None:
        print(f"\nComprehensive Metrics:")
        print(f"  Precision (macro): {metrics_dict['precision_macro']:.4f}")
        print(f"  Precision (weighted): {metrics_dict['precision_weighted']:.4f}")
        print(f"  Recall (macro): {metrics_dict['recall_macro']:.4f}")
        print(f"  Recall (weighted): {metrics_dict['recall_weighted']:.4f}")
        print(f"  F1 (macro): {metrics_dict['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics_dict['f1_weighted']:.4f}")
        print(f"  Specificity (macro): {metrics_dict['specificity_macro']:.4f}")
        print(f"  Specificity (weighted): {metrics_dict['specificity_weighted']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(val_labels, val_preds))
    
    # Log final metrics to wandb
    if wandb_run is not None:
        wandb_log_dict = {
            'final_test_loss': val_loss,
            'final_test_acc': val_acc,
            'best_val_f1_weighted': best_val_f1,
            'best_val_acc': best_val_acc
        }
        if metrics_dict is not None:
            wandb_log_dict.update({
                'final_precision_macro': metrics_dict['precision_macro'],
                'final_precision_weighted': metrics_dict['precision_weighted'],
                'final_recall_macro': metrics_dict['recall_macro'],
                'final_recall_weighted': metrics_dict['recall_weighted'],
                'final_f1_macro': metrics_dict['f1_macro'],
                'final_f1_weighted': metrics_dict['f1_weighted'],
                'final_specificity_macro': metrics_dict['specificity_macro'],
                'final_specificity_weighted': metrics_dict['specificity_weighted']
            })
        wandb_run.log(wandb_log_dict)
        wandb_run.finish()
    
    # Save training history with comprehensive metrics
    output_dir = Path(args.output_dir)
    history_path = output_dir / "training_history.json"
    
    # Add final metrics to history
    if metrics_dict is not None:
        history['final_metrics'] = {
            'accuracy': metrics_dict['accuracy'],
            'precision_macro': metrics_dict['precision_macro'],
            'precision_weighted': metrics_dict['precision_weighted'],
            'recall_macro': metrics_dict['recall_macro'],
            'recall_weighted': metrics_dict['recall_weighted'],
            'f1_macro': metrics_dict['f1_macro'],
            'f1_weighted': metrics_dict['f1_weighted'],
            'specificity_macro': metrics_dict['specificity_macro'],
            'specificity_weighted': metrics_dict['specificity_weighted']
        }
        # Save confusion matrix separately
        cm = metrics_dict['confusion_matrix']
        cm_path = output_dir / 'confusion_matrix.csv'
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Save normalized confusion matrix (per-row / true-class)
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_sum[cm_sum == 0] = 1  # avoid division by zero
        cm_normalized = cm.astype(float) / cm_sum
        cmn_path = output_dir / 'confusion_matrix_normalized.csv'
        pd.DataFrame(cm_normalized).to_csv(cmn_path, index=False)
        print(f"Normalized confusion matrix saved to: {cmn_path}")
        
        # Save standalone test_metrics.json for consistency with other scripts
        test_metrics = {
            'accuracy': metrics_dict['accuracy'],
            'precision_macro': metrics_dict['precision_macro'],
            'precision_weighted': metrics_dict['precision_weighted'],
            'recall_macro': metrics_dict['recall_macro'],
            'recall_weighted': metrics_dict['recall_weighted'],
            'f1_macro': metrics_dict['f1_macro'],
            'f1_weighted': metrics_dict['f1_weighted'],
            'specificity_macro': metrics_dict['specificity_macro'],
            'specificity_weighted': metrics_dict['specificity_weighted']
        }
        metrics_path = output_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test metrics saved to: {metrics_path}")
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModel saved to: {output_dir / 'best_model.pt'}")
    print(f"Training history saved to: {history_path}")
    print(f"Best validation weighted F1: {best_val_f1:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
