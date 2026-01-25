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


def load_multiomics_data(data_dir, modalities=None):
    """
    Load multi-omics data from parquet files.
    
    Args:
        data_dir: Directory containing data files
        modalities: List of modality names to load (if None, load all available)
        
    Returns:
        Tuple of (data_dict, labels, modality_dims)
    """
    data_dir = Path(data_dir)
    
    if modalities is None:
        modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot', 'SNV']
    
    data = {}
    modality_dims = {}
    labels = None
    
    for modality in modalities:
        file_path = data_dir / f"data_{modality}_.parquet"
        
        if file_path.exists():
            print(f"Loading {modality} from {file_path}")
            df = pd.read_parquet(file_path)
            
            # Extract features (exclude 'class' and 'split' columns)
            feature_cols = [col for col in df.columns if col not in ['class', 'split']]
            features = df[feature_cols].values.astype(np.float32)
            
            data[modality] = features
            modality_dims[modality] = features.shape[1]
            
            # Extract labels (from first modality encountered)
            if labels is None and 'class' in df.columns:
                labels = df['class'].values
            
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
    
    for batch_data, batch_labels in dataloader:
        # Move data to device
        for modality in batch_data:
            if batch_data[modality] is not None:
                batch_data[modality] = batch_data[modality].to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        logits, _ = model(batch_data)
        loss = criterion(logits, batch_labels)
        
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
    
    avg_loss = total_loss / len(dataloader)
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
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            # Move data to device
            for modality in batch_data:
                if batch_data[modality] is not None:
                    batch_data[modality] = batch_data[modality].to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            logits, _ = model(batch_data)
            loss = criterion(logits, batch_labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
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
  # Train from scratch
  python train_transformer_fusion.py --data_dir data --num_epochs 50
  
  # With pretrained encoders (fine-tuning)
  python train_transformer_fusion.py --pretrained_encoders_dir pretrained/encoders --num_epochs 30
  
  # With frozen pretrained encoders (linear probing)
  python train_transformer_fusion.py --pretrained_encoders_dir pretrained/encoders --freeze_encoders
  
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
    
    # Model architecture
    model_args = parser.add_argument_group('model architecture')
    model_args.add_argument('--embed_dim', type=int, default=256,
                           help='Embedding dimension, must be divisible by num_heads (default: 256)')
    model_args.add_argument('--num_heads', type=int, default=8,
                           help='Number of attention heads (default: 8)')
    model_args.add_argument('--num_layers', type=int, default=4,
                           help='Number of transformer layers (default: 4)')
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
    
    # Load data
    print("Loading multi-omics data...")
    data, labels, modality_dims = load_multiomics_data(args.data_dir)
    
    if not data or labels is None:
        print("Error: No data or labels found!")
        return
    
    print(f"\nLoaded {len(data)} modalities:")
    for modality, dim in modality_dims.items():
        print(f"  {modality}: {dim} features")
    
    # Determine number of classes
    num_classes = len(np.unique(labels))
    print(f"\nNumber of classes: {num_classes}")
    
    # Split data
    print(f"\nSplitting data (test_size={args.test_size})...")
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=42,
        stratify=labels
    )
    
    train_data = {k: v[train_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    
    # Create dataloaders
    train_loader = create_dataloader(train_data, train_labels, args.batch_size, shuffle=True)
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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    print(f"\nUsing device: {device}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('val_acc', 0)
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
        
        # Evaluate (skip comprehensive metrics during training for speed)
        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            })
        
        # Print progress
        print(f"Epoch [{epoch+1}/{args.num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
                'dropout': 0.1,
                'epoch': epoch,
                'val_acc': float(val_acc)
            }
            with open(output_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
        
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
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
