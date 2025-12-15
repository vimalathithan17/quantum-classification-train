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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from performance_extensions.transformer_fusion import MultimodalFusionClassifier
from performance_extensions.training_utils import load_pretrained_encoders


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
        modalities = ['GeneExp', 'miRNA', 'Meth', 'CNV', 'Prot', 'Mut']
    
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


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
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
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
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
    
    return avg_loss, accuracy, all_preds, all_labels


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
        description="Multimodal transformer fusion training"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='final_processed_datasets',
        help='Directory containing parquet data files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='transformer_models',
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--pretrained_encoders_dir',
        type=str,
        default=None,
        help='Directory containing pretrained encoders (optional)'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=256,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=4,
        help='Number of transformer layers'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--freeze_encoders',
        action='store_true',
        help='Freeze pretrained encoders (linear probing)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size (fraction)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
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
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{args.num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(args)
            }, output_dir / 'best_model.pt')
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80 + "\n")
    
    val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {val_acc:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(val_labels, val_preds))
    
    # Save training history
    import json
    output_dir = Path(args.output_dir)
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModel saved to: {output_dir / 'best_model.pt'}")
    print(f"Training history saved to: {history_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
