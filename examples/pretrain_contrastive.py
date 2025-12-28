#!/usr/bin/env python
"""
Example script for self-supervised contrastive pretraining.

This script demonstrates Option 2: Self-Supervised Contrastive Pretraining
as described in PERFORMANCE_EXTENSIONS.md.

Usage:
    python examples/pretrain_contrastive.py --data_dir /path/to/data --output_dir pretrained_models
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from performance_extensions.contrastive_learning import (
    ContrastiveMultiOmicsEncoder,
    ContrastiveLearningLoss
)
from performance_extensions.training_utils import (
    MultiOmicsDataset,
    pretrain_contrastive,
    save_pretrained_encoders
)


def load_multiomics_data(data_dir):
    """
    Load multi-omics data from parquet files.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dict mapping modality names to numpy arrays
    """
    data_dir = Path(data_dir)
    
    modalities = ['GeneExp', 'miRNA', 'Meth', 'CNV', 'Prot', 'Mut']
    data = {}
    modality_dims = {}
    
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
            
            print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
        else:
            print(f"Warning: {file_path} not found, skipping {modality}")
    
    return data, modality_dims


def main():
    parser = argparse.ArgumentParser(
        description="Self-supervised contrastive pretraining for multi-omics data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Basic pretraining with defaults
  python pretrain_contrastive.py --data_dir final_processed_datasets
  
  # With cross-modal contrastive learning
  python pretrain_contrastive.py --data_dir data --use_cross_modal --temperature 0.07
  
  # Large batch training on GPU
  python pretrain_contrastive.py --batch_size 128 --num_epochs 200 --device cuda
        """)
    
    # Data configuration
    data_args = parser.add_argument_group('data configuration')
    data_args.add_argument('--data_dir', type=str, default='final_processed_datasets',
                          help='Directory with parquet data files (default: final_processed_datasets)')
    data_args.add_argument('--output_dir', type=str, default='pretrained_models/contrastive',
                          help='Output directory for pretrained encoders (default: pretrained_models/contrastive)')
    
    # Model architecture
    model_args = parser.add_argument_group('model architecture')
    model_args.add_argument('--embed_dim', type=int, default=256,
                           help='Embedding dimension for all modalities (default: 256)')
    model_args.add_argument('--projection_dim', type=int, default=128,
                           help='Projection head dimension for contrastive loss (default: 128)')
    
    # Training configuration
    train_args = parser.add_argument_group('training configuration')
    train_args.add_argument('--batch_size', type=int, default=32,
                           help='Training batch size (default: 32, use 64-128 for GPU)')
    train_args.add_argument('--num_epochs', type=int, default=100,
                           help='Number of pretraining epochs (default: 100)')
    train_args.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate (default: 0.001)')
    
    # Contrastive learning
    contrast_args = parser.add_argument_group('contrastive learning')
    contrast_args.add_argument('--temperature', type=float, default=0.5,
                              help='Temperature for NT-Xent loss (lower=harder negatives, default: 0.5)')
    contrast_args.add_argument('--use_cross_modal', action='store_true',
                              help='Enable cross-modal contrastive loss (learns relationships between modalities)')
    
    # System configuration
    system_args = parser.add_argument_group('system configuration')
    system_args.add_argument('--device', type=str, 
                            default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='Device: cuda or cpu (default: cuda if available)')
    system_args.add_argument('--checkpoint_interval', type=int, default=10,
                            help='Save checkpoint every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Self-Supervised Contrastive Pretraining for Multi-Omics Data")
    print("="*80)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Load data
    print("Loading multi-omics data...")
    data, modality_dims = load_multiomics_data(args.data_dir)
    
    if not data:
        print("Error: No data files found!")
        return
    
    print(f"\nLoaded {len(data)} modalities:")
    for modality, dim in modality_dims.items():
        print(f"  {modality}: {dim} features")
    
    # Create dataset (no labels needed for pretraining)
    print("\nCreating dataset with augmentation...")
    dataset = MultiOmicsDataset(
        data=data,
        labels=None,  # No labels needed for pretraining
        apply_augmentation=True,
        num_augmented_views=2
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid pickling issues
        drop_last=True  # Drop last incomplete batch
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    # Create model
    print(f"\nInitializing contrastive encoder (embed_dim={args.embed_dim})...")
    model = ContrastiveMultiOmicsEncoder(
        modality_dims=modality_dims,
        embed_dim=args.embed_dim,
        projection_dim=args.projection_dim
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = ContrastiveLearningLoss(
        temperature=args.temperature,
        use_cross_modal=args.use_cross_modal
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Pretrain
    print("\n" + "="*80)
    print("Starting Pretraining")
    print("="*80 + "\n")
    
    metrics = pretrain_contrastive(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=args.num_epochs,
        log_interval=10,
        checkpoint_dir=checkpoint_dir,
        verbose=True
    )
    
    # Save final pretrained encoders
    print("\n" + "="*80)
    print("Saving Pretrained Encoders")
    print("="*80 + "\n")
    
    output_dir = Path(args.output_dir) / "encoders"
    save_pretrained_encoders(
        model=model,
        save_dir=output_dir,
        metadata={
            'embed_dim': args.embed_dim,
            'projection_dim': args.projection_dim,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'temperature': args.temperature,
            'use_cross_modal': args.use_cross_modal,
            'final_loss': metrics['epoch_losses'][-1]
        }
    )
    
    # Save training metrics
    import json
    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'epoch_losses': metrics['epoch_losses'],
            'config': vars(args)
        }, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot training curve
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch_losses'])
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.title('Pretraining Loss Curve')
        plt.grid(True)
        
        plot_path = Path(args.output_dir) / "loss_curve.png"
        plt.savefig(plot_path)
        print(f"Loss curve saved to {plot_path}")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    
    print("\n" + "="*80)
    print("Pretraining Complete!")
    print("="*80)
    print(f"\nPretrained encoders saved to: {output_dir}")
    print(f"Final loss: {metrics['epoch_losses'][-1]:.4f}")
    print(f"\nTo use these pretrained encoders, load them with:")
    print(f"  from performance_extensions.training_utils import load_pretrained_encoders")
    print(f"  encoders, metadata = load_pretrained_encoders('{output_dir}')")


if __name__ == '__main__':
    main()
