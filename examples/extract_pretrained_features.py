#!/usr/bin/env python
"""
Extract features from pretrained contrastive encoders for use in QML pipeline.

This script takes pretrained modality encoders and extracts embeddings that can
be used as input to the QML models (replacing PCA/UMAP dimensionality reduction).

Usage:
    python examples/extract_pretrained_features.py \
        --encoder_dir pretrained_models/contrastive/encoders \
        --data_dir final_processed_datasets \
        --output_dir pretrained_features

Example integration with QML pipeline:
    # Step 1: Pretrain encoders
    python examples/pretrain_contrastive.py --data_dir data --output_dir pretrained_models
    
    # Step 2: Extract features
    python examples/extract_pretrained_features.py \
        --encoder_dir pretrained_models/contrastive/encoders \
        --data_dir data \
        --output_dir pretrained_features
    
    # Step 3: Use features in QML (features will be in pretrained_features/*.npy)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from performance_extensions.training_utils import load_pretrained_encoders


def load_modality_data(data_dir: Path, modality: str) -> tuple:
    """
    Load a single modality's data from parquet file.
    
    Args:
        data_dir: Directory containing parquet files
        modality: Name of the modality (e.g., 'GeneExpr', 'miRNA')
        
    Returns:
        Tuple of (features, labels, case_ids) where features is np.ndarray
    """
    file_path = data_dir / f"data_{modality}_.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # CRITICAL: Sort by case_id for consistent ordering across all scripts
    if 'case_id' in df.columns:
        df = df.sort_values('case_id')
    
    # Metadata columns to exclude from features (only case_id and class exist in the data)
    METADATA_COLS = {'class', 'case_id'}
    
    # Extract features (exclude metadata columns)
    feature_cols = [col for col in df.columns if col not in METADATA_COLS]
    features = df[feature_cols].values.astype(np.float32)
    
    # Extract labels if available
    labels = df['class'].values if 'class' in df.columns else None
    
    # Extract case_id if available
    case_ids = df['case_id'].values if 'case_id' in df.columns else None
    
    return features, labels, case_ids


def extract_features(
    encoder_dir: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 256,
    device: str = 'auto'
):
    """
    Extract embeddings from pretrained encoders for all available modalities.
    
    Args:
        encoder_dir: Directory containing pretrained encoders
        data_dir: Directory containing parquet data files
        output_dir: Output directory for extracted features
        batch_size: Batch size for encoding (for memory efficiency)
        device: Device to use ('auto', 'cpu', 'cuda')
    """
    encoder_dir = Path(encoder_dir)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load pretrained encoders
    encoders, metadata = load_pretrained_encoders(encoder_dir, device)
    
    # Store extraction metadata
    extraction_info = {
        'source_encoder_dir': str(encoder_dir),
        'source_data_dir': str(data_dir),
        'embed_dim': metadata['embed_dim'],
        'modalities_extracted': [],
        'samples_per_modality': {}
    }
    
    labels_saved = False
    case_ids_saved = False
    
    for modality in metadata['modality_names']:
        print(f"\nProcessing {modality}...")
        
        # Check if data file exists
        data_file = data_dir / f"data_{modality}_.parquet"
        if not data_file.exists():
            print(f"  Warning: {data_file} not found, skipping")
            continue
        
        # Load data
        features, labels, case_ids = load_modality_data(data_dir, modality)
        n_samples = features.shape[0]
        print(f"  Loaded {n_samples} samples with {features.shape[1]} features")
        
        # Get encoder and set to eval mode
        encoder = encoders[modality]
        encoder.eval()
        encoder.to(device)
        
        # Extract embeddings in batches (for memory efficiency)
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = features[i:i + batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(device)
                
                result = encoder(batch_tensor)
                # Handle tuple return (embedding, valid_mask) from contrastive_learning encoders
                if isinstance(result, tuple):
                    embeddings = result[0]
                else:
                    embeddings = result
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        output_file = output_dir / f"{modality}_embeddings.npy"
        np.save(output_file, embeddings)
        print(f"  Saved embeddings: {embeddings.shape} -> {output_file}")
        
        extraction_info['modalities_extracted'].append(modality)
        extraction_info['samples_per_modality'][modality] = n_samples
        
        # Save labels once (should be same across modalities)
        if labels is not None and not labels_saved:
            labels_file = output_dir / "labels.npy"
            np.save(labels_file, labels)
            print(f"  Saved labels -> {labels_file}")
            labels_saved = True
        
        # Save case_ids once (for sample identification)
        if case_ids is not None and not case_ids_saved:
            case_ids_file = output_dir / "case_ids.npy"
            np.save(case_ids_file, case_ids)
            print(f"  Saved case_ids -> {case_ids_file}")
            case_ids_saved = True
    
    # Save extraction metadata
    metadata_file = output_dir / "extraction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(extraction_info, f, indent=2)
    print(f"\nSaved extraction metadata -> {metadata_file}")
    
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Extracted {len(extraction_info['modalities_extracted'])} modalities")
    print(f"Output directory: {output_dir}")
    print(f"Embedding dimension: {metadata['embed_dim']}")
    print(f"{'='*60}")
    
    return extraction_info


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from pretrained contrastive encoders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Basic extraction
  python extract_pretrained_features.py \\
      --encoder_dir pretrained_models/contrastive/encoders \\
      --data_dir final_processed_datasets \\
      --output_dir pretrained_features
  
  # Using GPU with larger batches
  python extract_pretrained_features.py \\
      --encoder_dir pretrained_models/encoders \\
      --data_dir data \\
      --output_dir features \\
      --batch_size 512 \\
      --device cuda

Output files:
  <output_dir>/
  ├── GeneExpr_embeddings.npy     # Shape: (N, embed_dim)
  ├── miRNA_embeddings.npy       # Shape: (N, embed_dim)
  ├── Meth_embeddings.npy        # Shape: (N, embed_dim)
  ├── ...
  ├── labels.npy                 # Shape: (N,) - class labels
  ├── case_ids.npy               # Shape: (N,) - sample identifiers
  └── extraction_metadata.json   # Extraction configuration
        """)
    
    parser.add_argument('--encoder_dir', type=str, required=True,
                        help='Directory containing pretrained encoders (with metadata.json)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing parquet data files (data_*.parquet)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for extracted features')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for encoding (default: 256)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    extract_features(
        encoder_dir=args.encoder_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
