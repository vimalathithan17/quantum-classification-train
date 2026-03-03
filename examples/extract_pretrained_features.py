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
    
    If the encoder was trained with train/test split (no leakage mode), this will
    save train and test embeddings separately.
    
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
    
    # Check for train/test split info (from pretrain_contrastive.py --test_size)
    # Split files are saved in encoder's parent directory
    split_dir = encoder_dir.parent
    train_indices_file = split_dir / 'train_indices.npy'
    test_indices_file = split_dir / 'test_indices.npy'
    scalers_file = split_dir / 'scalers.joblib'
    
    has_split = train_indices_file.exists() and test_indices_file.exists()
    train_indices = None
    test_indices = None
    
    if has_split:
        train_indices = np.load(train_indices_file)
        test_indices = np.load(test_indices_file)
        print(f"\\n✓ Found train/test split info (no data leakage)")
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Test samples: {len(test_indices)}")
        print(f"  Will save train/test embeddings separately")
    else:
        print(f"\\n⚠️  WARNING: No train/test split info found!")
        print(f"   Encoder may have been trained on ALL data (potential leakage)")
        print(f"   Recommend re-running pretrain_contrastive.py with --test_size 0.2")
    
    # Load scalers if available (required if encoder was trained on standardized data)
    scalers = None
    if scalers_file.exists():
        import joblib
        scalers = joblib.load(scalers_file)
        print(f"\\n✓ Loaded scalers from pretraining (will standardize input data)")
        print(f"  Scalers available for: {list(scalers.keys())}")
    else:
        print(f"\\n⚠️  WARNING: No scalers found at {scalers_file}")
        print(f"   If encoder was trained on standardized data, features may be incorrect!")
    
    # Store extraction metadata
    extraction_info = {
        'source_encoder_dir': str(encoder_dir),
        'source_data_dir': str(data_dir),
        'embed_dim': metadata['embed_dim'],
        'modalities_extracted': [],
        'samples_per_modality': {},
        'has_train_test_split': has_split,
        'n_train_samples': len(train_indices) if train_indices is not None else None,
        'n_test_samples': len(test_indices) if test_indices is not None else None,
        'scalers_applied': scalers is not None
    }
    
    labels_saved = False
    case_ids_saved = False
    
    for modality in metadata['modality_names']:
        print(f"\\nProcessing {modality}...")
        
        # Check if data file exists
        data_file = data_dir / f"data_{modality}_.parquet"
        if not data_file.exists():
            print(f"  Warning: {data_file} not found, skipping")
            continue
        
        # Load data
        features, labels, case_ids = load_modality_data(data_dir, modality)
        n_samples = features.shape[0]
        print(f"  Loaded {n_samples} samples with {features.shape[1]} features")
        
        # Apply scaler if available (must use same standardization as training)
        if scalers is not None and modality in scalers:
            features = scalers[modality].transform(features).astype(np.float32)
            print(f"  Applied standardization from pretraining")
        elif scalers is not None:
            print(f"  Warning: No scaler for {modality}, using raw features")
        
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
        
        if has_split:
            # Save train/test embeddings separately
            train_emb = embeddings[train_indices]
            test_emb = embeddings[test_indices]
            
            train_file = output_dir / f"{modality}_train_embeddings.npy"
            test_file = output_dir / f"{modality}_test_embeddings.npy"
            np.save(train_file, train_emb)
            np.save(test_file, test_emb)
            print(f"  Saved train embeddings: {train_emb.shape} -> {train_file}")
            print(f"  Saved test embeddings: {test_emb.shape} -> {test_file}")
            
            # Also save combined (for backward compat) but mark it as potentially leaky
            all_file = output_dir / f"{modality}_embeddings.npy"
            np.save(all_file, embeddings)
            print(f"  Saved all embeddings (use train/test files for proper evaluation): {all_file}")
        else:
            # No split info - save as single file
            output_file = output_dir / f"{modality}_embeddings.npy"
            np.save(output_file, embeddings)
            print(f"  Saved embeddings: {embeddings.shape} -> {output_file}")
        
        extraction_info['modalities_extracted'].append(modality)
        extraction_info['samples_per_modality'][modality] = n_samples
        
        # Save labels (handle split if present)
        if labels is not None and not labels_saved:
            if has_split:
                np.save(output_dir / "train_labels.npy", labels[train_indices])
                np.save(output_dir / "test_labels.npy", labels[test_indices])
                np.save(output_dir / "labels.npy", labels)
                print(f"  Saved train/test/all labels")
            else:
                np.save(output_dir / "labels.npy", labels)
                print(f"  Saved labels")
            labels_saved = True
        
        # Save case_ids (handle split if present)
        if case_ids is not None and not case_ids_saved:
            if has_split:
                np.save(output_dir / "train_case_ids.npy", case_ids[train_indices])
                np.save(output_dir / "test_case_ids.npy", case_ids[test_indices])
                np.save(output_dir / "case_ids.npy", case_ids)
                print(f"  Saved train/test/all case_ids")
            else:
                np.save(output_dir / "case_ids.npy", case_ids)
                print(f"  Saved case_ids")
            case_ids_saved = True
    
    # Save extraction metadata
    metadata_file = output_dir / "extraction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(extraction_info, f, indent=2)
    print(f"\\nSaved extraction metadata -> {metadata_file}")
    
    print(f"\\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Extracted {len(extraction_info['modalities_extracted'])} modalities")
    print(f"Output directory: {output_dir}")
    print(f"Embedding dimension: {metadata['embed_dim']}")
    if has_split:
        print(f"\\n✓ Train/test split preserved (no leakage)")
        print(f"  Use *_train_embeddings.npy for training")
        print(f"  Use *_test_embeddings.npy for evaluation")
    else:
        print(f"\\n⚠️  No split - ALL samples were encoded (potential leakage)")
    print(f"{'='*60}")
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
