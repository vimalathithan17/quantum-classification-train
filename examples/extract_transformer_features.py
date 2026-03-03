#!/usr/bin/env python
"""
Extract transformer predictions/features for use as QML meta-learner input.

This script takes a trained multimodal transformer and extracts its predictions
or penultimate layer features, which can then be used as input to the QML 
meta-learner for hybrid transformer+quantum classification.

Usage:
    python examples/extract_transformer_features.py \
        --model_dir transformer_models \
        --data_dir final_processed_datasets \
        --output_dir transformer_predictions

Example integration with QML meta-learner:
    # Step 1: Train transformer
    python examples/train_transformer_fusion.py --data_dir data --output_dir transformer_models
    
    # Step 2: Extract predictions
    python examples/extract_transformer_features.py \
        --model_dir transformer_models \
        --data_dir data \
        --output_dir transformer_predictions
    
    # Step 3: Use in QML meta-learner
    python metalearner.py --preds_dir transformer_predictions
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import json

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from performance_extensions.transformer_fusion import MultimodalFusionClassifier


def load_multiomics_data(data_dir: Path, modalities: list = None) -> tuple:
    """
    Load multi-omics data from parquet files.
    
    Args:
        data_dir: Directory containing parquet files
        modalities: List of modalities to load (default: all available)
        
    Returns:
        Tuple of (data_dict, labels, modality_dims, case_ids)
    """
    if modalities is None:
        modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot', 'SNV']
    
    data = {}
    modality_dims = {}
    labels = None
    case_ids = None
    
    for modality in modalities:
        file_path = data_dir / f"data_{modality}_.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            
            # CRITICAL: Sort by case_id for consistent ordering across all scripts
            if 'case_id' in df.columns:
                df = df.sort_values('case_id')
            
            # Metadata columns to exclude from features (only case_id and class exist in the data)
            METADATA_COLS = {'class', 'case_id'}
            
            # Extract features (exclude metadata columns)
            feature_cols = [col for col in df.columns if col not in METADATA_COLS]
            features = df[feature_cols].values.astype(np.float32)
            
            data[modality] = features
            modality_dims[modality] = features.shape[1]
            
            # Get labels and case_ids from first modality
            if labels is None and 'class' in df.columns:
                labels = df['class'].values
                # If labels are strings, encode them to integers
                if labels.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    labels = le.fit_transform(labels)
                    print(f"  Encoded string labels to integers: {le.classes_}")
                labels = labels.astype(np.int64)
            if case_ids is None:
                # Try to get case_id from column or index
                if 'case_id' in df.columns:
                    case_ids = df['case_id'].values
                elif df.index.name == 'case_id' or 'case' in str(df.index.name).lower():
                    case_ids = df.index.values
                else:
                    # Generate synthetic case_ids if not found
                    case_ids = np.array([f"sample_{i}" for i in range(len(df))])
    
    return data, labels, modality_dims, case_ids


def load_pretrained_embeddings(features_dir: Path, modalities: list = None) -> tuple:
    """
    Load pretrained features from extracted embeddings.
    
    Supports two formats:
    1. Split format (preferred, no leakage): {modality}_train_embeddings.npy, {modality}_test_embeddings.npy
    2. Combined format (legacy): {modality}_embeddings.npy
    
    Args:
        features_dir: Directory containing *_embeddings.npy files
        modalities: List of modalities to load (default: all available)
        
    Returns:
        Tuple of (data_dict, labels, modality_dims, case_ids, is_split)
        - If is_split=True: data_dict contains 'train' and 'test' keys
        - If is_split=False: data_dict contains modality keys directly
    """
    if modalities is None:
        modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot', 'SNV']
    
    # Check if split files exist
    first_modality = modalities[0]
    train_file_check = features_dir / f"{first_modality}_train_embeddings.npy"
    test_file_check = features_dir / f"{first_modality}_test_embeddings.npy"
    has_split_files = train_file_check.exists() and test_file_check.exists()
    
    if has_split_files:
        # ✓ Properly split pretrained features
        print("Loading SPLIT pretrained features (no leakage)")
        
        train_data = {}
        test_data = {}
        modality_dims = {}
        
        # Load train/test labels and case_ids
        train_labels = np.load(features_dir / 'train_labels.npy', allow_pickle=True)
        test_labels = np.load(features_dir / 'test_labels.npy', allow_pickle=True)
        
        train_case_ids = None
        test_case_ids = None
        if (features_dir / 'train_case_ids.npy').exists():
            train_case_ids = np.load(features_dir / 'train_case_ids.npy', allow_pickle=True)
            test_case_ids = np.load(features_dir / 'test_case_ids.npy', allow_pickle=True)
        
        # Encode labels if string
        if train_labels.dtype.kind in ('U', 'S', 'O'):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(np.concatenate([train_labels, test_labels]))
            train_labels = le.transform(train_labels)
            test_labels = le.transform(test_labels)
            print(f"Encoded string labels to integers: {le.classes_}")
        train_labels = train_labels.astype(np.int64)
        test_labels = test_labels.astype(np.int64)
        
        print(f"Train: {len(train_labels)} samples, Test: {len(test_labels)} samples")
        
        # Load embeddings for each modality
        for modality in modalities:
            train_file = features_dir / f"{modality}_train_embeddings.npy"
            test_file = features_dir / f"{modality}_test_embeddings.npy"
            
            if train_file.exists() and test_file.exists():
                print(f"Loading pretrained {modality} (train + test)")
                train_data[modality] = np.load(train_file).astype(np.float32)
                test_data[modality] = np.load(test_file).astype(np.float32)
                modality_dims[modality] = train_data[modality].shape[1]
                print(f"  Train: {train_data[modality].shape}, Test: {test_data[modality].shape}")
        
        return {'train': train_data, 'test': test_data}, \
               {'train': train_labels, 'test': test_labels}, \
               modality_dims, \
               {'train': train_case_ids, 'test': test_case_ids}, \
               True  # is_split
    
    # Legacy combined format
    print("⚠️ Loading combined pretrained features (potential leakage)")
    
    data = {}
    modality_dims = {}
    
    # Load case_ids and labels first
    case_ids = None
    labels = None
    
    case_ids_file = features_dir / 'case_ids.npy'
    if case_ids_file.exists():
        case_ids = np.load(case_ids_file, allow_pickle=True)
        print(f"Loaded case_ids: {len(case_ids)} samples")
    
    labels_file = features_dir / 'labels.npy'
    if labels_file.exists():
        labels = np.load(labels_file, allow_pickle=True)
        # If labels are strings, encode them to integers
        if labels.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            print(f"Encoded string labels to integers: {le.classes_}")
        labels = labels.astype(np.int64)
        print(f"Loaded labels: {len(labels)} samples, classes={np.unique(labels)}")
    
    # Load embeddings for each modality
    for modality in modalities:
        file_path = features_dir / f"{modality}_embeddings.npy"
        
        if file_path.exists():
            print(f"Loading pretrained {modality} from {file_path}")
            features = np.load(file_path).astype(np.float32)
            
            data[modality] = features
            modality_dims[modality] = features.shape[1]
            
            print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
    
    if not data:
        raise ValueError(f"No embeddings found in {features_dir}")
    
    return data, labels, modality_dims, case_ids, False  # is_split=False


def load_transformer_model(model_dir: Path, device: torch.device) -> tuple:
    """
    Load trained transformer model and its configuration.
    
    Args:
        model_dir: Directory containing model checkpoint and config
        device: Device to load model to
        
    Returns:
        Tuple of (model, config)
    """
    # Load configuration
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Reconstruct model
    model = MultimodalFusionClassifier(
        modality_dims=config['modality_dims'],
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.1)
    )
    
    # Load weights
    checkpoint_path = model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found in {model_dir}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded transformer model from {checkpoint_path}")
    print(f"  Modalities: {list(config['modality_dims'].keys())}")
    print(f"  Num classes: {config['num_classes']}")
    print(f"  Embed dim: {config.get('embed_dim', 256)}")
    
    return model, config


def extract_features(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    extract_type: str = 'both',
    batch_size: int = 64,
    device: str = 'auto',
    output_format: str = 'both',
    use_pretrained_features: bool = False,
    pretrained_features_dir: str = None
):
    """
    Extract transformer predictions and/or features.
    
    Args:
        model_dir: Directory containing trained transformer model
        data_dir: Directory containing parquet data files
        output_dir: Output directory for extracted features
        extract_type: What to extract ('logits', 'probabilities', 'embeddings', 'both', 'all')
        batch_size: Batch size for inference
        device: Device to use
        output_format: Output format ('npy', 'csv', 'both'). CSV format is compatible with metalearner.py
    """
    model_dir = Path(model_dir)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_transformer_model(model_dir, device)
    modalities = list(config['modality_dims'].keys())
    
    # Load data (now includes case_ids and split info)
    is_split = False
    if use_pretrained_features and pretrained_features_dir:
        print(f"Loading pretrained features from: {pretrained_features_dir}")
        data, labels, _, case_ids, is_split = load_pretrained_embeddings(
            Path(pretrained_features_dir), modalities
        )
    else:
        data, labels, _, case_ids = load_multiomics_data(data_dir, modalities)
    
    # Helper function to extract features for a dataset
    def extract_from_data(data_dict, description=""):
        """Extract logits/embeddings from a data dict."""
        n_samples = list(data_dict.values())[0].shape[0]
        print(f"\nExtracting {description}: {n_samples} samples across {len(data_dict)} modalities")
        
        all_logits = []
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                # Prepare batch
                batch_data = {}
                for modality in modalities:
                    if modality in data_dict:
                        batch = data_dict[modality][i:i + batch_size]
                        batch_data[modality] = torch.from_numpy(batch).float().to(device)
                    else:
                        batch_data[modality] = None
                
                # Forward pass
                logits, attn_weights = model(batch_data)
                all_logits.append(logits.cpu().numpy())
                
                # Extract embeddings if requested
                if extract_type in ['embeddings', 'both', 'all']:
                    embeddings = model.get_embeddings(batch_data) if hasattr(model, 'get_embeddings') else logits
                    if hasattr(embeddings, 'cpu'):
                        all_embeddings.append(embeddings.cpu().numpy())
        
        logits = np.vstack(all_logits)
        probabilities = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        predictions = np.argmax(logits, axis=-1)
        embeddings = np.vstack(all_embeddings) if all_embeddings else None
        
        return logits, probabilities, predictions, embeddings
    
    # Process based on split vs non-split
    if is_split:
        print("\n✓ Processing SPLIT data (train and test separately)")
        train_logits, train_probs, train_preds, train_embeds = extract_from_data(data['train'], "train set")
        test_logits, test_probs, test_preds, test_embeds = extract_from_data(data['test'], "test set")
        
        train_labels = labels['train']
        test_labels = labels['test']
        train_case_ids = case_ids['train'] if case_ids else None
        test_case_ids = case_ids['test'] if case_ids else None
    else:
        n_samples = list(data.values())[0].shape[0]
        logits, probabilities, predictions, embeddings = extract_from_data(data, "all samples")
    
    # Save outputs based on extract_type
    extraction_info = {
        'source_model_dir': str(model_dir),
        'source_data_dir': str(data_dir),
        'num_classes': config['num_classes'],
        'modalities_used': modalities,
        'extract_type': extract_type,
        'output_format': output_format,
        'has_split': is_split,
        'outputs_saved': []
    }
    
    # Helper to save in NPY format
    def save_npy(data, filename):
        np.save(output_dir / filename, data)
        extraction_info['outputs_saved'].append(filename)
        print(f"Saved {filename}: {data.shape}")
    
    # Helper to save predictions in metalearner-compatible CSV format
    def save_metalearner_csv(probabilities, case_ids, filename):
        """Save predictions in format compatible with metalearner.py"""
        # Create column names for each class probability
        prob_cols = [f'class_{i}_prob' for i in range(probabilities.shape[1])]
        
        # Create DataFrame with case_id and probabilities
        df = pd.DataFrame(probabilities, columns=prob_cols)
        df.insert(0, 'case_id', case_ids)
        
        df.to_csv(output_dir / filename, index=False)
        extraction_info['outputs_saved'].append(filename)
        print(f"Saved {filename}: {df.shape} (metalearner-compatible)")
        return df
    
    if is_split:
        # Save train/test separately
        extraction_info['n_train_samples'] = len(train_labels)
        extraction_info['n_test_samples'] = len(test_labels)
        
        if output_format in ['npy', 'both']:
            if extract_type in ['logits', 'both', 'all']:
                save_npy(train_logits, "train_transformer_logits.npy")
                save_npy(test_logits, "test_transformer_logits.npy")
            
            if extract_type in ['probabilities', 'both', 'all']:
                save_npy(train_probs, "train_transformer_probabilities.npy")
                save_npy(test_probs, "test_transformer_probabilities.npy")
            
            if extract_type in ['embeddings', 'all'] and train_embeds is not None:
                save_npy(train_embeds, "train_transformer_embeddings.npy")
                save_npy(test_embeds, "test_transformer_embeddings.npy")
            
            save_npy(train_preds, "train_transformer_predictions.npy")
            save_npy(test_preds, "test_transformer_predictions.npy")
            save_npy(train_labels, "train_labels.npy")
            save_npy(test_labels, "test_labels.npy")
            
            if train_case_ids is not None:
                save_npy(train_case_ids, "train_case_ids.npy")
                save_npy(test_case_ids, "test_case_ids.npy")
        
        if output_format in ['csv', 'both']:
            print("\n--- Generating metalearner-compatible CSV files ---")
            save_metalearner_csv(train_probs, train_case_ids, 'train_oof_preds_Transformer.csv')
            save_metalearner_csv(test_probs, test_case_ids, 'test_preds_Transformer.csv')
        
        # Compute accuracy
        train_acc = np.mean(train_preds == train_labels)
        test_acc = np.mean(test_preds == test_labels)
        extraction_info['train_accuracy'] = float(train_acc)
        extraction_info['test_accuracy'] = float(test_acc)
        print(f"\nTrain accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
    
    else:
        # Legacy: save combined
        extraction_info['num_samples'] = n_samples
        
        if output_format in ['npy', 'both']:
            if extract_type in ['logits', 'both', 'all']:
                save_npy(logits, "transformer_logits.npy")
            
            if extract_type in ['probabilities', 'both', 'all']:
                save_npy(probabilities, "transformer_probabilities.npy")
            
            if extract_type in ['embeddings', 'all'] and embeddings is not None:
                save_npy(embeddings, "transformer_embeddings.npy")
            
            save_npy(predictions, "transformer_predictions.npy")
            
            if labels is not None:
                save_npy(labels, "labels.npy")
            
            if case_ids is not None:
                save_npy(case_ids, "case_ids.npy")
        
        if output_format in ['csv', 'both']:
            print("\n--- Generating metalearner-compatible CSV files ---")
            save_metalearner_csv(probabilities, case_ids, 'train_oof_preds_Transformer.csv')
        
        # Compute accuracy
        if labels is not None:
            accuracy = np.mean(predictions == labels)
            extraction_info['accuracy'] = float(accuracy)
            print(f"\nTransformer accuracy: {accuracy:.4f}")
    
    # Save extraction metadata
    metadata_file = output_dir / "extraction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(extraction_info, f, indent=2)
    print(f"Saved extraction metadata -> {metadata_file}")
    
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return extraction_info


def main():
    parser = argparse.ArgumentParser(
        description="Extract transformer predictions/features for QML meta-learner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Extract probabilities in CSV format for meta-learner input
  python extract_transformer_features.py \\
      --model_dir transformer_models \\
      --data_dir final_processed_datasets \\
      --output_dir transformer_predictions \\
      --output_format csv
  
  # Extract all feature types in both formats
  python extract_transformer_features.py \\
      --model_dir transformer_models \\
      --data_dir data \\
      --output_dir predictions \\
      --extract_type all \\
      --output_format both

Output files (NPY format):
  <output_dir>/
  ├── transformer_logits.npy         # Raw logits (N, num_classes)
  ├── transformer_probabilities.npy  # Softmax probabilities (N, num_classes)
  ├── transformer_predictions.npy    # Class predictions (N,)
  ├── transformer_embeddings.npy     # Penultimate layer features (N, embed_dim)
  ├── labels.npy                     # Ground truth labels (N,)
  ├── case_ids.npy                   # Sample identifiers (N,)
  └── extraction_metadata.json       # Extraction configuration

Output files (CSV format - metalearner compatible):
  <output_dir>/
  ├── train_oof_preds_Transformer.csv  # Training set probabilities
  ├── test_preds_Transformer.csv       # Test set probabilities
  └── extraction_metadata.json         # Extraction configuration

Meta-learner usage:
  # Use CSV outputs directly with metalearner.py:
  python metalearner.py \\
      --preds_dir transformer_predictions \\
      --indicator_file final_processed_datasets/indicator_features.parquet
        """)
    
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained transformer model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing parquet data files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for extracted features')
    parser.add_argument('--extract_type', type=str, default='both',
                        choices=['logits', 'probabilities', 'embeddings', 'both', 'all'],
                        help='Type of features to extract (default: both)')
    parser.add_argument('--output_format', type=str, default='both',
                        choices=['npy', 'csv', 'both'],
                        help='Output format: npy, csv (metalearner compatible), or both (default: both)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    parser.add_argument('--use_pretrained_features', action='store_true',
                        help='Use pretrained encoder features instead of raw parquet data')
    parser.add_argument('--pretrained_features_dir', type=str, default=None,
                        help='Directory containing pretrained *_embeddings.npy files')
    
    args = parser.parse_args()
    
    extract_features(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        extract_type=args.extract_type,
        batch_size=args.batch_size,
        device=args.device,
        output_format=args.output_format,
        use_pretrained_features=args.use_pretrained_features,
        pretrained_features_dir=args.pretrained_features_dir
    )


if __name__ == "__main__":
    main()
