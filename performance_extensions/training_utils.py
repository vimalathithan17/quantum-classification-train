"""
Training utilities for performance extensions.

Provides training loops, data loaders, and utilities for:
- Self-supervised contrastive pretraining
- Supervised fine-tuning
- Transformer fusion training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json

from .contrastive_learning import ContrastiveMultiOmicsEncoder, ContrastiveLearningLoss, ModalityEncoder
from .augmentations import get_augmentation_pipeline


__all__ = [
    'MultiOmicsDataset',
    'collate_augmented_multiomics',
    'pretrain_contrastive',
    'finetune_supervised',
    'save_pretrained_encoders',
    'load_pretrained_encoders',
    'load_single_modality_encoder'
]


def collate_augmented_multiomics(batch):
    """
    Custom collate function for augmented multi-omics data.
    
    When apply_augmentation=True in MultiOmicsDataset, each sample's data_dict
    contains lists of augmented view tensors. The default PyTorch collate
    won't correctly handle this structure. This function properly batches
    the augmented views.
    
    Args:
        batch: List of (data_dict, label) tuples from MultiOmicsDataset
        
    Returns:
        Tuple of (collated_data_dict, labels_tensor)
        - collated_data_dict: Dict mapping modality to list of batched tensors
          (one batched tensor per augmented view)
        - labels_tensor: Batched labels or None if all labels are None
    """
    data_dicts, labels = zip(*batch)
    
    collated_data = {}
    modalities = data_dicts[0].keys()
    
    for modality in modalities:
        first_sample = data_dicts[0][modality]
        
        # Check if this is augmented (list of tensors) or non-augmented (single tensor)
        if isinstance(first_sample, list):
            # Augmented: list of view tensors
            n_views = len(first_sample)
            
            # Stack each view across batch
            collated_data[modality] = [
                torch.stack([d[modality][v] for d in data_dicts])
                for v in range(n_views)
            ]
        else:
            # Non-augmented: single tensor per sample
            collated_data[modality] = torch.stack([d[modality] for d in data_dicts])
    
    # Handle labels
    if labels[0] is None:
        labels_tensor = None
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return collated_data, labels_tensor


class MultiOmicsDataset(Dataset):
    """
    Dataset for multi-omics data with multiple modalities.
    """
    
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        apply_augmentation: bool = False,
        num_augmented_views: int = 2
    ):
        """
        Initialize multi-omics dataset.
        
        Args:
            data: Dict mapping modality names to feature arrays (N, features)
            labels: Optional labels array (N,)
            apply_augmentation: Whether to apply augmentation
            num_augmented_views: Number of augmented views to generate per sample
        """
        self.data = data
        self.labels = labels
        self.modality_names = list(data.keys())
        self.apply_augmentation = apply_augmentation
        self.num_augmented_views = num_augmented_views
        
        # Verify all modalities have same number of samples
        n_samples_list = [arr.shape[0] for arr in data.values()]
        if len(set(n_samples_list)) > 1:
            raise ValueError(f"Modality sample counts don't match: {dict(zip(data.keys(), n_samples_list))}")
        
        self.n_samples = n_samples_list[0]
        
        # Create augmentation pipelines if needed
        if apply_augmentation:
            self.augmentations = {
                modality: get_augmentation_pipeline(modality)
                for modality in self.modality_names
            }
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Optional[int]]:
        """
        Get a sample.
        
        Returns:
            Tuple of (data_dict, label)
            - data_dict: Dict mapping modality to tensor (or list of augmented tensors)
            - label: Integer label or None
        """
        sample_data = {}
        
        for modality in self.modality_names:
            features = torch.from_numpy(self.data[modality][idx]).float()
            
            if self.apply_augmentation:
                # Generate augmented views
                augmented_views = self.augmentations[modality](features, self.num_augmented_views)
                sample_data[modality] = augmented_views
            else:
                sample_data[modality] = features
        
        label = None if self.labels is None else int(self.labels[idx])
        
        return sample_data, label


def pretrain_contrastive(
    model: ContrastiveMultiOmicsEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ContrastiveLearningLoss,
    device: torch.device,
    num_epochs: int = 100,
    start_epoch: int = 0,
    log_interval: int = 10,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_interval: int = 10,
    keep_last_n_checkpoints: int = 3,
    max_grad_norm: Optional[float] = 1.0,
    verbose: bool = True,
    wandb_run = None
) -> Dict[str, List[float]]:
    """
    Pretrain model with contrastive learning.
    
    Args:
        model: Contrastive encoder model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: Contrastive loss function
        device: Device to train on
        num_epochs: Number of epochs
        start_epoch: Starting epoch (for resuming training)
        log_interval: How often to log (in batches)
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        keep_last_n_checkpoints: Keep only the last N checkpoints + best (0 = keep all)
        max_grad_norm: Maximum gradient norm for clipping (None or 0 to disable)
        verbose: Whether to print progress
        wandb_run: Weights & Biases run object for logging (optional)
        
    Returns:
        Dictionary of training metrics
    """
    model.to(device)
    model.train()
    
    metrics = {'epoch_losses': [], 'batch_losses': []}
    best_loss = float('inf')
    saved_checkpoints = []  # Track checkpoint paths for cleanup
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_loss_dict = {}
        
        for batch_idx, (data_dict, _) in enumerate(dataloader):
            # Move augmented views to device
            augmented_views = {}
            for modality, views in data_dict.items():
                augmented_views[modality] = [v.to(device) for v in views]
            
            # Forward pass: encode all views
            embeddings = {}
            projections = {}
            valid_masks = {}
            
            for modality, views in augmented_views.items():
                # Encode first view (non-augmented for cross-modal)
                emb, proj, valid_mask = model(views[0], modality, return_projection=True)
                embeddings[modality] = emb
                projections[modality] = proj
                valid_masks[modality] = valid_mask
            
            # Compute contrastive loss (passes valid masks to exclude all-NaN samples)
            loss, loss_dict = loss_fn(augmented_views, embeddings, projections, model, valid_masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Track metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss
            metrics['batch_losses'].append(batch_loss)
            
            # Aggregate loss components
            for key, val in loss_dict.items():
                if key not in epoch_loss_dict:
                    epoch_loss_dict[key] = []
                epoch_loss_dict[key].append(val)
            
            # Log progress
            if verbose and (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {batch_loss:.4f} (avg: {avg_loss:.4f})")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        metrics['epoch_losses'].append(avg_epoch_loss)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] Complete. Avg Loss: {avg_epoch_loss:.4f}")
            
            # Print component losses
            for key, vals in epoch_loss_dict.items():
                print(f"  {key}: {np.mean(vals):.4f}")
        
        # Log to wandb
        if wandb_run is not None:
            try:
                log_dict = {'epoch': epoch + 1, 'loss': avg_epoch_loss}
                for key, vals in epoch_loss_dict.items():
                    log_dict[key] = np.mean(vals)
                wandb_run.log(log_dict)
            except Exception:
                pass  # Non-fatal, continue training
        
        # Save best model (based on contrastive loss - lower is better)
        # Note: Contrastive loss indicates how well the model learns to distinguish samples.
        # Lower loss = better separation of positive/negative pairs in embedding space.
        if checkpoint_dir is not None and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            
            # Save combined model (for easy loading)
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'modality_dims': model.modality_dims,
                'embed_dim': model.embed_dim,
                'projection_dim': model.projection_dim,
                'encoder_type': model.encoder_type,
            }, best_model_path)
            
            # Save each modality encoder separately for flexible downstream use
            encoders_dir = checkpoint_dir / "encoders"
            encoders_dir.mkdir(parents=True, exist_ok=True)
            
            for modality_name, encoder in model.encoders.items():
                encoder_path = encoders_dir / f"{modality_name}_encoder.pt"
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'input_dim': model.modality_dims[modality_name],
                    'embed_dim': model.embed_dim,
                    'encoder_type': model.encoder_type,
                    'epoch': epoch,
                    'loss': avg_epoch_loss,
                }, encoder_path)
            
            # Also save projection heads separately (useful for continued pretraining)
            projections_dir = checkpoint_dir / "projections"
            projections_dir.mkdir(parents=True, exist_ok=True)
            
            for modality_name, proj_head in model.projection_heads.items():
                proj_path = projections_dir / f"{modality_name}_projection.pt"
                torch.save({
                    'projection_state_dict': proj_head.state_dict(),
                    'embed_dim': model.embed_dim,
                    'projection_dim': model.projection_dim,
                }, proj_path)
            
            if verbose:
                print(f"  New best model saved (loss: {best_loss:.4f})")
                print(f"    - Combined: {best_model_path}")
                print(f"    - Per-modality encoders: {encoders_dir}/")
        
        # Save periodic checkpoint
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"contrastive_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            
            if verbose:
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Track saved checkpoints for cleanup
            saved_checkpoints.append(checkpoint_path)
            
            # Cleanup old checkpoints, keeping only last N + best
            if keep_last_n_checkpoints > 0 and len(saved_checkpoints) > keep_last_n_checkpoints:
                # Remove oldest checkpoint(s) beyond the limit
                checkpoints_to_remove = saved_checkpoints[:-keep_last_n_checkpoints]
                for old_checkpoint in checkpoints_to_remove:
                    try:
                        if old_checkpoint.exists():
                            old_checkpoint.unlink()
                            if verbose:
                                print(f"  Removed old checkpoint: {old_checkpoint.name}")
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Could not remove {old_checkpoint}: {e}")
                # Update list to only keep the recent ones
                saved_checkpoints = saved_checkpoints[-keep_last_n_checkpoints:]
    
    # Add best_loss to metrics
    metrics['best_loss'] = best_loss
    
    return metrics


def finetune_supervised(
    encoders: nn.ModuleDict,
    classifier: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    freeze_encoders: bool = False,
    log_interval: int = 10,
    checkpoint_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Fine-tune pretrained encoders with supervised learning.
    
    Args:
        encoders: Pretrained modality encoders
        classifier: Classification head
        dataloader: DataLoader for labeled training data
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        freeze_encoders: Whether to freeze encoder weights
        log_interval: How often to log (in batches)
        checkpoint_dir: Directory to save checkpoints
        verbose: Whether to print progress
        
    Returns:
        Dictionary of training metrics
    """
    encoders.to(device)
    classifier.to(device)
    
    # Freeze encoders if requested
    if freeze_encoders:
        for param in encoders.parameters():
            param.requires_grad = False
        encoders.eval()
    else:
        encoders.train()
    
    classifier.train()
    
    metrics = {'epoch_losses': [], 'epoch_accuracies': [], 'batch_losses': []}
    loss_fn = nn.CrossEntropyLoss()
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data_dict, labels) in enumerate(dataloader):
            # Move data to device
            for modality in data_dict:
                data_dict[modality] = data_dict[modality].to(device)
            labels = labels.to(device)
            
            # Extract features with encoders
            features = []
            for modality, data in data_dict.items():
                if modality in encoders:
                    result = encoders[modality](data)
                    # Handle tuple return (embedding, valid_mask) from contrastive_learning encoders
                    if isinstance(result, tuple):
                        feat = result[0]
                    else:
                        feat = result
                    features.append(feat)
            
            # Concatenate features
            combined_features = torch.cat(features, dim=1)
            
            # Classify
            logits = classifier(combined_features)
            loss = loss_fn(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_loss = loss.item()
            epoch_loss += batch_loss
            metrics['batch_losses'].append(batch_loss)
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Log progress
            if verbose and (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                batch_acc = 100.0 * correct / total
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {batch_loss:.4f} (avg: {avg_loss:.4f}) Acc: {batch_acc:.2f}%")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_accuracy = 100.0 * correct / total
        metrics['epoch_losses'].append(avg_epoch_loss)
        metrics['epoch_accuracies'].append(epoch_accuracy)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] Complete. "
                  f"Loss: {avg_epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")
        
        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"finetune_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'encoders_state_dict': encoders.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'accuracy': epoch_accuracy,
            }, checkpoint_path)
            
            if verbose:
                print(f"Saved checkpoint: {checkpoint_path}")
    
    return metrics


def save_pretrained_encoders(
    model: ContrastiveMultiOmicsEncoder,
    save_dir: Path,
    metadata: Optional[dict] = None
):
    """
    Save pretrained encoders (without projection heads).
    
    Args:
        model: Trained contrastive model
        save_dir: Directory to save encoders
        metadata: Optional metadata to save
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each encoder separately
    for modality, encoder in model.encoders.items():
        encoder_path = save_dir / f"encoder_{modality}.pt"
        torch.save(encoder.state_dict(), encoder_path)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'modality_dims': model.modality_dims,
        'embed_dim': model.embed_dim,
        'modality_names': list(model.encoders.keys()),
        'hidden_dim': 512  # Default hidden_dim used in ModalityEncoder
    })
    
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved pretrained encoders to {save_dir}")


def load_single_modality_encoder(
    encoder_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, dict]:
    """
    Load a single pretrained modality encoder.
    
    This loads encoders saved by pretrain_contrastive() in the per-modality format.
    
    Args:
        encoder_path: Path to the encoder .pt file (e.g., encoders/mRNA_encoder.pt)
        device: Device to load to
        
    Returns:
        Tuple of (encoder, metadata) where metadata contains:
            - input_dim: Input feature dimension
            - embed_dim: Output embedding dimension
            - encoder_type: 'mlp' or 'transformer'
            - epoch: Training epoch when saved
            - loss: Contrastive loss when saved
    """
    from .contrastive_learning import ModalityEncoder, TransformerModalityEncoder
    
    encoder_path = Path(encoder_path)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    
    # Load checkpoint
    checkpoint = torch.load(encoder_path, map_location=device)
    
    # Extract metadata
    input_dim = checkpoint['input_dim']
    embed_dim = checkpoint['embed_dim']
    encoder_type = checkpoint.get('encoder_type', 'mlp')  # Default to MLP for older saves
    
    # Create encoder based on type
    if encoder_type == 'transformer':
        # Transformer encoder may have additional params
        d_model = checkpoint.get('d_model', 64)
        num_heads = checkpoint.get('num_heads', 4)
        num_layers = checkpoint.get('num_layers', 2)
        encoder = TransformerModalityEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )
    else:
        hidden_dim = checkpoint.get('hidden_dim', 512)
        encoder = ModalityEncoder(input_dim, embed_dim, hidden_dim=hidden_dim)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    encoder.eval()
    
    metadata = {
        'input_dim': input_dim,
        'embed_dim': embed_dim,
        'encoder_type': encoder_type,
        'epoch': checkpoint.get('epoch'),
        'loss': checkpoint.get('loss'),
    }
    
    return encoder, metadata


def load_pretrained_encoders(
    load_dir: Path,
    device: Optional[torch.device] = None,
    modalities: Optional[List[str]] = None
) -> Tuple[nn.ModuleDict, dict]:
    """
    Load pretrained encoders.
    
    Supports two formats:
    1. New format: encoders/ subdirectory with {modality}_encoder.pt files
    2. Legacy format: encoder_{modality}.pt files with metadata.json
    
    Args:
        load_dir: Directory containing saved encoders
        device: Device to load to
        modalities: Optional list of specific modalities to load.
                   If None, loads all available modalities.
        
    Returns:
        Tuple of (encoders, metadata)
        
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If metadata is malformed
    """
    from .contrastive_learning import ModalityEncoder, TransformerModalityEncoder
    
    load_dir = Path(load_dir)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for new format (encoders/ subdirectory)
    encoders_dir = load_dir / "encoders"
    if encoders_dir.exists():
        # New format: load from encoders/ subdirectory
        encoders = nn.ModuleDict()
        metadata = {'modality_names': [], 'modality_dims': {}, 'encoder_type': None}
        
        # Find all encoder files
        encoder_files = list(encoders_dir.glob("*_encoder.pt"))
        
        if not encoder_files:
            raise FileNotFoundError(f"No encoder files found in {encoders_dir}")
        
        for encoder_path in encoder_files:
            # Extract modality name from filename
            modality = encoder_path.stem.replace('_encoder', '')
            
            # Skip if specific modalities requested and this isn't one
            if modalities is not None and modality not in modalities:
                continue
            
            encoder, enc_meta = load_single_modality_encoder(encoder_path, device)
            encoders[modality] = encoder
            
            metadata['modality_names'].append(modality)
            metadata['modality_dims'][modality] = enc_meta['input_dim']
            metadata['embed_dim'] = enc_meta['embed_dim']
            metadata['encoder_type'] = enc_meta['encoder_type']
        
        print(f"Loaded {len(encoders)} pretrained encoders from {encoders_dir}")
        return encoders, metadata
    
    # Legacy format: metadata.json + encoder_{modality}.pt files
    metadata_path = load_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. "
                              f"Also checked for new format in {encoders_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Validate metadata structure
    required_keys = {'modality_names', 'modality_dims', 'embed_dim'}
    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError(f"Metadata missing required keys: {missing_keys}")
    
    if not isinstance(metadata['modality_dims'], dict):
        raise ValueError(f"modality_dims must be dict, got {type(metadata['modality_dims'])}")
    
    if not isinstance(metadata['embed_dim'], int):
        raise ValueError(f"embed_dim must be int, got {type(metadata['embed_dim'])}")
    
    # Reconstruct encoders with type checking
    encoders = nn.ModuleDict()
    modalities_to_load = modalities if modalities else metadata['modality_names']
    
    for modality in modalities_to_load:
        if modality not in metadata['modality_names']:
            raise ValueError(f"Requested modality '{modality}' not in saved modalities: {metadata['modality_names']}")
        
        encoder_path = load_dir / f"encoder_{modality}.pt"
        
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found for modality '{modality}': {encoder_path}")
        
        # Create encoder with validated dimensions
        if modality not in metadata['modality_dims']:
            raise ValueError(f"Modality '{modality}' not in modality_dims metadata")
        
        input_dim = metadata['modality_dims'][modality]
        embed_dim = metadata['embed_dim']
        hidden_dim = metadata.get('hidden_dim', 512)  # Default if not saved in older metadata
        
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"Invalid input_dim for {modality}: {input_dim}")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"Invalid embed_dim: {embed_dim}")
        
        encoder = ModalityEncoder(input_dim, embed_dim, hidden_dim=hidden_dim)
        
        # Load weights
        try:
            state_dict = torch.load(encoder_path, map_location=device)
            encoder.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load encoder weights for {modality}: {e}")
        
        encoder.to(device)
        encoder.eval()
        encoders[modality] = encoder
    
    print(f"Loaded {len(encoders)} pretrained encoders from {load_dir}")
    
    return encoders, metadata
