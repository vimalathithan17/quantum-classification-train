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

from .contrastive_learning import ContrastiveMultiOmicsEncoder, ContrastiveLearningLoss
from .augmentations import get_augmentation_pipeline


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
    log_interval: int = 10,
    checkpoint_dir: Optional[Path] = None,
    verbose: bool = True
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
        log_interval: How often to log (in batches)
        checkpoint_dir: Directory to save checkpoints
        verbose: Whether to print progress
        
    Returns:
        Dictionary of training metrics
    """
    model.to(device)
    model.train()
    
    metrics = {'epoch_losses': [], 'batch_losses': []}
    
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
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
            
            for modality, views in augmented_views.items():
                # Encode first view (non-augmented for cross-modal)
                emb, proj = model(views[0], modality, return_projection=True)
                embeddings[modality] = emb
                projections[modality] = proj
            
            # Compute contrastive loss
            loss, loss_dict = loss_fn(augmented_views, embeddings, projections, model)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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
        
        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"contrastive_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            
            if verbose:
                print(f"Saved checkpoint: {checkpoint_path}")
    
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
                    feat = encoders[modality](data)
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
        'modality_names': list(model.encoders.keys())
    })
    
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved pretrained encoders to {save_dir}")


def load_pretrained_encoders(
    load_dir: Path,
    device: Optional[torch.device] = None
) -> Tuple[nn.ModuleDict, dict]:
    """
    Load pretrained encoders.
    
    Args:
        load_dir: Directory containing saved encoders
        device: Device to load to
        
    Returns:
        Tuple of (encoders, metadata)
    """
    load_dir = Path(load_dir)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load metadata
    metadata_path = load_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Reconstruct encoders
    from .contrastive_learning import ModalityEncoder
    
    encoders = nn.ModuleDict()
    for modality in metadata['modality_names']:
        encoder_path = load_dir / f"encoder_{modality}.pt"
        
        # Create encoder
        input_dim = metadata['modality_dims'][modality]
        embed_dim = metadata['embed_dim']
        encoder = ModalityEncoder(input_dim, embed_dim)
        
        # Load weights
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoders[modality] = encoder
    
    print(f"Loaded pretrained encoders from {load_dir}")
    
    return encoders, metadata
