"""
Contrastive learning framework for multi-omics data.

Implements self-supervised pretraining using contrastive learning:
- Encoder networks with projection heads
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Intra-modal and cross-modal contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ModalityEncoder(nn.Module):
    """
    Deep neural encoder for a single modality.
    
    Converts raw features into a common embedding space.
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.2):
        """
        Initialize modality encoder.
        
        Args:
            input_dim: Number of input features
            embed_dim: Dimension of output embeddings
            hidden_dim: Dimension of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Embeddings of shape (batch, embed_dim)
        """
        return self.encoder(x)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Maps embeddings to a lower-dimensional space where contrastive loss is computed.
    This head is discarded after pretraining.
    """
    
    def __init__(self, embed_dim: int = 256, projection_dim: int = 128):
        """
        Initialize projection head.
        
        Args:
            embed_dim: Dimension of input embeddings
            projection_dim: Dimension of output projections
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.
        
        Args:
            x: Input embeddings of shape (batch, embed_dim)
            
        Returns:
            Projections of shape (batch, projection_dim)
        """
        return self.projection(x)


class ContrastiveMultiOmicsEncoder(nn.Module):
    """
    Multi-modality contrastive learning encoder.
    
    Contains separate encoders and projection heads for each modality.
    Can be used for both intra-modal and cross-modal contrastive learning.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        embed_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        """
        Initialize multi-omics contrastive encoder.
        
        Args:
            modality_dims: Dictionary mapping modality names to input dimensions
            embed_dim: Dimension of embeddings
            projection_dim: Dimension of projections for contrastive loss
            hidden_dim: Dimension of hidden layers in encoders
            dropout: Dropout probability
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        
        # Encoder for each modality
        self.encoders = nn.ModuleDict({
            modality: ModalityEncoder(input_dim, embed_dim, hidden_dim, dropout)
            for modality, input_dim in modality_dims.items()
        })
        
        # Projection head for each modality
        self.projection_heads = nn.ModuleDict({
            modality: ProjectionHead(embed_dim, projection_dim)
            for modality in modality_dims.keys()
        })
    
    def encode(self, x: torch.Tensor, modality_name: str) -> torch.Tensor:
        """
        Encode input for a specific modality (without projection).
        
        Args:
            x: Input tensor of shape (batch, features)
            modality_name: Name of the modality
            
        Returns:
            Embeddings of shape (batch, embed_dim)
        """
        if modality_name not in self.encoders:
            raise ValueError(f"Unknown modality: {modality_name}")
        
        return self.encoders[modality_name](x)
    
    def forward(
        self,
        x: torch.Tensor,
        modality_name: str,
        return_projection: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder and optionally projection head.
        
        Args:
            x: Input tensor of shape (batch, features)
            modality_name: Name of the modality
            return_projection: Whether to return projection for contrastive loss
            
        Returns:
            Tuple of (embedding, projection) if return_projection=True
            Otherwise just embedding
        """
        if modality_name not in self.encoders:
            raise ValueError(f"Unknown modality: {modality_name}")
        
        # Encode
        embedding = self.encoders[modality_name](x)
        
        if return_projection:
            # Project for contrastive learning
            projection = self.projection_heads[modality_name](embedding)
            return embedding, projection
        else:
            # For downstream tasks (after pretraining)
            return embedding, None


def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Used in SimCLR and other contrastive learning methods.
    
    Args:
        z_i: Projections from augmentation 1, shape (batch_size, projection_dim)
        z_j: Projections from augmentation 2, shape (batch_size, projection_dim)
        temperature: Temperature parameter for scaling
        eps: Small constant for numerical stability
        
    Returns:
        Scalar loss value
    """
    batch_size = z_i.shape[0]
    
    # Normalize projections
    z_i = F.normalize(z_i, dim=1, eps=eps)
    z_j = F.normalize(z_j, dim=1, eps=eps)
    
    # Concatenate to create 2N samples
    representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(representations, representations.T) / temperature  # (2N, 2N)
    
    # Create labels for positive pairs
    # For index i, positive is at i+N (or i-N if i>=N)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z_i.device),
        torch.arange(batch_size, device=z_i.device)
    ], dim=0)
    
    # Mask out diagonal (self-similarity)
    mask_diag = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix.masked_fill(mask_diag, -1e9)
    
    # Standard cross-entropy loss (InfoNCE)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def cross_modal_contrastive_loss(
    embedding_1: torch.Tensor,
    embedding_2: torch.Tensor,
    projection_head_1: nn.Module,
    projection_head_2: nn.Module,
    temperature: float = 0.5
) -> torch.Tensor:
    """
    Contrastive loss across two different modalities.
    
    For the same patient, embeddings from different modalities should be similar.
    
    Args:
        embedding_1: Embeddings from modality 1, shape (batch, embed_dim)
        embedding_2: Embeddings from modality 2, shape (batch, embed_dim)
        projection_head_1: Projection head for modality 1
        projection_head_2: Projection head for modality 2
        temperature: Temperature parameter
        
    Returns:
        Scalar loss value
    """
    # Project embeddings
    proj_1 = projection_head_1(embedding_1)
    proj_2 = projection_head_2(embedding_2)
    
    # Compute NT-Xent loss
    return nt_xent_loss(proj_1, proj_2, temperature)


class ContrastiveLearningLoss(nn.Module):
    """
    Combined contrastive learning loss for multi-omics data.
    
    Supports both intra-modal (same modality, different augmentations)
    and cross-modal (different modalities, same patient) contrastive learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        use_cross_modal: bool = True,
        cross_modal_pairs: Optional[list] = None
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for NT-Xent loss
            use_cross_modal: Whether to include cross-modal contrastive loss
            cross_modal_pairs: List of (modality1, modality2) pairs for cross-modal learning
                              If None, all pairs are used
        """
        super().__init__()
        
        self.temperature = temperature
        self.use_cross_modal = use_cross_modal
        self.cross_modal_pairs = cross_modal_pairs
    
    def forward(
        self,
        augmented_views: Dict[str, list],
        embeddings: Dict[str, torch.Tensor],
        projections: Dict[str, torch.Tensor],
        model: ContrastiveMultiOmicsEncoder
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total contrastive loss.
        
        Args:
            augmented_views: Dict mapping modality -> [view1, view2, ...]
            embeddings: Dict mapping modality -> embeddings
            projections: Dict mapping modality -> projections
            model: The contrastive encoder model
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        total_loss = 0.0
        loss_dict = {}
        n_losses = 0
        
        # Intra-modal contrastive loss (augmentations of same modality)
        for modality, views in augmented_views.items():
            if len(views) >= 2:
                # Encode both views
                _, proj_1 = model(views[0], modality, return_projection=True)
                _, proj_2 = model(views[1], modality, return_projection=True)
                
                # Compute loss
                loss = nt_xent_loss(proj_1, proj_2, self.temperature)
                total_loss += loss
                loss_dict[f'intra_{modality}'] = loss.item()
                n_losses += 1
        
        # Cross-modal contrastive loss (different modalities, same patient)
        if self.use_cross_modal:
            available_modalities = list(embeddings.keys())
            
            if self.cross_modal_pairs is None:
                # Use all pairs
                pairs = [
                    (available_modalities[i], available_modalities[j])
                    for i in range(len(available_modalities))
                    for j in range(i + 1, len(available_modalities))
                ]
            else:
                # Use specified pairs, filtered by availability
                pairs = [
                    (m1, m2) for m1, m2 in self.cross_modal_pairs
                    if m1 in available_modalities and m2 in available_modalities
                ]
            
            for mod1, mod2 in pairs:
                loss = cross_modal_contrastive_loss(
                    embeddings[mod1],
                    embeddings[mod2],
                    model.projection_heads[mod1],
                    model.projection_heads[mod2],
                    self.temperature
                )
                total_loss += loss
                loss_dict[f'cross_{mod1}_{mod2}'] = loss.item()
                n_losses += 1
        
        # Average loss
        if n_losses > 0:
            total_loss = total_loss / n_losses
        
        return total_loss, loss_dict
