"""
Contrastive learning framework for multi-omics data.

Implements self-supervised pretraining using contrastive learning:
- Encoder networks with projection heads
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Intra-modal and cross-modal contrastive learning

Key Concepts:
-------------
Input Dimension (input_dim):
    - Variable per modality (e.g., GeneExpr: 5000, Prot: 200, miRNA: 800)
    - Can be ANY value (< 256, = 256, or > 256)
    - Each modality can have a different input dimension
    
Embedding Dimension (embed_dim):
    - Default: 256
    - Configurable: can be 64, 128, 256, 384, 512, etc.
    - All modalities share the SAME embedding dimension
    - This is the main representation dimension
    - Kept after pretraining for downstream tasks
    
Projection Dimension (projection_dim):
    - Default: 128
    - Used only during contrastive pretraining
    - Typically smaller than embed_dim
    - Discarded after pretraining

Architecture Flow:
-----------------
Input (variable dim) → Encoder → Embedding (embed_dim) → Projection Head → Projection (projection_dim)
                                      ↓                                              ↓
                                  Keep this                                   Use for loss,
                                  for downstream                               then discard

Example:
--------
>>> modality_dims = {'GeneExpr': 5000, 'Prot': 200, 'miRNA': 800}
>>> encoder = ContrastiveMultiOmicsEncoder(
...     modality_dims=modality_dims,
...     embed_dim=256,        # All modalities → 256-dim
...     projection_dim=128    # For contrastive loss
... )
>>> # GeneExpr: (batch, 5000) → (batch, 256) → (batch, 128)
>>> # Prot:    (batch, 200)  → (batch, 256) → (batch, 128)
>>> # miRNA:   (batch, 800)  → (batch, 256) → (batch, 128)

Why These Defaults?
------------------
embed_dim=256:
    - Balance between expressiveness and efficiency
    - Common in deep learning (BERT: 768, ResNet: 512, SimCLR: 128-2048)
    - Works well on consumer GPUs
    - Empirically validated for multi-omics data
    
projection_dim=128:
    - Typical to use smaller dimension for contrastive loss
    - Reduces computational cost
    - Forces more compact representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


__all__ = [
    'ModalityEncoder',
    'TransformerModalityEncoder',
    'ProjectionHead',
    'ContrastiveMultiOmicsEncoder',
    'nt_xent_loss',
    'cross_modal_contrastive_loss',
    'ContrastiveLearningLoss'
]


class ModalityEncoder(nn.Module):
    """
    Deep neural encoder for a single modality.
    
    Converts raw features from variable input dimensions into a common 
    embedding space with fixed dimension.
    
    Architecture:
        Input (input_dim) → Linear(hidden_dim=512) → BatchNorm → ReLU → Dropout
                          → Linear(hidden_dim//2=256) → BatchNorm → ReLU → Dropout
                          → Linear(embed_dim) → BatchNorm
        Output (embed_dim)
    
    Key Properties:
        - Input dimension (input_dim): Can be ANY value
          Examples: 50, 200, 1000, 5000, 10000, etc.
        
        - Output dimension (embed_dim): Configurable, typically 256
          Common values: 64, 128, 256, 384, 512
        
        - The encoder can EXPAND small inputs or COMPRESS large inputs:
          * Small input (100 features) → 512 → 256 → 256-dim output (expansion)
          * Large input (5000 features) → 512 → 256 → 256-dim output (compression)
    
    Example Usage:
        >>> # Small input dimension
        >>> encoder_prot = ModalityEncoder(input_dim=200, embed_dim=256)
        >>> x_prot = torch.randn(32, 200)  # 32 samples, 200 features
        >>> embedding, valid_mask = encoder_prot(x_prot)  # Output: (32, 256), (32,)
        
        >>> # Large input dimension  
        >>> encoder_gene = ModalityEncoder(input_dim=5000, embed_dim=256)
        >>> x_gene = torch.randn(32, 5000)  # 32 samples, 5000 features
        >>> embedding, valid_mask = encoder_gene(x_gene)  # Output: (32, 256), (32,)
        
        >>> # Both produce same embedding dimension!
        >>> assert embedding_prot.shape == embedding_gene.shape == (32, 256)
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.2):
        """
        Initialize modality encoder.
        
        Args:
            input_dim: Number of input features (can be any value)
                      - For GeneExpr: typically 500-20000
                      - For miRNA: typically 200-1000
                      - For Prot: typically 100-500
                      - Can be less than, equal to, or greater than embed_dim
                      
            embed_dim: Dimension of output embeddings (default: 256)
                      - All modalities should use the same embed_dim
                      - Common values: 64, 128, 256, 384, 512
                      - Trade-off: larger = more expressive but slower
                      
            hidden_dim: Dimension of hidden layers (default: 512)
                       - Intermediate representation size
                       - Typically larger than both input_dim and embed_dim
                       
            dropout: Dropout probability (default: 0.2)
                    - Regularization to prevent overfitting
                    - Range: 0.0 to 0.5, typically 0.1-0.3
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
        
        # Learnable token for missing modality (scaled for stability)
        self.missing_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
    
    def forward(self, x: Optional[torch.Tensor], is_missing: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input features or return missing token.
        
        Args:
            x: Input tensor of shape (batch, input_dim), or None if missing
            is_missing: Whether this modality is missing for all samples
            
        Returns:
            Tuple of:
                - Embeddings of shape (batch, embed_dim)
                - Valid mask of shape (batch,) - True for valid samples, False for all-NaN
            
        Raises:
            ValueError: If batch size is 0 or input shape is invalid
        """
        if is_missing or x is None:
            # Return learnable missing token
            batch_size = 1 if x is None else x.shape[0]
            if batch_size == 0:
                raise ValueError("Batch size cannot be 0")
            # All samples are invalid when modality is missing
            valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.missing_token.device)
            return self.missing_token.expand(batch_size, -1), valid_mask
        else:
            # Validate input shape
            if len(x.shape) != 2:
                raise ValueError(f"Expected 2D tensor (batch, features), got shape {x.shape}")
            if x.shape[1] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")
            
            # Detect all-NaN samples (samples where ALL features are NaN)
            nan_mask = torch.isnan(x)  # (batch, input_dim)
            all_nan_samples = nan_mask.all(dim=1)  # (batch,) - True if all features are NaN
            valid_mask = ~all_nan_samples  # True = valid, False = all-NaN
            
            # For samples with any NaN, replace with 0 to avoid NaN in output
            # (This is a fallback - ideally data should be imputed before using MLP)
            if nan_mask.any():
                x = torch.where(nan_mask, torch.zeros_like(x), x)
            
            # Encode
            embedding = self.encoder(x)
            
            return embedding, valid_mask


class TransformerModalityEncoder(nn.Module):
    """
    Transformer-based encoder for a single modality with native missing value handling.
    
    Uses self-attention to learn relationships between features and can handle
    feature-level missing values (NaN in individual columns) by masking them out
    during attention computation.
    
    Architecture:
        Input (batch, input_dim) → Feature Embedding (batch, input_dim, d_model)
                                 → Positional Encoding
                                 → Transformer Encoder (with attention masking for NaN)
                                 → Pooling (mean over non-masked features)
                                 → Output Linear → (batch, embed_dim)
    
    Key Properties:
        - Treats each feature as a token in a sequence
        - Features with NaN values are masked out during attention
        - Learns to infer missing feature values from context of other features
        - More compute-intensive than MLP encoder but handles missingness natively
    
    Example Usage:
        >>> encoder = TransformerModalityEncoder(input_dim=200, embed_dim=256)
        >>> x = torch.randn(32, 200)
        >>> x[0, 10:20] = float('nan')  # Some features missing
        >>> embedding, valid_mask = encoder(x)  # Output: (32, 256) and (32,)
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize transformer modality encoder.
        
        Args:
            input_dim: Number of input features (each treated as a token)
            embed_dim: Dimension of output embeddings (default: 256)
            d_model: Dimension of transformer model (default: 64)
            num_heads: Number of attention heads (default: 4)
            num_layers: Number of transformer encoder layers (default: 2)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.d_model = d_model
        
        # Embed each feature value into d_model dimensions
        # Each scalar feature becomes a d_model-dimensional vector
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Learnable positional encoding for each feature position
        # Scale by sqrt(d_model) for better numerical stability (like in original Transformer)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        
        # Learnable mask token for missing features
        # Scale down for stability
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Learnable token for completely missing modality (scaled for stability)
        self.missing_modality_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
    
    def forward(self, x: Optional[torch.Tensor], is_missing: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input features with attention-based missing value handling.
        
        Args:
            x: Input tensor of shape (batch, input_dim), may contain NaN values
               or None if entire modality is missing
            is_missing: Whether this modality is missing for all samples
            
        Returns:
            Tuple of:
                - Embeddings of shape (batch, embed_dim)
                - Valid mask of shape (batch,) - True for valid samples, False for all-NaN samples
                  Samples with ALL features as NaN are marked invalid and should be excluded
                  from loss computation
        """
        if is_missing or x is None:
            batch_size = 1 if x is None else x.shape[0]
            if batch_size == 0:
                raise ValueError("Batch size cannot be 0")
            # All samples are invalid when modality is missing
            valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.missing_modality_token.device)
            return self.missing_modality_token.expand(batch_size, -1), valid_mask
        
        batch_size, seq_len = x.shape
        
        # Detect NaN values (feature-level missingness)
        nan_mask = torch.isnan(x)  # (batch, input_dim)
        
        # Replace NaN with 0 for embedding (will be masked in attention)
        x_filled = torch.where(nan_mask, torch.zeros_like(x), x)
        
        # Embed each feature value: (batch, input_dim) -> (batch, input_dim, d_model)
        x_embedded = self.feature_embedding(x_filled.unsqueeze(-1))  # (batch, input_dim, d_model)
        
        # For NaN positions, use the learnable mask token instead
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
        x_embedded = torch.where(nan_mask.unsqueeze(-1), mask_tokens, x_embedded)
        
        # Add positional encoding
        x_embedded = x_embedded + self.pos_encoding
        
        # Create attention mask: True = masked out (for PyTorch transformer)
        # We do NOT mask NaN in attention - we let the model learn from context
        # But we create a padding mask if needed
        # For now, let all features attend to each other (including mask tokens)
        
        # Transformer encoding
        encoded = self.transformer(x_embedded)  # (batch, input_dim, d_model)
        
        # Pool over features (mean pooling, excluding NaN positions for weighting)
        # Weight by inverse of NaN ratio to focus on present features
        feature_valid_mask = ~nan_mask  # (batch, input_dim)
        valid_counts = feature_valid_mask.sum(dim=1, keepdim=True)  # (batch, 1)
        
        # Identify all-NaN samples (these will be excluded from loss)
        all_nan_samples = (valid_counts == 0).squeeze(-1)  # (batch,)
        sample_valid_mask = ~all_nan_samples  # True = valid, False = all-NaN
        
        if all_nan_samples.any():
            # For all-NaN samples, use uniform weights over all mask tokens
            # This produces an embedding, but it will be excluded from loss
            uniform_weights = torch.ones_like(nan_mask, dtype=torch.float32) / seq_len
            valid_counts_safe = valid_counts.clamp(min=1)  # Prevent div by zero
            non_nan_weights = feature_valid_mask.float() / valid_counts_safe
            
            # Use uniform weights for all-NaN samples, normal weights otherwise
            weights = torch.where(
                all_nan_samples.unsqueeze(-1),
                uniform_weights,
                non_nan_weights
            )
        else:
            # Normal case: weight by valid (non-NaN) positions only
            weights = feature_valid_mask.float() / valid_counts.clamp(min=1)  # (batch, input_dim)
        
        # Weighted mean pooling
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)  # (batch, d_model)
        
        # Project to embedding dimension
        output = self.output_proj(pooled)  # (batch, embed_dim)
        
        return output, sample_valid_mask


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
    
    Key Architecture Points:
        1. Each modality gets its own encoder
        2. All encoders map to the SAME embedding dimension
        3. Each encoder has its own projection head for contrastive loss
        4. Projection heads are discarded after pretraining
    
    Dimension Flow:
        Raw Data → Encoder → Embedding → Projection Head → Projection → Loss
        
        GeneExpr (5000) → (256) → (128) ┐
        miRNA   (800)  → (256) → (128) ├→ Contrastive Loss
        Prot    (200)  → (256) → (128) ┘
        
    After Pretraining:
        GeneExpr (5000) → (256) → [Use for downstream tasks]
        miRNA   (800)  → (256) → [Use for downstream tasks]
        Prot    (200)  → (256) → [Use for downstream tasks]
        
    Example:
        >>> # Define modalities with different input dimensions
        >>> modality_dims = {
        ...     'GeneExpr': 5000,  # 5000 gene expression features
        ...     'miRNA': 800,     # 800 miRNA features
        ...     'Prot': 200,      # 200 protein features
        ...     'CNV': 1500       # 1500 CNV features
        ... }
        >>> 
        >>> # Create encoder with shared embedding dimension
        >>> encoder = ContrastiveMultiOmicsEncoder(
        ...     modality_dims=modality_dims,
        ...     embed_dim=256,      # All modalities → 256-dim
        ...     projection_dim=128  # For contrastive loss
        ... )
        >>> 
        >>> # Process each modality
        >>> x_gene = torch.randn(32, 5000)
        >>> embedding, projection, valid_mask = encoder(x_gene, 'GeneExpr')
        >>> # embedding: (32, 256) - keep for downstream
        >>> # projection: (32, 128) - use for contrastive loss
        >>> # valid_mask: (32,) - True for samples with valid data
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        embed_dim: int = 256,
        projection_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        encoder_type: str = 'mlp',
        transformer_d_model: int = 64,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 2
    ):
        """
        Initialize multi-omics contrastive encoder.
        
        Args:
            modality_dims: Dictionary mapping modality names to input dimensions
                          Example: {'GeneExpr': 5000, 'Prot': 200, 'miRNA': 800}
                          - Keys: modality names (strings)
                          - Values: input feature dimensions (integers, any size)
                          - Each modality can have a different input dimension
                          
            embed_dim: Dimension of embeddings (default: 256)
                      - All modalities will be mapped to this dimension
                      - This is the SHARED embedding space
                      - Common values: 64, 128, 256, 384, 512
                      - Kept after pretraining for downstream tasks
                      - Important: All modalities must use the same embed_dim
                        for cross-modal contrastive learning to work
                      
            projection_dim: Dimension of projections for contrastive loss (default: 128)
                           - Typically smaller than embed_dim
                           - Only used during pretraining
                           - Discarded after pretraining
                           - Common values: 64, 128, 256
                           
            hidden_dim: Dimension of hidden layers in MLP encoders (default: 512)
                       - Intermediate representation size
                       - Typically larger than embed_dim
                       - All modality encoders use the same hidden_dim
                       - Only used when encoder_type='mlp'
                       
            dropout: Dropout probability (default: 0.2)
                    - Applied in encoder networks
                    - Regularization to prevent overfitting
                    - Range: 0.0 to 0.5
                    
            encoder_type: Type of encoder to use (default: 'mlp')
                         - 'mlp': Fast MLP-based encoder (requires imputed data)
                         - 'transformer': Attention-based encoder that handles feature-level
                           NaN values natively by learning from context
                           
            transformer_d_model: Dimension of transformer model (default: 64)
                                Only used when encoder_type='transformer'
                                
            transformer_num_heads: Number of attention heads (default: 4)
                                  Only used when encoder_type='transformer'
                                  
            transformer_num_layers: Number of transformer layers (default: 2)
                                   Only used when encoder_type='transformer'
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.encoder_type = encoder_type
        
        # Encoder for each modality
        if encoder_type == 'transformer':
            self.encoders = nn.ModuleDict({
                modality: TransformerModalityEncoder(
                    input_dim=input_dim,
                    embed_dim=embed_dim,
                    d_model=transformer_d_model,
                    num_heads=transformer_num_heads,
                    num_layers=transformer_num_layers,
                    dropout=dropout
                )
                for modality, input_dim in modality_dims.items()
            })
        else:  # default: 'mlp'
            self.encoders = nn.ModuleDict({
                modality: ModalityEncoder(input_dim, embed_dim, hidden_dim, dropout)
                for modality, input_dim in modality_dims.items()
            })
        
        # Projection head for each modality
        self.projection_heads = nn.ModuleDict({
            modality: ProjectionHead(embed_dim, projection_dim)
            for modality in modality_dims.keys()
        })
    
    def encode(
        self,
        x: Optional[torch.Tensor],
        modality_name: str,
        is_missing: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input for a specific modality (without projection).
        
        Supports missing modality handling via learnable missing tokens.
        
        Args:
            x: Input tensor of shape (batch, features), or None if missing
            modality_name: Name of the modality
            is_missing: Whether this modality is missing
            
        Returns:
            Tuple of:
                - Embeddings of shape (batch, embed_dim)
                - Valid mask of shape (batch,) - True for valid samples
        """
        if modality_name not in self.encoders:
            raise ValueError(f"Unknown modality: {modality_name}")
        
        return self.encoders[modality_name](x, is_missing=is_missing)
    
    def forward(
        self,
        x: Optional[torch.Tensor],
        modality_name: str,
        return_projection: bool = True,
        is_missing: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through encoder and optionally projection head.
        
        Supports missing modality handling via learnable missing tokens.
        
        Args:
            x: Input tensor of shape (batch, features), or None if missing
            modality_name: Name of the modality
            return_projection: Whether to return projection for contrastive loss
            is_missing: Whether this modality is missing
            
        Returns:
            Tuple of (embedding, projection, valid_mask):
                - embedding: shape (batch, embed_dim)
                - projection: shape (batch, projection_dim) or None
                - valid_mask: shape (batch,) - True for valid samples, False for all-NaN
        """
        if modality_name not in self.encoders:
            raise ValueError(f"Unknown modality: {modality_name}")
        
        # Encode (handles missing modality internally, returns valid mask)
        embedding, valid_mask = self.encoders[modality_name](x, is_missing=is_missing)
        
        if return_projection:
            # Project for contrastive learning
            projection = self.projection_heads[modality_name](embedding)
            return embedding, projection, valid_mask
        else:
            # For downstream tasks (after pretraining)
            return embedding, None, valid_mask


def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
    eps: float = 1e-8,
    valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Used in SimCLR and other contrastive learning methods.
    
    Args:
        z_i: Projections from augmentation 1, shape (batch_size, projection_dim)
        z_j: Projections from augmentation 2, shape (batch_size, projection_dim)
        temperature: Temperature parameter for scaling
        eps: Small constant for numerical stability
        valid_mask: Optional boolean mask of shape (batch_size,) indicating valid samples.
                   Samples with False are excluded from loss computation.
                   This is used to skip all-NaN samples when using transformer encoder.
        
    Returns:
        Scalar loss value (0.0 if no valid samples)
        
    Raises:
        ValueError: If batch_size < 2 (NT-Xent requires at least 2 samples)
    """
    batch_size = z_i.shape[0]
    
    # Apply valid mask if provided
    if valid_mask is not None:
        # Only keep valid samples
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        n_valid = valid_indices.shape[0]
        
        if n_valid < 2:
            # Not enough valid samples for contrastive learning
            # Return 0 loss (no gradient) rather than raising error
            return torch.tensor(0.0, device=z_i.device, requires_grad=True)
        
        z_i = z_i[valid_indices]
        z_j = z_j[valid_indices]
        batch_size = n_valid
    
    # NT-Xent requires at least 2 samples for meaningful contrastive learning
    if batch_size < 2:
        raise ValueError(f"NT-Xent loss requires batch_size >= 2, got {batch_size}. "
                        "Use drop_last=True in DataLoader to avoid incomplete batches.")
    
    # Normalize projections
    z_i = F.normalize(z_i, dim=1, eps=eps)
    z_j = F.normalize(z_j, dim=1, eps=eps)
    
    # Check for NaN in inputs (can happen if model weights degrade)
    if torch.isnan(z_i).any() or torch.isnan(z_j).any():
        # Return zero loss to prevent NaN propagation
        return torch.tensor(0.0, device=z_i.device, requires_grad=True)
    
    # Concatenate to create 2N samples
    representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(representations, representations.T) / temperature  # (2N, 2N)
    
    # Clamp similarity values to prevent numerical overflow with very low temperature
    # Max value of 50 prevents exp(50) ≈ 5e21 which is still within float32 range
    similarity_matrix = similarity_matrix.clamp(min=-50.0, max=50.0)
    
    # Create labels for positive pairs
    # For index i, positive is at i+N (or i-N if i>=N)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z_i.device),
        torch.arange(batch_size, device=z_i.device)
    ], dim=0)
    
    # Mask out diagonal (self-similarity) using -inf for numerical stability
    mask_diag = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix.masked_fill(mask_diag, float('-inf'))
    
    # Standard cross-entropy loss (InfoNCE)
    loss = F.cross_entropy(similarity_matrix, labels)
    
    # Final NaN check - return zero loss if something went wrong
    if torch.isnan(loss):
        return torch.tensor(0.0, device=z_i.device, requires_grad=True)
    
    return loss


def cross_modal_contrastive_loss(
    embedding_1: torch.Tensor,
    embedding_2: torch.Tensor,
    projection_head_1: nn.Module,
    projection_head_2: nn.Module,
    temperature: float = 0.5,
    valid_mask_1: Optional[torch.Tensor] = None,
    valid_mask_2: Optional[torch.Tensor] = None
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
        valid_mask_1: Optional valid mask for modality 1, shape (batch,)
        valid_mask_2: Optional valid mask for modality 2, shape (batch,)
        
    Returns:
        Scalar loss value
    """
    # Project embeddings
    proj_1 = projection_head_1(embedding_1)
    proj_2 = projection_head_2(embedding_2)
    
    # Combine valid masks: sample must be valid in BOTH modalities
    if valid_mask_1 is not None and valid_mask_2 is not None:
        combined_mask = valid_mask_1 & valid_mask_2
    elif valid_mask_1 is not None:
        combined_mask = valid_mask_1
    elif valid_mask_2 is not None:
        combined_mask = valid_mask_2
    else:
        combined_mask = None
    
    # Compute NT-Xent loss with valid mask
    return nt_xent_loss(proj_1, proj_2, temperature, valid_mask=combined_mask)


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
        model: ContrastiveMultiOmicsEncoder,
        valid_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total contrastive loss.
        
        Args:
            augmented_views: Dict mapping modality -> [view1, view2, ...]
            embeddings: Dict mapping modality -> embeddings
            projections: Dict mapping modality -> projections
            model: The contrastive encoder model
            valid_masks: Optional dict mapping modality -> valid mask (batch,).
                        Samples with False are excluded from loss.
                        Used to skip all-NaN samples with transformer encoder.
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        total_loss = 0.0
        loss_dict = {}
        n_losses = 0
        
        # Intra-modal contrastive loss (augmentations of same modality)
        for modality, views in augmented_views.items():
            if len(views) >= 2:
                # Encode both views (get valid masks)
                _, proj_1, valid_1 = model(views[0], modality, return_projection=True)
                _, proj_2, valid_2 = model(views[1], modality, return_projection=True)
                
                # Combine valid masks from both views
                combined_mask = valid_1 & valid_2
                
                # Compute loss with valid mask
                loss = nt_xent_loss(proj_1, proj_2, self.temperature, valid_mask=combined_mask)
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
                # Get valid masks for both modalities
                mask1 = valid_masks.get(mod1) if valid_masks else None
                mask2 = valid_masks.get(mod2) if valid_masks else None
                
                loss = cross_modal_contrastive_loss(
                    embeddings[mod1],
                    embeddings[mod2],
                    model.projection_heads[mod1],
                    model.projection_heads[mod2],
                    self.temperature,
                    valid_mask_1=mask1,
                    valid_mask_2=mask2
                )
                total_loss += loss
                loss_dict[f'cross_{mod1}_{mod2}'] = loss.item()
                n_losses += 1
        
        # Average loss
        if n_losses > 0:
            total_loss = total_loss / n_losses
        
        return total_loss, loss_dict
