"""
Multimodal Transformer Fusion for multi-omics data.

Implements cross-modal attention mechanisms to enable information exchange
between different modalities before final classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


__all__ = [
    'MultimodalTransformer',
    'ModalityFeatureEncoder',
    'MultimodalFusionClassifier'
]


class MultimodalTransformer(nn.Module):
    """
    Transformer-based fusion for multimodal data.
    
    Uses cross-modal attention to allow different modalities to exchange information
    and attend to relevant features from other modalities.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_modalities: int = 6,
        num_classes: int = 10,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cls_token: bool = False
    ):
        """
        Initialize multimodal transformer.
        
        Args:
            embed_dim: Dimension of embeddings (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            num_modalities: Maximum number of modalities
            num_classes: Number of output classes
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            use_cls_token: Whether to use a learnable CLS token for classification
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        
        # Learnable modality embeddings (like positional encodings)
        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        
        # Optional CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        if use_cls_token:
            # Use only CLS token representation
            classifier_input_dim = embed_dim
        else:
            # Use flattened all modality representations
            classifier_input_dim = embed_dim * num_modalities
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(
        self,
        modality_features: List[torch.Tensor],
        modality_masks: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multimodal transformer.
        
        Args:
            modality_features: List of (batch, embed_dim) tensors, one per modality
            modality_masks: (batch, num_modalities) binary mask (True = missing)
                           dtype must be torch.bool
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights)
            - logits: (batch, num_classes)
            - attention_weights: Optional tensor of attention weights
            
        Raises:
            ValueError: If shapes are inconsistent or invalid
        """
        # Validate inputs
        if not modality_features or len(modality_features) == 0:
            raise ValueError("modality_features cannot be empty")
        
        batch_size = modality_features[0].shape[0]
        if batch_size == 0:
            raise ValueError("Batch size cannot be 0")
        
        # Verify all features have same batch size and embed_dim
        for i, feat in enumerate(modality_features):
            if len(feat.shape) != 2:
                raise ValueError(f"Feature {i} has invalid shape {feat.shape}, expected (batch, embed_dim)")
            if feat.shape[0] != batch_size:
                raise ValueError(f"Feature {i} batch size {feat.shape[0]} != {batch_size}")
            if feat.shape[1] != self.embed_dim:
                raise ValueError(f"Feature {i} embed_dim {feat.shape[1]} != {self.embed_dim}")
        
        num_modalities = len(modality_features)
        
        # Validate masks if provided
        if modality_masks is not None:
            if modality_masks.dtype != torch.bool:
                modality_masks = modality_masks.to(torch.bool)
            if len(modality_masks.shape) != 2:
                raise ValueError(f"modality_masks has invalid shape {modality_masks.shape}, expected (batch, num_modalities)")
            if modality_masks.shape[0] != batch_size:
                raise ValueError(f"modality_masks batch size {modality_masks.shape[0]} != {batch_size}")
            if modality_masks.shape[1] != num_modalities:
                raise ValueError(f"modality_masks modality count {modality_masks.shape[1]} != {num_modalities}")
        
        # Stack modalities: (batch, num_modalities, embed_dim)
        modality_sequence = torch.stack(modality_features, dim=1)
        
        # Add modality embeddings
        modality_ids = torch.arange(num_modalities, device=modality_sequence.device)
        modality_emb = self.modality_embeddings(modality_ids)  # (num_modalities, embed_dim)
        modality_sequence = modality_sequence + modality_emb.unsqueeze(0)  # Broadcasting
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
            modality_sequence = torch.cat([cls_tokens, modality_sequence], dim=1)  # (batch, 1+num_mod, embed_dim)
            
            # Adjust mask to include CLS token (never masked)
            if modality_masks is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=modality_masks.device)
                modality_masks = torch.cat([cls_mask, modality_masks], dim=1)
        
        # Apply transformer with masking
        # src_key_padding_mask: True for positions to ignore (missing modalities)
        transformer_output = self.transformer(
            modality_sequence,
            src_key_padding_mask=modality_masks  # (batch, num_modalities) or (batch, 1+num_modalities)
        )
        
        # Extract features for classification
        if self.use_cls_token:
            # Use only CLS token output
            cls_output = transformer_output[:, 0, :]  # (batch, embed_dim)
            aggregated = cls_output
        else:
            # Aggregate all modality representations (flatten)
            aggregated = transformer_output.reshape(batch_size, -1)  # (batch, num_mod * embed_dim)
        
        # Classification
        logits = self.classifier(aggregated)
        
        # Attention weights (if requested)
        attention_weights = None
        if return_attention:
            # Note: Extracting attention weights requires modifying the transformer
            # For simplicity, we return None here. In practice, you would need to
            # register hooks or use a custom transformer that returns attention.
            pass
        
        return logits, attention_weights


class ModalityFeatureEncoder(nn.Module):
    """
    Feature encoder for a single modality.
    
    Converts modality-specific features into a common embedding space,
    with support for handling missing modalities via learnable tokens.
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 256, dropout: float = 0.2):
        """
        Initialize modality feature encoder.
        
        Args:
            input_dim: Number of input features
            embed_dim: Dimension of output embeddings
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Learnable token for missing modality
        self.missing_token = nn.Parameter(torch.randn(1, embed_dim))
    
    def forward(self, x: Optional[torch.Tensor], is_missing: bool = False) -> torch.Tensor:
        """
        Encode features or return missing token.
        
        Args:
            x: Input tensor of shape (batch, input_dim) or None if missing
            is_missing: Whether this modality is missing
            
        Returns:
            Embeddings of shape (batch, embed_dim)
            
        Raises:
            ValueError: If input shape is invalid
        """
        if is_missing or x is None:
            # Return learnable missing token
            batch_size = 1 if x is None else x.shape[0]
            if batch_size == 0:
                raise ValueError("Batch size cannot be 0")
            return self.missing_token.expand(batch_size, -1)
        else:
            # Validate input shape
            if len(x.shape) != 2:
                raise ValueError(f"Expected 2D tensor (batch, features), got shape {x.shape}")
            if x.shape[1] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")
            # Encode features
            return self.encoder(x)


class MultimodalFusionClassifier(nn.Module):
    """
    Complete multimodal fusion classifier.
    
    Combines modality-specific encoders with transformer fusion for
    end-to-end training on multimodal classification tasks.
    """
    
    def __init__(
        self,
        modality_dims: dict,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 10,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cls_token: bool = False,
        pretrained_encoders: Optional[dict] = None
    ):
        """
        Initialize multimodal fusion classifier.
        
        Args:
            modality_dims: Dict mapping modality names to input dimensions
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            use_cls_token: Whether to use CLS token
            pretrained_encoders: Optional dict of pretrained encoders to use
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Create or use pretrained encoders for each modality
        if pretrained_encoders is not None:
            self.encoders = nn.ModuleDict(pretrained_encoders)
        else:
            self.encoders = nn.ModuleDict({
                modality: ModalityFeatureEncoder(input_dim, embed_dim, dropout)
                for modality, input_dim in modality_dims.items()
            })
        
        # Transformer fusion
        self.transformer_fusion = MultimodalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_modalities=self.num_modalities,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_cls_token=use_cls_token
        )
    
    def forward(
        self,
        modality_data: dict,
        modality_missing: Optional[dict] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multimodal classifier.
        
        Args:
            modality_data: Dict mapping modality names to input tensors (batch, features)
                          Missing modalities can be None or not in dict
            modality_missing: Optional dict mapping modality names to boolean (True = missing)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights)
            
        Raises:
            ValueError: If input shapes are invalid or no modalities present
        """
        batch_size = None
        encoded_features = []
        missing_mask = []
        
        # Infer batch size from available data
        for modality in self.modality_names:
            if modality in modality_data and modality_data[modality] is not None:
                data = modality_data[modality]
                if not isinstance(data, torch.Tensor):
                    raise ValueError(f"modality_data[{modality}] must be torch.Tensor, got {type(data)}")
                if len(data.shape) != 2:
                    raise ValueError(f"modality_data[{modality}] must be 2D, got shape {data.shape}")
                batch_size = data.shape[0]
                break
        
        if batch_size is None:
            raise ValueError("At least one modality data must be provided")
        
        if batch_size == 0:
            raise ValueError("Batch size cannot be 0")
        
        # Encode each modality
        for modality in self.modality_names:
            # Check if modality is present
            is_missing = (
                modality not in modality_data or
                modality_data[modality] is None or
                (modality_missing is not None and modality_missing.get(modality, False))
            )
            
            if is_missing:
                # Use missing token
                encoder = self.encoders[modality]
                if isinstance(encoder, ModalityFeatureEncoder):
                    encoded = encoder(None, is_missing=True)
                elif hasattr(encoder, 'missing_token'):
                    # ModalityFeatureEncoder has missing_token
                    encoded = encoder.missing_token.expand(batch_size, -1)
                else:
                    # Fallback: use zeros
                    encoded = torch.zeros(batch_size, self.embed_dim, device=next(encoder.parameters()).device)
                
                if encoded.shape[0] == 1 and batch_size > 1:
                    encoded = encoded.expand(batch_size, -1)
                missing_mask.append(True)
            else:
                # Encode features
                data = modality_data[modality]
                if data.shape[0] != batch_size:
                    raise ValueError(f"Inconsistent batch size for {modality}: {data.shape[0]} != {batch_size}")
                
                # Check encoder type and call appropriately
                encoder = self.encoders[modality]
                if isinstance(encoder, ModalityFeatureEncoder):
                    encoded = encoder(data, is_missing=False)
                else:
                    # Standard encoder (e.g., from contrastive learning)
                    encoded = encoder(data)
                missing_mask.append(False)
            
            # Verify encoded output shape
            if encoded.shape != (batch_size, self.embed_dim):
                raise ValueError(f"Encoded {modality} shape {encoded.shape} != expected ({batch_size}, {self.embed_dim})")
            
            encoded_features.append(encoded)
        
        # Convert missing mask to tensor
        # Create per-sample masks (all samples have same modalities available)
        missing_mask_tensor = torch.tensor(
            missing_mask,
            dtype=torch.bool,
            device=encoded_features[0].device
        ).unsqueeze(0).expand(batch_size, -1)  # (batch, num_modalities)
        
        # Apply transformer fusion
        logits, attention_weights = self.transformer_fusion(
            encoded_features,
            missing_mask_tensor,
            return_attention
        )
        
        return logits, attention_weights
