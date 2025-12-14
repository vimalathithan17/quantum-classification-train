"""
Data augmentation strategies for multi-omics data.

Provides domain-specific augmentations suitable for cancer multi-omics datasets:
- Feature dropout
- Gaussian noise
- Feature masking
- Modality-specific augmentation pipelines
"""

import torch
import numpy as np
from typing import Callable, List, Optional


def feature_dropout(x: torch.Tensor, dropout_rate: float = 0.2) -> torch.Tensor:
    """
    Randomly zero out features.
    
    Args:
        x: Input tensor of shape (batch, features) or (features,)
        dropout_rate: Probability of dropping each feature
        
    Returns:
        Augmented tensor with randomly dropped features
    """
    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
    
    mask = torch.rand_like(x) > dropout_rate
    return x * mask


def add_gaussian_noise(x: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to the input.
    
    Args:
        x: Input tensor of shape (batch, features) or (features,)
        noise_level: Standard deviation of noise relative to input std
        
    Returns:
        Augmented tensor with added Gaussian noise
    """
    if noise_level < 0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")
    
    # Compute noise based on input statistics
    noise_std = noise_level * x.std()
    noise = torch.randn_like(x) * noise_std
    return x + noise


def random_feature_masking(x: torch.Tensor, mask_prob: float = 0.15) -> torch.Tensor:
    """
    Mask random features (BERT-style masking).
    
    Args:
        x: Input tensor of shape (batch, features) or (features,)
        mask_prob: Probability of masking each feature
        
    Returns:
        Augmented tensor with randomly masked features set to zero
    """
    if not 0 <= mask_prob < 1:
        raise ValueError(f"mask_prob must be in [0, 1), got {mask_prob}")
    
    mask = torch.rand_like(x) > mask_prob
    return x * mask


def mixup_augmentation(x1: torch.Tensor, x2: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    """
    Create synthetic sample by mixing two samples.
    
    Args:
        x1: First input tensor
        x2: Second input tensor  
        alpha: Beta distribution parameter for mixup coefficient
        
    Returns:
        Mixed tensor: lam * x1 + (1 - lam) * x2
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    
    if x1.shape != x2.shape:
        raise ValueError(f"Input shapes must match: {x1.shape} vs {x2.shape}")
    
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2


class OmicsAugmentation:
    """
    Augmentation pipeline for multi-omics data.
    
    Applies multiple augmentations sequentially to create augmented views
    of the same sample for contrastive learning.
    """
    
    def __init__(
        self,
        modality_name: str,
        dropout_rate: float = 0.2,
        noise_level: float = 0.1,
        mask_prob: float = 0.15,
        use_dropout: bool = True,
        use_noise: bool = True,
        use_masking: bool = False
    ):
        """
        Initialize augmentation pipeline for a specific modality.
        
        Args:
            modality_name: Name of the modality (e.g., 'GeneExp', 'Prot')
            dropout_rate: Feature dropout probability
            noise_level: Gaussian noise level
            mask_prob: Feature masking probability
            use_dropout: Whether to apply dropout augmentation
            use_noise: Whether to apply noise augmentation
            use_masking: Whether to apply masking augmentation
        """
        self.modality_name = modality_name
        self.dropout_rate = dropout_rate
        self.noise_level = noise_level
        self.mask_prob = mask_prob
        self.use_dropout = use_dropout
        self.use_noise = use_noise
        self.use_masking = use_masking
    
    def __call__(self, x: torch.Tensor, num_views: int = 2) -> List[torch.Tensor]:
        """
        Generate multiple augmented views of the input.
        
        Args:
            x: Input tensor of shape (features,) or (batch, features)
            num_views: Number of augmented views to generate
            
        Returns:
            List of augmented tensors
        """
        views = []
        
        for i in range(num_views):
            augmented = x.clone()
            
            # Apply augmentations with varying strength per view
            if self.use_dropout:
                # Vary dropout rate slightly for each view
                dropout_rate = self.dropout_rate * (1 + 0.2 * (i - num_views / 2) / num_views)
                dropout_rate = np.clip(dropout_rate, 0.1, 0.3)
                augmented = feature_dropout(augmented, dropout_rate)
            
            if self.use_noise:
                # Vary noise level slightly for each view
                noise_level = self.noise_level * (1 + 0.3 * (i - num_views / 2) / num_views)
                noise_level = np.clip(noise_level, 0.05, 0.2)
                augmented = add_gaussian_noise(augmented, noise_level)
            
            if self.use_masking:
                augmented = random_feature_masking(augmented, self.mask_prob)
            
            views.append(augmented)
        
        return views


# Modality-specific augmentation configurations
MODALITY_AUGMENTATION_CONFIGS = {
    'GeneExp': {
        'use_dropout': True,
        'use_noise': True,
        'use_masking': False,
        'dropout_rate': 0.2,
        'noise_level': 0.1
    },
    'miRNA': {
        'use_dropout': True,
        'use_noise': False,
        'use_masking': False,
        'dropout_rate': 0.2,
        'noise_level': 0.0
    },
    'Meth': {
        'use_dropout': True,
        'use_noise': False,
        'use_masking': False,
        'dropout_rate': 0.15,
        'noise_level': 0.0
    },
    'CNV': {
        'use_dropout': True,
        'use_noise': False,
        'use_masking': False,
        'dropout_rate': 0.2,
        'noise_level': 0.0
    },
    'Prot': {
        'use_dropout': True,
        'use_noise': True,
        'use_masking': False,
        'dropout_rate': 0.2,
        'noise_level': 0.1
    },
    'Mut': {
        'use_dropout': True,
        'use_noise': False,
        'use_masking': False,
        'dropout_rate': 0.15,
        'noise_level': 0.0
    }
}


def get_augmentation_pipeline(modality_name: str) -> OmicsAugmentation:
    """
    Get the recommended augmentation pipeline for a modality.
    
    Args:
        modality_name: Name of the modality
        
    Returns:
        OmicsAugmentation instance configured for the modality
    """
    if modality_name not in MODALITY_AUGMENTATION_CONFIGS:
        # Default configuration for unknown modalities
        config = {
            'use_dropout': True,
            'use_noise': False,
            'use_masking': False,
            'dropout_rate': 0.2,
            'noise_level': 0.0
        }
    else:
        config = MODALITY_AUGMENTATION_CONFIGS[modality_name]
    
    return OmicsAugmentation(modality_name=modality_name, **config)
