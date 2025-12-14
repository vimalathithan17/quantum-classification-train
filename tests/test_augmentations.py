"""
Tests for data augmentation functions.
"""

import pytest
import torch
import numpy as np

from performance_extensions.augmentations import (
    feature_dropout,
    add_gaussian_noise,
    random_feature_masking,
    mixup_augmentation,
    OmicsAugmentation,
    get_augmentation_pipeline
)


class TestFeatureDropout:
    """Test feature dropout augmentation."""
    
    def test_dropout_shape_preserved(self):
        """Test that dropout preserves tensor shape."""
        x = torch.randn(10, 50)
        x_aug = feature_dropout(x, dropout_rate=0.2)
        assert x_aug.shape == x.shape
    
    def test_dropout_zeros_features(self):
        """Test that dropout actually zeros some features."""
        x = torch.ones(100, 50)
        x_aug = feature_dropout(x, dropout_rate=0.5)
        
        # Should have some zeros (with high probability)
        assert (x_aug == 0).any()
        
        # Should not zero all features
        assert (x_aug != 0).any()
    
    def test_dropout_rate_validation(self):
        """Test that invalid dropout rates raise errors."""
        x = torch.randn(10, 50)
        
        with pytest.raises(ValueError):
            feature_dropout(x, dropout_rate=-0.1)
        
        with pytest.raises(ValueError):
            feature_dropout(x, dropout_rate=1.0)


class TestGaussianNoise:
    """Test Gaussian noise augmentation."""
    
    def test_noise_shape_preserved(self):
        """Test that noise preserves tensor shape."""
        x = torch.randn(10, 50)
        x_aug = add_gaussian_noise(x, noise_level=0.1)
        assert x_aug.shape == x.shape
    
    def test_noise_changes_values(self):
        """Test that noise actually changes values."""
        x = torch.randn(10, 50)
        x_aug = add_gaussian_noise(x, noise_level=0.1)
        
        # Should not be identical
        assert not torch.allclose(x, x_aug)
    
    def test_noise_level_validation(self):
        """Test that negative noise level raises error."""
        x = torch.randn(10, 50)
        
        with pytest.raises(ValueError):
            add_gaussian_noise(x, noise_level=-0.1)
    
    def test_zero_noise_unchanged(self):
        """Test that zero noise level returns input unchanged."""
        x = torch.randn(10, 50)
        x_aug = add_gaussian_noise(x, noise_level=0.0)
        
        assert torch.allclose(x, x_aug)


class TestFeatureMasking:
    """Test feature masking augmentation."""
    
    def test_masking_shape_preserved(self):
        """Test that masking preserves tensor shape."""
        x = torch.randn(10, 50)
        x_aug = random_feature_masking(x, mask_prob=0.15)
        assert x_aug.shape == x.shape
    
    def test_masking_zeros_features(self):
        """Test that masking zeros some features."""
        x = torch.ones(100, 50)
        x_aug = random_feature_masking(x, mask_prob=0.5)
        
        # Should have some zeros
        assert (x_aug == 0).any()
    
    def test_mask_prob_validation(self):
        """Test that invalid mask probabilities raise errors."""
        x = torch.randn(10, 50)
        
        with pytest.raises(ValueError):
            random_feature_masking(x, mask_prob=-0.1)
        
        with pytest.raises(ValueError):
            random_feature_masking(x, mask_prob=1.0)


class TestMixup:
    """Test mixup augmentation."""
    
    def test_mixup_shape_preserved(self):
        """Test that mixup preserves tensor shape."""
        x1 = torch.randn(10, 50)
        x2 = torch.randn(10, 50)
        x_mixed = mixup_augmentation(x1, x2, alpha=0.2)
        
        assert x_mixed.shape == x1.shape
    
    def test_mixup_interpolation(self):
        """Test that mixup creates interpolation."""
        x1 = torch.zeros(10, 50)
        x2 = torch.ones(10, 50)
        x_mixed = mixup_augmentation(x1, x2, alpha=0.2)
        
        # Mixed values should be between 0 and 1
        assert (x_mixed >= 0).all()
        assert (x_mixed <= 1).all()
    
    def test_mixup_shape_mismatch_error(self):
        """Test that shape mismatch raises error."""
        x1 = torch.randn(10, 50)
        x2 = torch.randn(10, 40)
        
        with pytest.raises(ValueError):
            mixup_augmentation(x1, x2, alpha=0.2)
    
    def test_mixup_alpha_validation(self):
        """Test that negative alpha raises error."""
        x1 = torch.randn(10, 50)
        x2 = torch.randn(10, 50)
        
        with pytest.raises(ValueError):
            mixup_augmentation(x1, x2, alpha=-0.1)


class TestOmicsAugmentation:
    """Test OmicsAugmentation pipeline."""
    
    def test_augmentation_pipeline_creates_views(self):
        """Test that pipeline creates correct number of views."""
        aug = OmicsAugmentation('GeneExp')
        x = torch.randn(50)
        
        views = aug(x, num_views=2)
        
        assert len(views) == 2
        assert all(v.shape == x.shape for v in views)
    
    def test_augmentation_views_differ(self):
        """Test that different views are actually different."""
        aug = OmicsAugmentation('GeneExp', use_dropout=True, use_noise=True)
        x = torch.randn(50)
        
        views = aug(x, num_views=2)
        
        # Views should be different from original
        assert not torch.allclose(views[0], x)
        assert not torch.allclose(views[1], x)
        
        # Views should differ from each other
        assert not torch.allclose(views[0], views[1])
    
    def test_augmentation_batch_input(self):
        """Test augmentation with batch input."""
        aug = OmicsAugmentation('Prot')
        x = torch.randn(10, 50)
        
        views = aug(x, num_views=2)
        
        assert len(views) == 2
        assert all(v.shape == x.shape for v in views)
    
    def test_no_augmentation_enabled(self):
        """Test pipeline with all augmentations disabled."""
        aug = OmicsAugmentation(
            'Test',
            use_dropout=False,
            use_noise=False,
            use_masking=False
        )
        x = torch.randn(50)
        
        views = aug(x, num_views=2)
        
        # Without augmentation, views should be identical to original
        assert torch.allclose(views[0], x)
        assert torch.allclose(views[1], x)


class TestGetAugmentationPipeline:
    """Test getting modality-specific augmentation pipelines."""
    
    def test_get_known_modality(self):
        """Test getting pipeline for known modality."""
        aug = get_augmentation_pipeline('GeneExp')
        
        assert isinstance(aug, OmicsAugmentation)
        assert aug.modality_name == 'GeneExp'
        assert aug.use_dropout is True
        assert aug.use_noise is True
    
    def test_get_all_modalities(self):
        """Test getting pipelines for all defined modalities."""
        modalities = ['GeneExp', 'miRNA', 'Meth', 'CNV', 'Prot', 'Mut']
        
        for modality in modalities:
            aug = get_augmentation_pipeline(modality)
            assert isinstance(aug, OmicsAugmentation)
            assert aug.modality_name == modality
    
    def test_get_unknown_modality(self):
        """Test getting pipeline for unknown modality uses defaults."""
        aug = get_augmentation_pipeline('UnknownModality')
        
        assert isinstance(aug, OmicsAugmentation)
        assert aug.modality_name == 'UnknownModality'
        # Should use default config
        assert aug.use_dropout is True
