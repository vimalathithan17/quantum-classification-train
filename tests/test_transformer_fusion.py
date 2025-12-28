"""
Tests for multimodal transformer fusion.
"""

import pytest
import torch

from performance_extensions.transformer_fusion import (
    MultimodalTransformer,
    ModalityFeatureEncoder,
    MultimodalFusionClassifier
)


class TestMultimodalTransformer:
    """Test MultimodalTransformer."""
    
    def test_transformer_initialization(self):
        """Test transformer initializes correctly."""
        transformer = MultimodalTransformer(
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            num_modalities=6,
            num_classes=10
        )
        
        assert transformer.embed_dim == 256
        assert transformer.num_heads == 8
        assert transformer.num_layers == 4
        assert transformer.num_modalities == 6
        assert transformer.num_classes == 10
    
    def test_embed_dim_divisible_by_heads(self):
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            MultimodalTransformer(
                embed_dim=250,  # Not divisible by 8
                num_heads=8
            )
    
    def test_forward_pass_without_cls_token(self):
        """Test forward pass without CLS token."""
        transformer = MultimodalTransformer(
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_modalities=3,
            num_classes=5,
            use_cls_token=False
        )
        
        # Create modality features
        modality_features = [
            torch.randn(8, 256),  # Batch size 8
            torch.randn(8, 256),
            torch.randn(8, 256)
        ]
        
        logits, _ = transformer(modality_features)
        
        assert logits.shape == (8, 5)
    
    def test_forward_pass_with_cls_token(self):
        """Test forward pass with CLS token."""
        transformer = MultimodalTransformer(
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_modalities=3,
            num_classes=5,
            use_cls_token=True
        )
        
        # Create modality features
        modality_features = [
            torch.randn(8, 256),
            torch.randn(8, 256),
            torch.randn(8, 256)
        ]
        
        logits, _ = transformer(modality_features)
        
        assert logits.shape == (8, 5)
    
    def test_forward_with_missing_modality_mask(self):
        """Test forward pass with missing modality mask."""
        transformer = MultimodalTransformer(
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_modalities=3,
            num_classes=5,
            use_cls_token=False
        )
        
        modality_features = [
            torch.randn(8, 256),
            torch.randn(8, 256),
            torch.randn(8, 256)
        ]
        
        # Mark second modality as missing for all samples
        modality_masks = torch.zeros(8, 3, dtype=torch.bool)
        modality_masks[:, 1] = True
        
        logits, _ = transformer(modality_features, modality_masks)
        
        assert logits.shape == (8, 5)
        assert not torch.isnan(logits).any()


class TestModalityFeatureEncoder:
    """Test ModalityFeatureEncoder."""
    
    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        encoder = ModalityFeatureEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(8, 100)
        
        output = encoder(x, is_missing=False)
        
        assert output.shape == (8, 256)
    
    def test_encoder_missing_token(self):
        """Test encoder returns missing token when modality is missing."""
        encoder = ModalityFeatureEncoder(input_dim=100, embed_dim=256)
        
        # Request missing token for batch of 8
        x = torch.randn(8, 100)
        output = encoder(x, is_missing=True)
        
        assert output.shape == (8, 256)
        
        # All samples should get the same missing token
        assert torch.allclose(output[0], output[1])
    
    def test_encoder_missing_with_none_input(self):
        """Test encoder handles None input for missing modality."""
        encoder = ModalityFeatureEncoder(input_dim=100, embed_dim=256)
        
        output = encoder(None, is_missing=True)
        
        assert output.shape == (1, 256)
    
    def test_encoder_forward_pass(self):
        """Test encoder forward pass without errors."""
        encoder = ModalityFeatureEncoder(input_dim=50, embed_dim=128)
        x = torch.randn(4, 50)
        
        output = encoder(x, is_missing=False)
        
        assert not torch.isnan(output).any()


class TestMultimodalFusionClassifier:
    """Test MultimodalFusionClassifier."""
    
    def test_classifier_initialization(self):
        """Test classifier initializes correctly."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80, 'Meth': 120}
        
        classifier = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_classes=5
        )
        
        assert len(classifier.encoders) == 3
        assert classifier.num_modalities == 3
        assert classifier.num_classes == 5
    
    def test_forward_all_modalities_present(self):
        """Test forward pass with all modalities present."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80}
        
        classifier = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_classes=5
        )
        
        modality_data = {
            'GeneExpr': torch.randn(8, 100),
            'Prot': torch.randn(8, 80)
        }
        
        logits, _ = classifier(modality_data)
        
        assert logits.shape == (8, 5)
        assert not torch.isnan(logits).any()
    
    def test_forward_with_missing_modality(self):
        """Test forward pass with one modality missing."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80, 'Meth': 120}
        
        classifier = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_classes=5
        )
        
        # Only provide GeneExpr and Prot, Meth is missing
        modality_data = {
            'GeneExpr': torch.randn(8, 100),
            'Prot': torch.randn(8, 80)
        }
        
        logits, _ = classifier(modality_data)
        
        assert logits.shape == (8, 5)
        assert not torch.isnan(logits).any()
    
    def test_forward_with_none_modality(self):
        """Test forward pass with None for missing modality."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80}
        
        classifier = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_classes=5
        )
        
        modality_data = {
            'GeneExpr': torch.randn(8, 100),
            'Prot': None  # Explicitly missing
        }
        
        logits, _ = classifier(modality_data)
        
        assert logits.shape == (8, 5)
        assert not torch.isnan(logits).any()
    
    def test_forward_with_missing_mask(self):
        """Test forward pass with explicit missing mask."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80}
        
        classifier = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_classes=5
        )
        
        modality_data = {
            'GeneExpr': torch.randn(8, 100),
            'Prot': torch.randn(8, 80)
        }
        
        modality_missing = {
            'GeneExpr': False,
            'Prot': True  # Mark Prot as missing
        }
        
        logits, _ = classifier(modality_data, modality_missing)
        
        assert logits.shape == (8, 5)
        assert not torch.isnan(logits).any()
    
    def test_with_pretrained_encoders(self):
        """Test classifier with pretrained encoders."""
        from performance_extensions.contrastive_learning import ModalityEncoder
        import torch.nn as nn
        
        modality_dims = {'GeneExpr': 100, 'Prot': 80}
        
        # Create pretrained encoders
        pretrained_encoders = nn.ModuleDict({
            'GeneExpr': ModalityEncoder(100, 256),
            'Prot': ModalityEncoder(80, 256)
        })
        
        classifier = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=256,
            num_heads=8,
            num_layers=2,
            num_classes=5,
            pretrained_encoders=pretrained_encoders
        )
        
        # Verify it uses the pretrained encoders
        assert classifier.encoders['GeneExpr'] is pretrained_encoders['GeneExpr']
        assert classifier.encoders['Prot'] is pretrained_encoders['Prot']
        
        # Test forward pass
        modality_data = {
            'GeneExpr': torch.randn(8, 100),
            'Prot': torch.randn(8, 80)
        }
        
        logits, _ = classifier(modality_data)
        
        assert logits.shape == (8, 5)
