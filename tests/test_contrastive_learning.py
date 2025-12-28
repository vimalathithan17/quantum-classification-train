"""
Tests for contrastive learning components.
"""

import pytest
import torch
import torch.nn as nn

from performance_extensions.contrastive_learning import (
    ModalityEncoder,
    ProjectionHead,
    ContrastiveMultiOmicsEncoder,
    nt_xent_loss,
    cross_modal_contrastive_loss,
    ContrastiveLearningLoss
)


class TestModalityEncoder:
    """Test ModalityEncoder."""
    
    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        encoder = ModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(8, 100)
        
        output = encoder(x)
        
        assert output.shape == (8, 256)
    
    def test_encoder_forward_pass(self):
        """Test encoder forward pass runs without errors."""
        encoder = ModalityEncoder(input_dim=50, embed_dim=128)
        x = torch.randn(4, 50)
        
        output = encoder(x)
        
        assert output is not None
        assert not torch.isnan(output).any()


class TestProjectionHead:
    """Test ProjectionHead."""
    
    def test_projection_output_shape(self):
        """Test projection head produces correct output shape."""
        proj = ProjectionHead(embed_dim=256, projection_dim=128)
        x = torch.randn(8, 256)
        
        output = proj(x)
        
        assert output.shape == (8, 128)
    
    def test_projection_forward_pass(self):
        """Test projection head forward pass."""
        proj = ProjectionHead(embed_dim=128, projection_dim=64)
        x = torch.randn(4, 128)
        
        output = proj(x)
        
        assert output is not None
        assert not torch.isnan(output).any()


class TestContrastiveMultiOmicsEncoder:
    """Test ContrastiveMultiOmicsEncoder."""
    
    def test_encoder_initialization(self):
        """Test multi-omics encoder initializes correctly."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80, 'Meth': 120}
        
        encoder = ContrastiveMultiOmicsEncoder(
            modality_dims=modality_dims,
            embed_dim=256,
            projection_dim=128
        )
        
        assert len(encoder.encoders) == 3
        assert len(encoder.projection_heads) == 3
        assert 'GeneExpr' in encoder.encoders
        assert 'Prot' in encoder.encoders
        assert 'Meth' in encoder.encoders
    
    def test_encode_single_modality(self):
        """Test encoding a single modality."""
        modality_dims = {'GeneExpr': 100}
        encoder = ContrastiveMultiOmicsEncoder(modality_dims, embed_dim=256)
        
        x = torch.randn(8, 100)
        output = encoder.encode(x, 'GeneExpr')
        
        assert output.shape == (8, 256)
    
    def test_forward_with_projection(self):
        """Test forward pass with projection."""
        modality_dims = {'Prot': 80}
        encoder = ContrastiveMultiOmicsEncoder(
            modality_dims,
            embed_dim=256,
            projection_dim=128
        )
        
        x = torch.randn(8, 80)
        embedding, projection = encoder(x, 'Prot', return_projection=True)
        
        assert embedding.shape == (8, 256)
        assert projection.shape == (8, 128)
    
    def test_forward_without_projection(self):
        """Test forward pass without projection."""
        modality_dims = {'Meth': 120}
        encoder = ContrastiveMultiOmicsEncoder(modality_dims, embed_dim=256)
        
        x = torch.randn(8, 120)
        embedding, projection = encoder(x, 'Meth', return_projection=False)
        
        assert embedding.shape == (8, 256)
        assert projection is None
    
    def test_unknown_modality_error(self):
        """Test that unknown modality raises error."""
        modality_dims = {'GeneExpr': 100}
        encoder = ContrastiveMultiOmicsEncoder(modality_dims, embed_dim=256)
        
        x = torch.randn(8, 100)
        
        with pytest.raises(ValueError, match="Unknown modality"):
            encoder.encode(x, 'UnknownModality')


class TestNTXentLoss:
    """Test NT-Xent loss function."""
    
    def test_loss_output_is_scalar(self):
        """Test that loss returns a scalar."""
        z_i = torch.randn(16, 128)
        z_j = torch.randn(16, 128)
        
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
    
    def test_identical_projections_low_loss(self):
        """Test that identical projections give low loss."""
        z = torch.randn(16, 128)
        
        # Same projections should give low loss
        loss = nt_xent_loss(z, z, temperature=0.5)
        
        # Loss should be relatively low (not zero due to other negatives)
        assert loss.item() < 5.0
    
    def test_orthogonal_projections_high_loss(self):
        """Test that orthogonal projections give higher loss."""
        # Create orthogonal vectors
        z_i = torch.randn(16, 128)
        z_j = torch.randn(16, 128)
        
        # Make them more orthogonal by normalizing
        z_i = torch.nn.functional.normalize(z_i, dim=1)
        z_j = torch.nn.functional.normalize(z_j, dim=1)
        
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        
        assert loss.item() > 0
    
    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        z_i = torch.randn(16, 128)
        z_j = torch.randn(16, 128)
        
        loss_low_temp = nt_xent_loss(z_i, z_j, temperature=0.1)
        loss_high_temp = nt_xent_loss(z_i, z_j, temperature=1.0)
        
        # Different temperatures should give different losses
        assert not torch.isclose(loss_low_temp, loss_high_temp)


class TestCrossModalContrastiveLoss:
    """Test cross-modal contrastive loss."""
    
    def test_cross_modal_loss_output(self):
        """Test cross-modal loss returns scalar."""
        embed_1 = torch.randn(16, 256)
        embed_2 = torch.randn(16, 256)
        proj_head_1 = ProjectionHead(256, 128)
        proj_head_2 = ProjectionHead(256, 128)
        
        loss = cross_modal_contrastive_loss(
            embed_1, embed_2, proj_head_1, proj_head_2, temperature=0.5
        )
        
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestContrastiveLearningLoss:
    """Test ContrastiveLearningLoss module."""
    
    def test_loss_initialization(self):
        """Test loss module initializes correctly."""
        loss_fn = ContrastiveLearningLoss(
            temperature=0.5,
            use_cross_modal=True
        )
        
        assert loss_fn.temperature == 0.5
        assert loss_fn.use_cross_modal is True
    
    def test_intra_modal_loss_only(self):
        """Test loss with only intra-modal component."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80}
        model = ContrastiveMultiOmicsEncoder(modality_dims, embed_dim=256, projection_dim=128)
        
        loss_fn = ContrastiveLearningLoss(temperature=0.5, use_cross_modal=False)
        
        # Create augmented views
        augmented_views = {
            'GeneExpr': [torch.randn(8, 100), torch.randn(8, 100)],
            'Prot': [torch.randn(8, 80), torch.randn(8, 80)]
        }
        
        # Create embeddings and projections
        embeddings = {}
        projections = {}
        for modality, views in augmented_views.items():
            emb, proj = model(views[0], modality, return_projection=True)
            embeddings[modality] = emb
            projections[modality] = proj
        
        total_loss, loss_dict = loss_fn(augmented_views, embeddings, projections, model)
        
        assert total_loss.item() >= 0
        assert 'intra_GeneExpr' in loss_dict
        assert 'intra_Prot' in loss_dict
        # Should not have cross-modal losses
        assert not any('cross' in key for key in loss_dict.keys())
    
    def test_cross_modal_loss_included(self):
        """Test loss with cross-modal component."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80}
        model = ContrastiveMultiOmicsEncoder(modality_dims, embed_dim=256, projection_dim=128)
        
        loss_fn = ContrastiveLearningLoss(temperature=0.5, use_cross_modal=True)
        
        # Create augmented views
        augmented_views = {
            'GeneExpr': [torch.randn(8, 100), torch.randn(8, 100)],
            'Prot': [torch.randn(8, 80), torch.randn(8, 80)]
        }
        
        # Create embeddings and projections
        embeddings = {}
        projections = {}
        for modality, views in augmented_views.items():
            emb, proj = model(views[0], modality, return_projection=True)
            embeddings[modality] = emb
            projections[modality] = proj
        
        total_loss, loss_dict = loss_fn(augmented_views, embeddings, projections, model)
        
        assert total_loss.item() >= 0
        # Should have both intra and cross-modal losses
        assert 'intra_GeneExpr' in loss_dict
        assert 'intra_Prot' in loss_dict
        assert any('cross' in key for key in loss_dict.keys())
    
    def test_specified_cross_modal_pairs(self):
        """Test loss with specified cross-modal pairs."""
        modality_dims = {'GeneExpr': 100, 'Prot': 80, 'Meth': 120}
        model = ContrastiveMultiOmicsEncoder(modality_dims, embed_dim=256, projection_dim=128)
        
        # Only use GeneExpr-Prot pair
        loss_fn = ContrastiveLearningLoss(
            temperature=0.5,
            use_cross_modal=True,
            cross_modal_pairs=[('GeneExpr', 'Prot')]
        )
        
        # Create augmented views for all modalities
        augmented_views = {
            'GeneExpr': [torch.randn(8, 100), torch.randn(8, 100)],
            'Prot': [torch.randn(8, 80), torch.randn(8, 80)],
            'Meth': [torch.randn(8, 120), torch.randn(8, 120)]
        }
        
        # Create embeddings and projections
        embeddings = {}
        projections = {}
        for modality, views in augmented_views.items():
            emb, proj = model(views[0], modality, return_projection=True)
            embeddings[modality] = emb
            projections[modality] = proj
        
        total_loss, loss_dict = loss_fn(augmented_views, embeddings, projections, model)
        
        assert total_loss.item() >= 0
        # Should have cross_GeneExpr_Prot but not other cross-modal pairs
        assert 'cross_GeneExpr_Prot' in loss_dict
        assert 'cross_GeneExpr_Meth' not in loss_dict
        assert 'cross_Prot_Meth' not in loss_dict
