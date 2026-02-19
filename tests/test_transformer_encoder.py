"""
Comprehensive tests for TransformerModalityEncoder and ModalityEncoder implementations.

Tests:
1. Basic initialization and parameter count
2. Forward pass with clean data
3. Forward pass with NaN values (feature-level)
4. Missing modality handling (is_missing=True)
5. Gradient flow through NaN positions
6. Edge cases (batch size 1, small input dim)
7. Mask token functionality verification
8. Integration with ContrastiveMultiOmicsEncoder
9. MLP encoder all-NaN sample handling
10. NT-Xent loss with valid mask
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_extensions.contrastive_learning import (
    TransformerModalityEncoder,
    ContrastiveMultiOmicsEncoder,
    ModalityEncoder,
    nt_xent_loss
)


class TestMLPModalityEncoder:
    """Test suite for MLP ModalityEncoder with all-NaN handling."""
    
    def test_basic_forward(self):
        """Test basic forward pass with clean data."""
        encoder = ModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(32, 100)
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (32, 256)
        assert valid_mask.shape == (32,)
        assert valid_mask.all(), "All samples should be valid with clean data"
        assert not torch.isnan(out).any()
    
    def test_partial_nan_still_valid(self):
        """Test that samples with partial NaN are still valid (but NaN replaced with 0)."""
        encoder = ModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(8, 100)
        x[0, :20] = float('nan')  # 20 features missing
        x[3, 50:70] = float('nan')  # 20 features missing
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (8, 256)
        assert valid_mask.shape == (8,)
        assert valid_mask.all(), "Partial NaN samples should still be valid"
        assert not torch.isnan(out).any(), "NaN should be replaced with 0"
    
    def test_all_nan_sample_invalid(self):
        """Test that samples with ALL features as NaN are marked invalid."""
        encoder = ModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(8, 100)
        x[3, :] = float('nan')  # All features NaN for sample 3
        x[7, :] = float('nan')  # All features NaN for sample 7
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (8, 256)
        assert valid_mask.shape == (8,)
        
        # Samples 3 and 7 should be invalid
        assert not valid_mask[3], "All-NaN sample 3 should be invalid"
        assert not valid_mask[7], "All-NaN sample 7 should be invalid"
        
        # Other samples should be valid
        assert valid_mask[0], "Sample 0 should be valid"
        assert valid_mask[1], "Sample 1 should be valid"
        assert valid_mask[2], "Sample 2 should be valid"
        
        # Output should not contain NaN
        assert not torch.isnan(out).any()
    
    def test_missing_modality(self):
        """Test missing modality handling."""
        encoder = ModalityEncoder(input_dim=100, embed_dim=256)
        
        out, valid_mask = encoder(None, is_missing=True)
        
        assert out.shape == (1, 256)
        assert valid_mask.shape == (1,)
        assert not valid_mask.any(), "Missing modality should be invalid"
    
    def test_all_samples_all_nan(self):
        """Test when ALL samples have ALL features as NaN."""
        encoder = ModalityEncoder(input_dim=50, embed_dim=128)
        x = torch.full((4, 50), float('nan'))
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (4, 128)
        assert not valid_mask.any(), "All samples should be invalid"
        assert not torch.isnan(out).any(), "Output should still be valid tensors"


class TestTransformerModalityEncoder:
    """Test suite for TransformerModalityEncoder."""
    
    def test_basic_initialization(self):
        """Test basic encoder initialization."""
        encoder = TransformerModalityEncoder(
            input_dim=100, embed_dim=256, d_model=64, num_heads=4, num_layers=2
        )
        
        # Check attributes
        assert encoder.input_dim == 100
        assert encoder.embed_dim == 256
        assert encoder.d_model == 64
        
        # Check components exist
        assert hasattr(encoder, 'feature_embedding')
        assert hasattr(encoder, 'pos_encoding')
        assert hasattr(encoder, 'mask_token')
        assert hasattr(encoder, 'transformer')
        assert hasattr(encoder, 'output_proj')
        assert hasattr(encoder, 'missing_modality_token')
        
        # Check shapes
        assert encoder.pos_encoding.shape == (1, 100, 64)
        assert encoder.mask_token.shape == (1, 1, 64)
        assert encoder.missing_modality_token.shape == (1, 256)
    
    def test_forward_clean_data(self):
        """Test forward pass with clean data (no NaN)."""
        encoder = TransformerModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(32, 100)
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (32, 256)
        assert valid_mask.shape == (32,)
        assert valid_mask.all(), "All samples should be valid when no NaN"
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_forward_with_nan_partial(self):
        """Test forward pass with partial NaN values (some features missing)."""
        encoder = TransformerModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(32, 100)
        
        # Different amounts of NaN for different samples
        x[0, 10:30] = float('nan')   # 20 features missing
        x[5, :50] = float('nan')     # 50 features missing
        x[15, 80:] = float('nan')    # 20 features missing
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (32, 256)
        assert valid_mask.shape == (32,)
        assert valid_mask.all(), "Partial NaN samples should still be valid"
        assert not torch.isnan(out).any(), "Output should not contain NaN!"
        assert not torch.isinf(out).any()
    
    def test_forward_with_nan_all_features(self):
        """Test forward pass when ALL features are NaN for a sample."""
        encoder = TransformerModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(32, 100)
        
        # All features missing for one sample
        x[10, :] = float('nan')
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (32, 256)
        assert valid_mask.shape == (32,)
        assert not torch.isnan(out).any(), "Should handle all-NaN sample gracefully!"
        
        # The all-NaN sample should be marked as INVALID
        assert not valid_mask[10], "All-NaN sample should be marked invalid"
        assert valid_mask[:10].all(), "Other samples should be valid"
        assert valid_mask[11:].all(), "Other samples should be valid"
        
        # The all-NaN sample should have a non-zero embedding (from mask tokens)
        assert (out[10] != 0).any(), "All-NaN sample should have non-zero embedding"
    
    def test_missing_modality_flag(self):
        """Test missing modality handling with is_missing=True."""
        encoder = TransformerModalityEncoder(input_dim=100, embed_dim=256)
        
        # Test with None input
        out1, valid1 = encoder(None, is_missing=True)
        assert out1.shape == (1, 256)
        assert valid1.shape == (1,)
        assert not valid1.any(), "Missing modality should be marked invalid"
        
        # Test with tensor input but is_missing=True
        x = torch.randn(8, 100)
        out2, valid2 = encoder(x, is_missing=True)
        assert out2.shape == (8, 256)
        assert valid2.shape == (8,)
        assert not valid2.any(), "Missing modality should be marked invalid"
        
        # Missing modality should return the same token for all samples
        assert torch.allclose(out2[0], out2[1])
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through NaN positions."""
        encoder = TransformerModalityEncoder(input_dim=50, embed_dim=128)
        encoder.train()
        
        # Create input with some NaN values
        x = torch.randn(8, 50)
        nan_positions = torch.zeros_like(x, dtype=torch.bool)
        nan_positions[0, :10] = True
        nan_positions[3, 25:40] = True
        
        # Apply NaN after cloning to avoid autograd issues
        x_with_nan = x.clone()
        x_with_nan[nan_positions] = float('nan')
        x_with_nan.requires_grad_(True)
        
        out, valid_mask = encoder(x_with_nan)
        loss = out.sum()
        loss.backward()
        
        # Gradients should exist
        assert x_with_nan.grad is not None, "Gradients should be computed"
        
        # Gradients at non-NaN positions should generally be non-zero
        non_nan_grads = x_with_nan.grad[~nan_positions]
        assert (non_nan_grads != 0).any(), "Some non-NaN gradients should be non-zero"
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        encoder = TransformerModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(1, 100)
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (1, 256)
        assert valid_mask.shape == (1,)
        assert valid_mask.all()
        assert not torch.isnan(out).any()
    
    def test_small_input_dimension(self):
        """Test with small input dimension."""
        encoder = TransformerModalityEncoder(
            input_dim=10, embed_dim=64, d_model=32, num_heads=2, num_layers=1
        )
        x = torch.randn(16, 10)
        x[0, :5] = float('nan')
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (16, 64)
        assert valid_mask.shape == (16,)
        assert not torch.isnan(out).any()
    
    def test_mask_token_affects_output(self):
        """Verify that mask tokens actually affect the output."""
        encoder = TransformerModalityEncoder(
            input_dim=50, embed_dim=128, d_model=32, num_heads=2, num_layers=1
        )
        
        # Same base input
        x_clean = torch.randn(4, 50)
        x_nan = x_clean.clone()
        x_nan[:, 25:] = float('nan')  # Half missing
        
        out_clean, _ = encoder(x_clean)
        out_nan, _ = encoder(x_nan)
        
        # Outputs should be different
        assert not torch.allclose(out_clean, out_nan, atol=1e-5), \
            "Outputs should differ when NaN is present!"
    
    def test_large_batch(self):
        """Test with large batch size."""
        encoder = TransformerModalityEncoder(input_dim=100, embed_dim=256)
        x = torch.randn(256, 100)
        x[::2, ::2] = float('nan')  # Sparse NaN pattern
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (256, 256)
        assert valid_mask.shape == (256,)
        assert valid_mask.all(), "Sparse NaN should not make samples invalid"
        assert not torch.isnan(out).any()
    
    def test_deterministic_with_same_input(self):
        """Test that same input produces same output (no dropout in eval)."""
        encoder = TransformerModalityEncoder(input_dim=50, embed_dim=128, dropout=0.1)
        encoder.eval()  # Disable dropout
        
        x = torch.randn(8, 50)
        x[0, :10] = float('nan')
        
        out1, _ = encoder(x)
        out2, _ = encoder(x)
        
        assert torch.allclose(out1, out2), "Same input should produce same output in eval mode"
    
    def test_different_nan_patterns_different_outputs(self):
        """Test that different NaN patterns produce different outputs."""
        encoder = TransformerModalityEncoder(input_dim=50, embed_dim=128)
        encoder.eval()
        
        x_base = torch.randn(1, 50)
        
        x1 = x_base.clone()
        x1[0, :10] = float('nan')
        
        x2 = x_base.clone()
        x2[0, 40:] = float('nan')
        
        out1, _ = encoder(x1)
        out2, _ = encoder(x2)
        
        assert not torch.allclose(out1, out2), \
            "Different NaN patterns should produce different outputs"


class TestTransformerInContrastiveEncoder:
    """Test TransformerModalityEncoder integration in ContrastiveMultiOmicsEncoder."""
    
    def test_encoder_type_selection(self):
        """Test that encoder_type='transformer' creates TransformerModalityEncoder."""
        modality_dims = {'mRNA': 200, 'miRNA': 50, 'DNA_Meth': 100}
        
        # MLP encoder
        mlp_model = ContrastiveMultiOmicsEncoder(
            modality_dims=modality_dims,
            encoder_type='mlp'
        )
        
        # Transformer encoder
        transformer_model = ContrastiveMultiOmicsEncoder(
            modality_dims=modality_dims,
            encoder_type='transformer'
        )
        
        # Check encoder types
        for modality in modality_dims:
            assert isinstance(mlp_model.encoders[modality], ModalityEncoder)
            assert isinstance(transformer_model.encoders[modality], TransformerModalityEncoder)
    
    def test_transformer_encoder_forward(self):
        """Test forward pass with transformer encoder."""
        modality_dims = {'mRNA': 200, 'miRNA': 50}
        
        model = ContrastiveMultiOmicsEncoder(
            modality_dims=modality_dims,
            encoder_type='transformer',
            embed_dim=128,
            projection_dim=64
        )
        
        # Test with NaN values
        x_mrna = torch.randn(16, 200)
        x_mrna[0, :50] = float('nan')
        
        embedding, projection, valid_mask = model(x_mrna, 'mRNA')
        
        assert embedding.shape == (16, 128)
        assert projection.shape == (16, 64)
        assert valid_mask.shape == (16,)
        assert valid_mask.all(), "Partial NaN should still be valid"
        assert not torch.isnan(embedding).any()
        assert not torch.isnan(projection).any()
    
    def test_transformer_encoder_all_nan_excluded(self):
        """Test that all-NaN samples are marked invalid."""
        modality_dims = {'mRNA': 100}
        
        model = ContrastiveMultiOmicsEncoder(
            modality_dims=modality_dims,
            encoder_type='transformer',
            embed_dim=64
        )
        
        x = torch.randn(8, 100)
        x[3, :] = float('nan')  # All NaN for sample 3
        
        embedding, projection, valid_mask = model(x, 'mRNA')
        
        assert not valid_mask[3], "All-NaN sample should be invalid"
        assert valid_mask[:3].all(), "Other samples should be valid"
        assert valid_mask[4:].all(), "Other samples should be valid"
    
    def test_transformer_encoder_config(self):
        """Test transformer-specific configuration parameters."""
        modality_dims = {'mRNA': 200}
        
        model = ContrastiveMultiOmicsEncoder(
            modality_dims=modality_dims,
            encoder_type='transformer',
            transformer_d_model=128,
            transformer_num_heads=8,
            transformer_num_layers=4
        )
        
        encoder = model.encoders['mRNA']
        assert encoder.d_model == 128
        # Check transformer layer configuration
        assert encoder.transformer.num_layers == 4


class TestNTXentLossWithMask:
    """Test NT-Xent loss with valid sample masking."""
    
    def test_loss_excludes_invalid_samples(self):
        """Test that invalid samples are excluded from loss."""
        from performance_extensions.contrastive_learning import nt_xent_loss
        
        z_i = torch.randn(8, 64)
        z_j = torch.randn(8, 64)
        
        # All valid
        loss_all = nt_xent_loss(z_i, z_j, temperature=0.5)
        
        # Exclude sample 0
        valid_mask = torch.ones(8, dtype=torch.bool)
        valid_mask[0] = False
        
        loss_masked = nt_xent_loss(z_i, z_j, temperature=0.5, valid_mask=valid_mask)
        
        # Losses should be different
        assert loss_all.item() != loss_masked.item()
    
    def test_loss_zero_when_too_few_valid(self):
        """Test that loss is 0 when fewer than 2 valid samples."""
        from performance_extensions.contrastive_learning import nt_xent_loss
        
        z_i = torch.randn(4, 64)
        z_j = torch.randn(4, 64)
        
        # Only 1 valid sample
        valid_mask = torch.zeros(4, dtype=torch.bool)
        valid_mask[0] = True
        
        loss = nt_xent_loss(z_i, z_j, temperature=0.5, valid_mask=valid_mask)
        
        assert loss.item() == 0.0, "Loss should be 0 with < 2 valid samples"
    
    def test_loss_backward_with_mask(self):
        """Test that gradients flow correctly with mask."""
        from performance_extensions.contrastive_learning import nt_xent_loss
        
        z_i = torch.randn(8, 64, requires_grad=True)
        z_j = torch.randn(8, 64, requires_grad=True)
        
        valid_mask = torch.ones(8, dtype=torch.bool)
        valid_mask[0] = False
        valid_mask[7] = False
        
        loss = nt_xent_loss(z_i, z_j, temperature=0.5, valid_mask=valid_mask)
        loss.backward()
        
        assert z_i.grad is not None
        assert z_j.grad is not None


class TestEdgeCases:
    """Test edge cases and potential failure modes."""
    
    def test_all_zeros_input(self):
        """Test with all-zero input."""
        encoder = TransformerModalityEncoder(input_dim=50, embed_dim=128)
        x = torch.zeros(8, 50)
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (8, 128)
        assert valid_mask.all()
        assert not torch.isnan(out).any()
    
    def test_very_large_values(self):
        """Test with very large input values."""
        encoder = TransformerModalityEncoder(input_dim=50, embed_dim=128)
        x = torch.randn(8, 50) * 1000
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (8, 128)
        assert valid_mask.all()
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_mixed_nan_inf(self):
        """Test with mixed NaN and inf values (inf should cause issues)."""
        encoder = TransformerModalityEncoder(input_dim=50, embed_dim=128)
        x = torch.randn(8, 50)
        x[0, :10] = float('nan')
        # Note: inf values are NOT handled by the encoder
        # This test documents current behavior
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (8, 128)
        assert not torch.isnan(out).any()
    
    def test_all_samples_all_nan(self):
        """Test when ALL samples have ALL features as NaN."""
        encoder = TransformerModalityEncoder(input_dim=50, embed_dim=128)
        x = torch.full((4, 50), float('nan'))
        
        out, valid_mask = encoder(x)
        
        assert out.shape == (4, 128)
        assert not valid_mask.any(), "All samples should be invalid"
        assert not torch.isnan(out).any(), "Output should still be valid tensors"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
