"""
Tests for the transformer fusion hyperparameter tuning script.
"""

import os
import sys
import tempfile
import shutil
import sqlite3

import pytest
import numpy as np
import torch

# Add parent and examples directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples'))

# Check if dependencies are available
try:
    from examples.tune_transformer_fusion import (
        is_db_writable,
        ensure_writable_db,
        set_seed,
        MultiOmicsDataset,
        create_dataloader,
        train_epoch,
        evaluate,
        load_pretrained_embeddings,
        load_multiomics_data
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import from tune_transformer_fusion: {e}")
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="tune_transformer_fusion imports not available")
class TestDatabaseUtilities:
    """Test database utility functions."""
    
    def test_is_db_writable_nonexistent(self):
        """Test checking writability of a non-existent database in writable directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            assert is_db_writable(db_path), "Should be writable in temp directory"
    
    def test_is_db_writable_existing(self):
        """Test checking writability of an existing database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            # Create a database file
            conn = sqlite3.connect(db_path)
            conn.close()
            
            assert is_db_writable(db_path), "Should be writable in temp directory"
            
            # Make it read-only
            os.chmod(db_path, 0o444)
            assert not is_db_writable(db_path), "Should not be writable when read-only"
            
            # Restore permissions for cleanup
            os.chmod(db_path, 0o644)
    
    def test_ensure_writable_db_already_writable(self):
        """Test ensure_writable_db when database is already writable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            # Create a database file
            conn = sqlite3.connect(db_path)
            conn.close()
            
            result = ensure_writable_db(db_path)
            assert result == db_path, "Should return original path when writable"
    
    def test_ensure_writable_db_readonly_fallback(self):
        """Test ensure_writable_db falls back when database is read-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            # Create a database file
            conn = sqlite3.connect(db_path)
            conn.close()
            
            # Make it read-only
            os.chmod(db_path, 0o444)
            
            result = ensure_writable_db(db_path)
            assert result != db_path, "Should return different path when original is read-only"
            # Note: In some CI environments, all candidate paths may be read-only,
            # so :memory: is an acceptable fallback
            assert result is not None, "Should return some path"
            
            # Restore permissions for cleanup
            os.chmod(db_path, 0o644)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="tune_transformer_fusion imports not available")
class TestPretrainedEmbeddings:
    """Test pretrained embeddings loading."""
    
    def test_load_pretrained_embeddings(self):
        """Test loading embeddings from numpy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock embedding files
            n_samples = 50
            embed_dim = 64
            
            np.save(os.path.join(tmpdir, 'GeneExpr_embeddings.npy'), 
                   np.random.randn(n_samples, embed_dim).astype(np.float32))
            np.save(os.path.join(tmpdir, 'miRNA_embeddings.npy'), 
                   np.random.randn(n_samples, embed_dim).astype(np.float32))
            np.save(os.path.join(tmpdir, 'labels.npy'), 
                   np.array(['A', 'B'] * 25))  # String labels
            
            data, labels, modality_dims = load_pretrained_embeddings(
                tmpdir, modalities=['GeneExpr', 'miRNA'])
            
            assert data is not None
            assert 'GeneExpr' in data
            assert 'miRNA' in data
            assert data['GeneExpr'].shape == (n_samples, embed_dim)
            assert data['miRNA'].shape == (n_samples, embed_dim)
            assert len(labels) == n_samples
            assert modality_dims == {'GeneExpr': embed_dim, 'miRNA': embed_dim}
    
    def test_load_pretrained_embeddings_numeric_labels(self):
        """Test loading embeddings with numeric labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 30
            embed_dim = 32
            
            np.save(os.path.join(tmpdir, 'GeneExpr_embeddings.npy'), 
                   np.random.randn(n_samples, embed_dim).astype(np.float32))
            np.save(os.path.join(tmpdir, 'labels.npy'), 
                   np.array([0, 1, 2] * 10))  # Numeric labels
            
            data, labels, modality_dims = load_pretrained_embeddings(
                tmpdir, modalities=['GeneExpr'])
            
            assert data is not None
            assert len(labels) == n_samples
            assert set(labels) == {0, 1, 2}
    
    def test_load_pretrained_embeddings_missing_labels(self):
        """Test that missing labels file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, 'GeneExpr_embeddings.npy'), 
                   np.random.randn(10, 32).astype(np.float32))
            # No labels.npy file
            
            data, labels, modality_dims = load_pretrained_embeddings(
                tmpdir, modalities=['GeneExpr'])
            
            # Should return None for labels since labels.npy is missing
            assert labels is None
    
    def test_load_pretrained_embeddings_missing_modality(self):
        """Test loading with missing modality files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, 'GeneExpr_embeddings.npy'), 
                   np.random.randn(20, 64).astype(np.float32))
            np.save(os.path.join(tmpdir, 'labels.npy'), np.array([0, 1] * 10))
            # No miRNA_embeddings.npy file
            
            data, labels, modality_dims = load_pretrained_embeddings(
                tmpdir, modalities=['GeneExpr', 'miRNA'])
            
            assert 'GeneExpr' in data
            assert 'miRNA' not in data  # Missing modality skipped
            assert modality_dims == {'GeneExpr': 64}


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="tune_transformer_fusion imports not available")
class TestSeedUtility:
    """Test seed setting utility."""
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible random numbers."""
        set_seed(42)
        rand1 = np.random.rand(10)
        torch_rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = np.random.rand(10)
        torch_rand2 = torch.randn(10)
        
        np.testing.assert_array_equal(rand1, rand2, "NumPy random should be reproducible")
        assert torch.allclose(torch_rand1, torch_rand2), "PyTorch random should be reproducible"
    
    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different random numbers."""
        set_seed(42)
        rand1 = np.random.rand(10)
        
        set_seed(123)
        rand2 = np.random.rand(10)
        
        assert not np.array_equal(rand1, rand2), "Different seeds should produce different random numbers"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="tune_transformer_fusion imports not available")
class TestMultiOmicsDataset:
    """Test MultiOmicsDataset class."""
    
    def test_dataset_creation(self):
        """Test dataset can be created from numpy arrays."""
        data = {
            'GeneExpr': np.random.randn(100, 50).astype(np.float32),
            'miRNA': np.random.randn(100, 30).astype(np.float32)
        }
        labels = np.random.randint(0, 2, size=100)
        
        dataset = MultiOmicsDataset(data, labels)
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format."""
        data = {
            'GeneExpr': np.random.randn(100, 50).astype(np.float32),
            'miRNA': np.random.randn(100, 30).astype(np.float32)
        }
        labels = np.random.randint(0, 2, size=100)
        
        dataset = MultiOmicsDataset(data, labels)
        sample_data, sample_label = dataset[0]
        
        assert isinstance(sample_data, dict)
        assert 'GeneExpr' in sample_data
        assert 'miRNA' in sample_data
        assert isinstance(sample_data['GeneExpr'], torch.Tensor)
        assert sample_data['GeneExpr'].shape == (50,)
        assert sample_data['miRNA'].shape == (30,)
        assert isinstance(sample_label, torch.Tensor)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="tune_transformer_fusion imports not available")
class TestDataLoader:
    """Test dataloader creation."""
    
    def test_create_dataloader(self):
        """Test dataloader is created correctly."""
        data = {
            'GeneExpr': np.random.randn(100, 50).astype(np.float32),
            'miRNA': np.random.randn(100, 30).astype(np.float32)
        }
        labels = np.random.randint(0, 2, size=100)
        
        dataloader = create_dataloader(data, labels, batch_size=16, shuffle=True)
        
        assert dataloader.batch_size == 16
        
        # Test iteration
        batch_data, batch_labels = next(iter(dataloader))
        assert isinstance(batch_data, dict)
        assert batch_data['GeneExpr'].shape[0] <= 16  # batch size
        assert batch_data['GeneExpr'].shape[1] == 50  # features
        assert batch_labels.shape[0] <= 16


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="tune_transformer_fusion imports not available")
class TestTrainingFunctions:
    """Test training and evaluation functions."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and data for testing."""
        from performance_extensions.transformer_fusion import MultimodalFusionClassifier
        
        modality_dims = {'GeneExpr': 50, 'miRNA': 30}
        model = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            num_classes=2,
            dropout=0.1
        )
        
        data = {
            'GeneExpr': np.random.randn(32, 50).astype(np.float32),
            'miRNA': np.random.randn(32, 30).astype(np.float32)
        }
        labels = np.random.randint(0, 2, size=32)
        
        return model, data, labels
    
    def test_train_epoch(self, model_and_data):
        """Test train_epoch runs without error."""
        model, data, labels = model_and_data
        device = torch.device('cpu')
        model.to(device)
        
        dataloader = create_dataloader(data, labels, batch_size=8, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)
    
    def test_evaluate(self, model_and_data):
        """Test evaluate runs without error."""
        model, data, labels = model_and_data
        device = torch.device('cpu')
        model.to(device)
        
        dataloader = create_dataloader(data, labels, batch_size=8, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        
        loss, accuracy, f1 = evaluate(model, dataloader, criterion, device)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert isinstance(f1, float)
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="tune_transformer_fusion imports not available")
class TestIntegration:
    """Integration tests for transformer fusion tuning."""
    
    def test_full_training_loop(self):
        """Test a complete mini training loop."""
        from performance_extensions.transformer_fusion import MultimodalFusionClassifier
        
        # Create synthetic data
        n_samples = 64
        modality_dims = {'GeneExpr': 50, 'miRNA': 30, 'Prot': 20}
        
        data = {
            mod: np.random.randn(n_samples, dim).astype(np.float32)
            for mod, dim in modality_dims.items()
        }
        labels = np.random.randint(0, 2, size=n_samples)
        
        # Split into train/val
        train_idx = np.arange(48)
        val_idx = np.arange(48, 64)
        
        train_data = {k: v[train_idx] for k, v in data.items()}
        val_data = {k: v[val_idx] for k, v in data.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Create model
        model = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            num_classes=2,
            dropout=0.1
        )
        
        device = torch.device('cpu')
        model.to(device)
        
        # Create dataloaders
        train_loader = create_dataloader(train_data, train_labels, batch_size=8, shuffle=True)
        val_loader = create_dataloader(val_data, val_labels, batch_size=8, shuffle=False)
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Run a few epochs
        for epoch in range(3):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            
            assert not np.isnan(train_loss), f"Train loss is NaN at epoch {epoch}"
            assert not np.isnan(val_loss), f"Val loss is NaN at epoch {epoch}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
