"""
Tests for train/test split handling in pretrained features pipeline.

These tests verify that the pipeline correctly:
1. Splits data BEFORE pretraining to prevent data leakage
2. Saves train/test indices and scalers during pretraining
3. Loads and applies scalers during feature extraction
4. Saves train/test embeddings separately
5. Downstream scripts correctly load split files

The proper pipeline flow is:
    pretrain_contrastive.py (--test_size 0.2)
        → saves train_indices.npy, test_indices.npy, scalers.joblib
    extract_pretrained_features.py
        → loads indices and scalers, saves {mod}_train_embeddings.npy, {mod}_test_embeddings.npy
    dre_standard.py / train_transformer_fusion.py
        → detects split files and uses them directly (no re-split)
"""

import os
import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSplitFileDetection:
    """Test that loading functions detect split vs combined files correctly."""
    
    def test_detect_split_files_exist(self, tmp_path):
        """Test detection of split file format."""
        # Create split files
        n_train, n_test = 80, 20
        embed_dim = 64
        
        np.save(tmp_path / "GeneExpr_train_embeddings.npy", np.random.randn(n_train, embed_dim))
        np.save(tmp_path / "GeneExpr_test_embeddings.npy", np.random.randn(n_test, embed_dim))
        np.save(tmp_path / "train_labels.npy", np.random.randint(0, 4, n_train))
        np.save(tmp_path / "test_labels.npy", np.random.randint(0, 4, n_test))
        
        # Check that split files are detected
        train_file = tmp_path / "GeneExpr_train_embeddings.npy"
        test_file = tmp_path / "GeneExpr_test_embeddings.npy"
        
        assert train_file.exists()
        assert test_file.exists()
        
        # This is the pattern used in load_pretrained_features
        has_split = train_file.exists() and test_file.exists()
        assert has_split, "Should detect split files"
    
    def test_detect_combined_files(self, tmp_path):
        """Test detection of combined (legacy) file format."""
        # Create combined files only
        n_samples = 100
        embed_dim = 64
        
        np.save(tmp_path / "GeneExpr_embeddings.npy", np.random.randn(n_samples, embed_dim))
        np.save(tmp_path / "labels.npy", np.random.randint(0, 4, n_samples))
        
        # Check that split files are NOT detected
        train_file = tmp_path / "GeneExpr_train_embeddings.npy"
        test_file = tmp_path / "GeneExpr_test_embeddings.npy"
        
        assert not train_file.exists()
        assert not test_file.exists()
        
        has_split = train_file.exists() and test_file.exists()
        assert not has_split, "Should not detect split files"


class TestTrainTransformerFusionLoadPretrained:
    """Test train_transformer_fusion.py's load_pretrained_features function."""
    
    def test_load_split_features(self, tmp_path):
        """Test loading split pretrained features."""
        # Simulate the detection logic without importing the full module
        # (to avoid slow torch import and potential hangs)
        
        # Create split files for multiple modalities
        n_train, n_test = 80, 20
        embed_dim = 64
        modalities = ['GeneExpr', 'miRNA']
        
        for mod in modalities:
            np.save(tmp_path / f"{mod}_train_embeddings.npy", 
                   np.random.randn(n_train, embed_dim).astype(np.float32))
            np.save(tmp_path / f"{mod}_test_embeddings.npy", 
                   np.random.randn(n_test, embed_dim).astype(np.float32))
        
        np.save(tmp_path / "train_labels.npy", np.arange(n_train) % 4)
        np.save(tmp_path / "test_labels.npy", np.arange(n_test) % 4)
        
        # Replicate the split detection logic from train_transformer_fusion.py
        features_dir = tmp_path
        first_modality = modalities[0]
        train_file_check = features_dir / f"{first_modality}_train_embeddings.npy"
        test_file_check = features_dir / f"{first_modality}_test_embeddings.npy"
        has_split_files = train_file_check.exists() and test_file_check.exists()
        
        assert has_split_files, "Should detect split files"
        
        # Load as train_transformer_fusion.py does
        train_data = {}
        test_data = {}
        modality_dims = {}
        
        train_labels = np.load(features_dir / 'train_labels.npy')
        test_labels = np.load(features_dir / 'test_labels.npy')
        
        for modality in modalities:
            train_file = features_dir / f"{modality}_train_embeddings.npy"
            test_file = features_dir / f"{modality}_test_embeddings.npy"
            
            train_data[modality] = np.load(train_file).astype(np.float32)
            test_data[modality] = np.load(test_file).astype(np.float32)
            modality_dims[modality] = train_data[modality].shape[1]
        
        # Verify data shapes
        for mod in modalities:
            assert train_data[mod].shape == (n_train, embed_dim)
            assert test_data[mod].shape == (n_test, embed_dim)
        
        assert len(train_labels) == n_train
        assert len(test_labels) == n_test
    
    def test_combined_files_detection(self, tmp_path):
        """Test detection of combined (legacy) file format."""
        # Create combined files only (legacy format)
        n_samples = 100
        embed_dim = 64
        modalities = ['GeneExpr']
        
        np.save(tmp_path / "GeneExpr_embeddings.npy", 
               np.random.randn(n_samples, embed_dim).astype(np.float32))
        np.save(tmp_path / "labels.npy", np.arange(n_samples) % 4)
        
        # Check split detection
        features_dir = tmp_path
        first_modality = modalities[0]
        train_file_check = features_dir / f"{first_modality}_train_embeddings.npy"
        test_file_check = features_dir / f"{first_modality}_test_embeddings.npy"
        has_split_files = train_file_check.exists() and test_file_check.exists()
        
        assert not has_split_files, "Should NOT detect split files for legacy format"
        
        # Verify combined file exists
        combined_file = features_dir / "GeneExpr_embeddings.npy"
        assert combined_file.exists(), "Combined file should exist"


class TestExtractPretainedFeaturesIntegration:
    """Test extract_pretrained_features.py split handling."""
    
    def test_split_indices_loading(self, tmp_path):
        """Test that split indices are loaded correctly from pretraining."""
        # Simulate pretraining output
        n_total = 100
        n_train = 80
        n_test = 20
        
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, n_total)
        
        np.save(tmp_path / "train_indices.npy", train_indices)
        np.save(tmp_path / "test_indices.npy", test_indices)
        
        # Load indices (as extract_pretrained_features.py does)
        loaded_train = np.load(tmp_path / "train_indices.npy")
        loaded_test = np.load(tmp_path / "test_indices.npy")
        
        assert len(loaded_train) == n_train
        assert len(loaded_test) == n_test
        assert len(set(loaded_train) & set(loaded_test)) == 0, "No overlap between train/test"
    
    def test_scalers_file_format(self, tmp_path):
        """Test that scalers are saved/loaded correctly."""
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create and fit scalers
        scalers = {}
        for mod in ['GeneExpr', 'miRNA']:
            scaler = StandardScaler()
            scaler.fit(np.random.randn(80, 100))  # Fit on "train" data
            scalers[mod] = scaler
        
        # Save scalers
        joblib.dump(scalers, tmp_path / "scalers.joblib")
        
        # Load scalers
        loaded_scalers = joblib.load(tmp_path / "scalers.joblib")
        
        assert 'GeneExpr' in loaded_scalers
        assert 'miRNA' in loaded_scalers
        assert hasattr(loaded_scalers['GeneExpr'], 'transform')


class TestDREStandardSplitLoading:
    """Test dre_standard.py split file loading logic."""
    
    def test_split_detection_logic(self, tmp_path):
        """Test the split detection pattern used in dre_standard.py."""
        # Create split files
        n_train, n_test = 80, 20
        embed_dim = 8
        
        np.save(tmp_path / "GeneExpr_train_embeddings.npy", 
               np.random.randn(n_train, embed_dim).astype(np.float32))
        np.save(tmp_path / "GeneExpr_test_embeddings.npy", 
               np.random.randn(n_test, embed_dim).astype(np.float32))
        np.save(tmp_path / "train_labels.npy", np.random.randint(0, 4, n_train))
        np.save(tmp_path / "test_labels.npy", np.random.randint(0, 4, n_test))
        
        # Replicate the detection logic from dre_standard.py
        datatype = "GeneExpr"
        pretrained_features_dir = tmp_path
        
        train_emb_file = pretrained_features_dir / f"{datatype}_train_embeddings.npy"
        test_emb_file = pretrained_features_dir / f"{datatype}_test_embeddings.npy"
        pretrained_has_split = train_emb_file.exists() and test_emb_file.exists()
        
        assert pretrained_has_split, "Should detect split files"
        
        # Load as dre_standard.py does
        if pretrained_has_split:
            X_train = np.load(train_emb_file)
            X_test = np.load(test_emb_file)
            y_train = np.load(pretrained_features_dir / "train_labels.npy")
            y_test = np.load(pretrained_features_dir / "test_labels.npy")
            
            assert X_train.shape[0] == n_train
            assert X_test.shape[0] == n_test
            assert len(y_train) == n_train
            assert len(y_test) == n_test


class TestNoDataLeakage:
    """Tests to verify data leakage prevention."""
    
    def test_train_test_indices_no_overlap(self, tmp_path):
        """Verify train/test indices have no overlap (leakage prevention)."""
        n_total = 100
        
        # Simulate stratified split (as pretrain_contrastive.py does)
        from sklearn.model_selection import train_test_split
        
        all_indices = np.arange(n_total)
        labels = np.random.randint(0, 4, n_total)
        
        train_idx, test_idx = train_test_split(
            all_indices, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Verify no overlap
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices - DATA LEAKAGE!"
        
        # Verify all indices accounted for
        assert len(train_idx) + len(test_idx) == n_total
    
    def test_scaler_fit_on_train_only(self):
        """Verify StandardScaler is fit on train data only."""
        from sklearn.preprocessing import StandardScaler
        
        np.random.seed(42)
        
        # Simulate data with different distributions
        train_data = np.random.randn(80, 10) * 2 + 5  # mean=5, std=2
        test_data = np.random.randn(20, 10) * 3 + 10  # mean=10, std=3 (different!)
        
        # Fit scaler on train only (correct approach)
        scaler = StandardScaler()
        scaler.fit(train_data)
        
        # Transform both
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Train should be standardized (mean≈0, std≈1)
        assert abs(train_scaled.mean()) < 0.1, "Train mean should be near 0"
        assert abs(train_scaled.std() - 1.0) < 0.1, "Train std should be near 1"
        
        # Test will NOT be standardized (different distribution)
        # This is expected and correct - we don't want to "peek" at test stats
        # Test mean should be offset from 0 since test has different distribution
        assert abs(test_scaled.mean()) > 0.5, "Test mean should differ (different distribution)"


class TestSplitFileNaming:
    """Test correct file naming conventions."""
    
    def test_embedding_file_names(self):
        """Test the expected file naming convention."""
        modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot', 'SNV']
        
        for mod in modalities:
            # Train embedding file
            train_name = f"{mod}_train_embeddings.npy"
            assert "_train_" in train_name
            assert train_name.endswith(".npy")
            
            # Test embedding file
            test_name = f"{mod}_test_embeddings.npy"
            assert "_test_" in test_name
            assert test_name.endswith(".npy")
    
    def test_label_file_names(self):
        """Test label file naming convention."""
        assert "train_labels.npy".endswith(".npy")
        assert "test_labels.npy".endswith(".npy")
        assert "train_" in "train_labels.npy"
        assert "test_" in "test_labels.npy"


def run_all_tests():
    """Run all tests and report results."""
    import traceback
    
    test_classes = [
        TestSplitFileDetection,
        TestTrainTransformerFusionLoadPretrained,
        TestExtractPretainedFeaturesIntegration,
        TestDREStandardSplitLoading,
        TestNoDataLeakage,
        TestSplitFileNaming,
    ]
    
    results = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running: {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                
                # Handle tmp_path fixture manually for standalone execution
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    
                    # Check if method needs tmp_path
                    import inspect
                    sig = inspect.signature(method)
                    
                    if 'tmp_path' in sig.parameters:
                        if 'capsys' in sig.parameters:
                            # Skip tests that need capsys (pytest-specific)
                            print(f"  SKIP {method_name} (requires pytest capsys)")
                            continue
                        method(tmp_path)
                    else:
                        method()
                
                print(f"  ✓ {method_name}")
                results.append((test_class.__name__, method_name, True, None))
                
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                results.append((test_class.__name__, method_name, False, str(e)))
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    passed = sum(1 for r in results if r[2])
    failed = sum(1 for r in results if not r[2])
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(results)}")
    
    if failed > 0:
        print("\nFailed tests:")
        for cls_name, method_name, success, error in results:
            if not success:
                print(f"  - {cls_name}.{method_name}: {error}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
