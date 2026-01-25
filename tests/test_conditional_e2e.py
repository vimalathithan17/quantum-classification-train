"""
End-to-end test for conditional model inference flow.

Tests the complete inference pipeline including:
1. Base learner prediction with tuple input (X_scaled, is_missing_mask)
2. Meta-learner prediction with tuple input (X_base, X_mask)
3. Mask building from indicator columns
"""
import os
import joblib
import json
import numpy as np
import pandas as pd
import pytest

# Defer imports that rely on package path until inside the test function


class DummyQMLModel:
    """Mock QML model that validates tuple input."""
    
    def predict_proba(self, inp):
        # inp MUST be a tuple (X_scaled, is_missing_mask) for conditional models
        if not isinstance(inp, tuple):
            raise TypeError(f"Conditional QML model expects tuple input, got {type(inp)}")
        if len(inp) != 2:
            raise ValueError(f"Expected 2-element tuple (X_scaled, mask), got {len(inp)} elements")
        
        X_scaled, is_missing_mask = inp
        assert X_scaled.shape == is_missing_mask.shape, \
            f"X_scaled shape {X_scaled.shape} != mask shape {is_missing_mask.shape}"
        
        # Return deterministic 2-class probability
        return np.array([[0.25, 0.75]])


class DummyMetaLearner:
    """Mock meta-learner that validates tuple input for gated models."""
    
    def __init__(self):
        self.received_tuple = False
        self.input_shapes = None
    
    def predict(self, X):
        # X MUST be a tuple (base_preds, mask) for gated meta-learners
        if not isinstance(X, tuple):
            raise TypeError(f"Gated meta-learner expects tuple input, got {type(X)}")
        if len(X) != 2:
            raise ValueError(f"Expected 2-element tuple (base_preds, mask), got {len(X)} elements")
        
        X_base, X_mask = X
        self.received_tuple = True
        self.input_shapes = (X_base.shape, X_mask.shape)
        
        # Validate shapes match
        assert X_base.shape == X_mask.shape, \
            f"Base shape {X_base.shape} != mask shape {X_mask.shape}"
        
        # Always predict class index 1
        return np.array([1])


def test_conditional_model_inference_flow(tmp_path):
    # Import the inference module by path to avoid import-time path issues when
    # running a single test file in isolation.
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = os.getcwd()
    inference_path = os.path.join(repo_root, 'inference.py')
    spec = importlib.util.spec_from_file_location('inference', inference_path)
    inference = importlib.util.module_from_spec(spec)
    sys.modules['inference'] = inference
    # Ensure repo root is on sys.path so imports inside inference.py work
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec.loader.exec_module(inference)

    from logging_utils import log
    # Prepare temporary model deployment directory
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    # Label encoder with two classes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(["A", "B"])
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))

    # Create meta-learner and save
    meta = DummyMetaLearner()
    joblib.dump(meta, os.path.join(model_dir, 'meta_learner_final.joblib'))

    # Build meta columns order (pred cols for each data type + missing flags)
    DATA_TYPES = ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']
    meta_cols = []
    for dt in DATA_TYPES:
        for cls in le.classes_:
            meta_cols.append(f"pred_{dt}_{cls}")
    for dt in DATA_TYPES:
        meta_cols.append(f"is_missing_{dt}")

    with open(os.path.join(model_dir, 'meta_learner_columns.json'), 'w') as fh:
        json.dump(meta_cols, fh)

    # Create artifacts for one conditional base-learner (CNV)
    data_type = 'CNV'
    selected_features = ['f1', 'f2']
    joblib.dump(selected_features, os.path.join(model_dir, f'selected_features_{data_type}.joblib'))

    # Save a sentinel None scaler
    joblib.dump(None, os.path.join(model_dir, f'scaler_{data_type}.joblib'))

    # Save a dummy QML model that returns a predictable probability
    qml = DummyQMLModel()
    joblib.dump(qml, os.path.join(model_dir, f'qml_model_{data_type}.joblib'))

    # Create a new patient parquet for CNV with the required selected features
    patient_dir = tmp_path / 'patient'
    patient_dir.mkdir()
    df = pd.DataFrame([{ 'case_id': 'P1', 'class': 'A', 'f1': 0.1, 'f2': 0.2 }])
    # Write CSV instead of parquet to avoid external parquet engine dependency in tests.
    csv_path = os.path.join(patient_dir, 'data_CNV_.csv')
    df.to_csv(csv_path, index=False)

    # Monkeypatch inference.safe_load_parquet to read CSV for our test files.
    def fake_safe_load_parquet(file_path):
        alt = file_path.replace('.parquet', '.csv')
        if os.path.exists(alt):
            return pd.read_csv(alt)
        return None

    inference.safe_load_parquet = fake_safe_load_parquet

    # Run inference - this will raise TypeError if meta-learner doesn't receive tuple
    pred = inference.make_single_prediction(str(model_dir), str(patient_dir))

    # Our DummyMetaLearner predicts class index 1 -> LabelEncoder inverse gives 'B'
    assert pred == 'B'


def test_mask_dimensions_match_base_predictions(tmp_path):
    """Verify that the mask built by inference.py has correct dimensions."""
    import importlib.util
    import sys
    from sklearn.preprocessing import LabelEncoder
    
    repo_root = os.getcwd()
    inference_path = os.path.join(repo_root, 'inference.py')
    spec = importlib.util.spec_from_file_location('inference', inference_path)
    inference = importlib.util.module_from_spec(spec)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec.loader.exec_module(inference)
    
    # Create a meta-learner that captures and validates input shapes
    class ShapeCapturingMetaLearner:
        def __init__(self):
            self.captured_shapes = None
            
        def predict(self, X):
            if not isinstance(X, tuple) or len(X) != 2:
                raise TypeError(f"Expected tuple (X_base, X_mask), got {type(X)}")
            X_base, X_mask = X
            self.captured_shapes = {
                'X_base': X_base.shape,
                'X_mask': X_mask.shape
            }
            # Validate shapes match
            assert X_base.shape == X_mask.shape, \
                f"Shape mismatch: X_base={X_base.shape}, X_mask={X_mask.shape}"
            return np.array([0])
    
    # Setup
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    
    le = LabelEncoder()
    le.fit(["ClassA", "ClassB", "ClassC"])
    joblib.dump(le, model_dir / 'label_encoder.joblib')
    
    meta = ShapeCapturingMetaLearner()
    joblib.dump(meta, model_dir / 'meta_learner_final.joblib')
    
    # Build meta columns with 3 classes
    DATA_TYPES = ['CNV', 'GeneExpr', 'miRNA', 'Meth', 'Prot', 'SNV']
    meta_cols = []
    for dt in DATA_TYPES:
        for cls in le.classes_:
            meta_cols.append(f"pred_{dt}_{cls}")
    for dt in DATA_TYPES:
        meta_cols.append(f"is_missing_{dt}")
    
    with open(model_dir / 'meta_learner_columns.json', 'w') as fh:
        json.dump(meta_cols, fh)
    
    # Create dummy base learners for all data types
    class Dummy3ClassQML:
        def predict_proba(self, inp):
            return np.array([[0.33, 0.33, 0.34]])
    
    for dt in DATA_TYPES:
        joblib.dump(['f1', 'f2'], model_dir / f'selected_features_{dt}.joblib')
        joblib.dump(None, model_dir / f'scaler_{dt}.joblib')
        joblib.dump(Dummy3ClassQML(), model_dir / f'qml_model_{dt}.joblib')
    
    # Create patient data
    patient_dir = tmp_path / 'patient'
    patient_dir.mkdir()
    
    for dt in DATA_TYPES:
        df = pd.DataFrame([{'case_id': 'P1', 'class': 'ClassA', 'f1': 0.1, 'f2': 0.2}])
        df.to_csv(patient_dir / f'data_{dt}_.csv', index=False)
    
    def fake_safe_load_parquet(file_path):
        alt = file_path.replace('.parquet', '.csv')
        if os.path.exists(alt):
            return pd.read_csv(alt)
        return None
    
    inference.safe_load_parquet = fake_safe_load_parquet
    
    pred = inference.make_single_prediction(str(model_dir), str(patient_dir))
    
    # With 6 data types and 3 classes, we expect:
    # Base columns = 6 * 3 = 18 (but one class per modality is dropped for degeneracy)
    # So base columns should be 6 * 2 = 12 after dropping
    # Or if no degeneracy removal: 18 columns
    # The key is X_base.shape == X_mask.shape
    assert pred == 'ClassA'
