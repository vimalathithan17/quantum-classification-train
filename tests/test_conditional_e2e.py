import os
import joblib
import json
import numpy as np
import pandas as pd

# Defer imports that rely on package path until inside the test function


class DummyQMLModel:
    def predict_proba(self, inp):
        # inp is a tuple (X_scaled, is_missing_mask)
        # return a deterministic 2-class probability
        return np.array([[0.25, 0.75]])


class DummyMetaLearner:
    def predict(self, X):
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

    # Run inference
    pred = inference.make_single_prediction(str(model_dir), str(patient_dir))

    # Our DummyMetaLearner predicts class index 1 -> LabelEncoder inverse gives 'B'
    assert pred == 'B'
