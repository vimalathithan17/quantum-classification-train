import os
import json
import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


def test_assemble_meta_data_and_mask(tmp_path, monkeypatch):
    # Prepare environment so metalearner uses tmp dirs for encoder and output
    out_dir = tmp_path / 'out'
    enc_dir = tmp_path / 'encoder'
    out_dir.mkdir()
    enc_dir.mkdir()
    monkeypatch.setenv('OUTPUT_DIR', str(out_dir))
    monkeypatch.setenv('ENCODER_DIR', str(enc_dir))

    # Create a label encoder and persist it where metalearner expects
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    classes = ['astrocytoma', 'glioblastoma']
    le.fit(classes)
    encoder_path = Path(os.environ['ENCODER_DIR']) / 'label_encoder.joblib'
    joblib.dump(le, encoder_path)

    # Create a preds_dir with one base-learner producing 2 class probability columns
    preds_dir = tmp_path / 'preds1'
    preds_dir.mkdir()
    df_train = pd.DataFrame({
        'case_id': ['caseA', 'caseB', 'caseC'],
        'pred_miRNA_astrocytoma': [0.8, 0.1, 0.5],
        'pred_miRNA_glioblastoma': [0.2, 0.9, 0.5],
    })
    df_test = pd.DataFrame({
        'case_id': ['caseD', 'caseE'],
        'pred_miRNA_astrocytoma': [0.6, 0.3],
        'pred_miRNA_glioblastoma': [0.4, 0.7],
    })
    df_train.to_csv(preds_dir / 'train_oof_preds_miRNA.csv', index=False)
    df_test.to_csv(preds_dir / 'test_preds_miRNA.csv', index=False)

    # Create indicator parquet file with class labels and indicator column
    indicators = pd.DataFrame({
        'case_id': ['caseA', 'caseB', 'caseC', 'caseD', 'caseE'],
        'class': ['astrocytoma', 'glioblastoma', 'astrocytoma', 'glioblastoma', 'astrocytoma'],
        'is_missing_miRNA_': [0.0, 0.0, 0.0, 0.0, 0.0]
    })
    ind_file = tmp_path / 'indicators.parquet'
    # to_parquet requires a parquet engine; assume available in CI
    indicators.to_parquet(ind_file, index=False)

    # Import metalearner by file path so tests run correctly regardless of
    # package import paths used by pytest/CI.
    # Ensure repository root is on sys.path so relative imports inside
    # metalearner.py (e.g., logging_utils) are resolvable.
    import sys
    repo_root = str(Path.cwd())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec = importlib.util.spec_from_file_location('metalearner', str(Path.cwd() / 'metalearner.py'))
    metalearner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metalearner)

    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le_loaded, indicator_cols = metalearner.assemble_meta_data([str(preds_dir)], str(ind_file))

    # Basic sanity
    assert X_meta_train is not None
    assert isinstance(X_meta_train, pd.DataFrame)

    # Base prediction columns should be present and reduced (one dropped)
    pred_cols = [c for c in X_meta_train.columns if str(c).startswith('pred_')]
    # Originally 2 preds for miRNA, after dropping one expect 1
    assert len(pred_cols) == 1

    # Build mask using the helper and ensure shapes align
    base_cols = pred_cols
    mask = metalearner._build_mask_from_indicators(X_meta_train, base_cols, indicator_cols)
    assert mask.shape[0] == X_meta_train.shape[0]
    assert mask.shape[1] == len(base_cols)

    # Check that dropped mapping file exists in OUTPUT_DIR
    mapping_path = Path(metalearner.OUTPUT_DIR) / 'dropped_pred_columns.json'
    assert mapping_path.exists(), f"Dropped mapping file not found at {mapping_path}"
    with open(mapping_path, 'r') as fh:
        mapping = json.load(fh)
    # mapping should include miRNA key and a dropped column name
    assert 'miRNA' in mapping or any('miRNA' in k for k in mapping.keys())
