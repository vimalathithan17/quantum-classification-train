import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Ensure project root is on sys.path so tests can import utils
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.masked_transformers import MaskedTransformer


def test_standard_scaler_ignores_all_zero_rows():
    # Rows 0 and 3 are all-zero and should be ignored when fitting
    X = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [0.0, 0.0],
        [5.0, 6.0]
    ])

    mt = MaskedTransformer(StandardScaler())
    mt.fit(X)

    # Inner scaler mean should equal mean of rows with data (rows 1,2,4)
    mask = np.any(np.abs(X) > 1e-12, axis=1)
    expected_mean = X[mask].mean(axis=0)
    np.testing.assert_allclose(mt.transformer.mean_, expected_mean)

    Xt = mt.transform(X)
    # Outputs should preserve row order and zero-rows should remain zeros after transform
    assert Xt.shape == (5, 2)
    np.testing.assert_allclose(Xt[0], np.zeros(2))
    np.testing.assert_allclose(Xt[3], np.zeros(2))

    # Non-zero rows should match StandardScaler fitted on the subset
    scaler_subset = StandardScaler().fit(X[mask])
    np.testing.assert_allclose(Xt[mask], scaler_subset.transform(X[mask]))


def test_pca_preserves_order_and_zero_rows():
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0],
        [0.5, -1.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    mt = MaskedTransformer(PCA(n_components=1))
    Xt = mt.fit_transform(X)

    # Should have same number of rows, single column
    assert Xt.shape == (4, 1)
    # Zero rows should remain zeros
    np.testing.assert_allclose(Xt[0], np.zeros(1))
    np.testing.assert_allclose(Xt[3], np.zeros(1))
    # Non-zero rows should not be zero
    assert not np.allclose(Xt[1], 0)
    assert not np.allclose(Xt[2], 0)


def test_masked_transformer_with_imputer_and_eps():
    # Test that eps threshold works: very small values treated as zeros
    X = np.array([
        [0.0, 0.0],
        [1e-13, 0.0],  # below default eps=1e-12 -> treated as zero
        [1.0, np.nan],
        [0.0, 0.0]
    ])

    imputer = SimpleImputer(strategy='median')
    mt = MaskedTransformer(imputer, eps=1e-12)
    Xt = mt.fit_transform(X)

    # Row 2 had a NaN and should be imputed; rows 0,1,3 treated as zeros and kept as zeros
    assert Xt.shape[0] == 4
    ncols = Xt.shape[1]
    # Check imputation for the non-empty row (index 2)
    assert not np.any(np.isnan(Xt[2]))
    # Zero rows remain zeros (with the transformed column count)
    np.testing.assert_allclose(Xt[0], np.zeros(ncols))
    np.testing.assert_allclose(Xt[1], np.zeros(ncols))
