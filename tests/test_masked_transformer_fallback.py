import logging

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from utils.masked_transformers import MaskedTransformer


def test_masked_transformer_fallback_warn_on_fit_and_transform(caplog):
    """When all rows are zeros, MaskedTransformer should warn (fallback='warn')."""
    X = np.zeros((5, 3))
    mt = MaskedTransformer(StandardScaler(), eps=1e-12, fallback='warn')

    with caplog.at_level(logging.WARNING):
        # fit should trigger a warning and fall back to fitting on full X
        mt.fit(X)
        assert any('MaskedTransformer: no rows' in rec.getMessage() for rec in caplog.records)

        # clear and test transform also triggers the warning path
        caplog.clear()
        _ = mt.transform(X)
        assert any('MaskedTransformer: no rows' in rec.getMessage() for rec in caplog.records)


def test_masked_transformer_fallback_raise_on_fit():
    """When fallback='raise', fitting on an all-zero array should raise."""
    X = np.zeros((4, 2))
    mt = MaskedTransformer(StandardScaler(), eps=1e-12, fallback='raise')
    with pytest.raises(RuntimeError):
        mt.fit(X)
