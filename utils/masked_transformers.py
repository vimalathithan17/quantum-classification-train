import logging as _logging
import numpy as _np
from sklearn.base import TransformerMixin, BaseEstimator


class MaskedTransformer(BaseEstimator, TransformerMixin):
    """Wrapper that fits a transformer only on non-all-zero rows and leaves
    all-zero rows unchanged during transform.

    Behavior:
    - fit(X): detect rows with any feature magnitude > eps and fit the inner
      transformer on those rows. If no rows have data, fit on the full X as a
      fallback.
    - transform(X): apply inner.transform to rows with data and set all-zero
      rows to zeros of the transformed dimension.

    This works for scalers, PCA, imputers and similar sklearn-style transformers.
    """

    def __init__(self, transformer, eps=1e-12, fallback='warn'):
        """Initialize MaskedTransformer.

        Args:
            transformer: an sklearn-style transformer instance to wrap.
            eps: threshold to treat values as zero (abs(val) > eps means present).
            fallback: behavior when no rows contain signal. Options:
                - 'warn' (default): fit/transform on full X and emit a warning.
                - 'raise': raise a RuntimeError when this degenerate case occurs.
                - 'all': silently fall back to fitting/transforming on full X.
        """
        self.transformer = transformer
        self.eps = eps
        self.fallback = fallback
        self._logger = _logging.getLogger(__name__)
        if self.fallback not in ('warn', 'raise', 'all'):
            raise ValueError("fallback must be one of 'warn', 'raise' or 'all'")

    def fit(self, X, y=None, **fit_kwargs):
        X_arr = _np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Row has data if any absolute value exceeds eps
        try:
            mask = _np.any(_np.abs(X_arr) > float(self.eps), axis=1)
        except Exception:
            mask = _np.ones(X_arr.shape[0], dtype=bool)

        # If no rows have data, fall back to fitting on full X (configurable)
        if _np.any(mask):
            X_fit = X_arr[mask]
            used_mask = True
        else:
            used_mask = False
            msg = (
                f"MaskedTransformer: no rows with abs(x)>{self.eps} found in fit; "
                "falling back to fitting on full X."
            )
            if self.fallback == 'raise':
                raise RuntimeError(msg)
            if self.fallback == 'warn':
                self._logger.warning(msg + f" Transformer: %s", type(self.transformer).__name__)
            X_fit = X_arr

        # Delegate fit (some transformers accept y)
        if hasattr(self.transformer, 'fit'):
            try:
                if y is None:
                    self.transformer.fit(X_fit, **fit_kwargs)
                else:
                    # If y provided, try to subselect y to match X_fit rows when mask used
                    if _np.any(mask):
                        # y may be array-like or pandas Series
                        try:
                            y_arr = _np.asarray(y)
                            y_fit = y_arr[mask]
                        except Exception:
                            y_fit = y
                        self.transformer.fit(X_fit, y_fit, **fit_kwargs)
                    else:
                        self.transformer.fit(X_fit, y, **fit_kwargs)
            except TypeError:
                # Fallback when transformer.fit doesn't accept y or extra kwargs
                self.transformer.fit(X_fit)

        return self

    def transform(self, X):
        X_arr = _np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        try:
            mask = _np.any(_np.abs(X_arr) > float(self.eps), axis=1)
        except Exception:
            mask = _np.ones(X_arr.shape[0], dtype=bool)

        # If some rows have data, transform only those and leave zeros for others
        if _np.any(mask):
            X_out_sub = self.transformer.transform(X_arr[mask])
            # Ensure 2D for stacking
            X_out_sub = _np.atleast_2d(X_out_sub)
            n_out_cols = X_out_sub.shape[1]
            out = _np.zeros((X_arr.shape[0], n_out_cols), dtype=X_out_sub.dtype)
            out[mask] = X_out_sub
            return out
        else:
            # No rows with data -> fallback behavior
            msg = (
                f"MaskedTransformer: no rows with abs(x)>{self.eps} found in transform; "
                "falling back to transforming full X."
            )
            if self.fallback == 'raise':
                raise RuntimeError(msg)
            if self.fallback == 'warn':
                self._logger.warning(msg + f" Transformer: %s", type(self.transformer).__name__)
            return self.transformer.transform(X_arr)

    def fit_transform(self, X, y=None, **fit_kwargs):
        # Fit first (this will choose subset) then transform
        self.fit(X, y=y, **fit_kwargs)
        return self.transform(X)

    def inverse_transform(self, X):
        if not hasattr(self.transformer, 'inverse_transform'):
            raise AttributeError('Inner transformer has no inverse_transform')
        X_arr = _np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        try:
            mask = _np.any(_np.abs(X_arr) > float(self.eps), axis=1)
        except Exception:
            mask = _np.ones(X_arr.shape[0], dtype=bool)
        if _np.any(mask):
            inv_sub = self.transformer.inverse_transform(X_arr[mask])
            inv_sub = _np.atleast_2d(inv_sub)
            n_cols = inv_sub.shape[1]
            out = _np.zeros((X_arr.shape[0], n_cols), dtype=inv_sub.dtype)
            out[mask] = inv_sub
            return out
        else:
            msg = (
                f"MaskedTransformer: no rows with abs(x)>{self.eps} found in inverse_transform; "
                "falling back to inverse_transform on full X."
            )
            if self.fallback == 'raise':
                raise RuntimeError(msg)
            if self.fallback == 'warn':
                self._logger.warning(msg + f" Transformer: %s", type(self.transformer).__name__)
            return self.transformer.inverse_transform(X_arr)

    def get_params(self, deep=True):
        # Allow sklearn to access inner transformer params
        return {'transformer': self.transformer, 'eps': self.eps, 'fallback': self.fallback}

    def set_params(self, **params):
        if 'transformer' in params:
            self.transformer = params.pop('transformer')
        if 'eps' in params:
            self.eps = params.pop('eps')
        if 'fallback' in params:
            self.fallback = params.pop('fallback')
        return self
