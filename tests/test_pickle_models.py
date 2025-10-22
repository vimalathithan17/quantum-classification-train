import os
import sys
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qml_models import (
    MulticlassQuantumClassifierDR,
    MulticlassQuantumClassifierDataReuploadingDR,
    ConditionalMulticlassQuantumClassifierFS,
    ConditionalMulticlassQuantumClassifierDataReuploadingFS,
)


def _basic_dataset(n_qubits):
    X = np.random.rand(3, n_qubits)
    y = np.array([0, 1, 0])
    return X, y


def test_pickle_unpickle_all_models():
    """Pickle and unpickle each model class and ensure basic methods work."""
    classes = [
        MulticlassQuantumClassifierDR,
        MulticlassQuantumClassifierDataReuploadingDR,
        ConditionalMulticlassQuantumClassifierFS,
        ConditionalMulticlassQuantumClassifierDataReuploadingFS,
    ]

    for cls in classes:
        n_qubits = 3
        model = cls(n_qubits=n_qubits, n_classes=2, steps=1, verbose=False, validation_frac=0.0)

        # Fit minimally where applicable (some classes expect tuple input)
        X, y = _basic_dataset(n_qubits)

        if 'Conditional' in cls.__name__:
            # conditional models expect (X, mask) input
            mask = np.zeros_like(X)
            # ensure mask shape is (N, n_qubits)
            mask = (mask > 0).astype(int)
            model.fit((X, mask), y)
        else:
            model.fit(X, y)

        # Ensure predict_proba works before pickling
        if 'Conditional' in cls.__name__:
            probs = model.predict_proba((X, mask))
        else:
            probs = model.predict_proba(X)

        assert probs.shape[0] == X.shape[0]

        # Pickle and unpickle, with diagnostics on failure
        try:
            data = pickle.dumps(model)
            model2 = pickle.loads(data)
        except Exception as e:
            # Diagnose which attribute cannot be pickled
            bad = []
            for k, v in sorted(model.__dict__.items()):
                try:
                    pickle.dumps(v)
                except Exception:
                    bad.append(k)
            raise AssertionError(f"Pickling model {cls.__name__} failed: {e}. Non-picklable attrs: {bad}")

        # After unpickle, predict_proba should still work (may recreate qnode)
        if 'Conditional' in cls.__name__:
            probs2 = model2.predict_proba((X, mask))
        else:
            probs2 = model2.predict_proba(X)

        assert probs2.shape[0] == X.shape[0]
        preds = model2.predict(X if 'Conditional' not in cls.__name__ else (X, mask))
        # preds length should match samples
        assert preds.shape[0] == X.shape[0]
