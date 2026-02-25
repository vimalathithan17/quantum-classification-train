#!/usr/bin/env python3
"""
Entangler Comparison Experiment

Compares different quantum entangler architectures for the multiclass classification task:
- BasicEntanglerLayers (CNOT chain)
- StronglyEntanglingLayers (more expressive 2-qubit gates)
- RandomLayers (random gate placement)
- SimplifiedTwoDesign (hardware-efficient)

This script validates claims in ARCHITECTURE_DEEP_DIVE.md about entangler performance.

Usage:
    python examples/compare_entanglers.py --data_dir final_processed_datasets --datatype GeneExpr
    python examples/compare_entanglers.py --data_dir final_processed_datasets --datatype GeneExpr --n_trials 3
    python examples/compare_entanglers.py --synthetic  # Use synthetic data for quick testing

Output:
    - Console: Summary table with accuracy, F1, training time
    - File: examples/entangler_comparison_results.csv
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_utils import log


# ============================================================================
# ENTANGLER DEFINITIONS
# ============================================================================

def basic_entangler_circuit(inputs, weights, n_qubits, n_layers):
    """
    BasicEntanglerLayers: Simple CNOT chain entanglement.
    
    Architecture per layer:
        RX(θ₀) RX(θ₁) RX(θ₂) ... RX(θₙ₋₁)
        CNOT(0→1) CNOT(1→2) ... CNOT(n-2→n-1)
    
    Weight shape: (n_layers, n_qubits)
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def strongly_entangling_circuit(inputs, weights, n_qubits, n_layers):
    """
    StronglyEntanglingLayers: High expressivity with Rot gates and CNOTs.
    
    Architecture per layer:
        Rot(θ₀,θ₁,θ₂) on each qubit  (3 params per qubit)
        CNOT entanglement with varying patterns (all-to-all style)
    
    Weight shape: (n_layers, n_qubits, 3) - 3x more parameters than Basic
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def random_layers_circuit(inputs, weights, n_qubits, n_layers, seed=42):
    """
    RandomLayers: Random gate placement (fixed per seed).
    
    Architecture:
        Random single-qubit rotations (Hadamard, PauliX, PauliY, PauliZ, RX, RY, RZ)
        Random CNOT placement
        
    Weight shape: (n_layers, n_qubits)
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.RandomLayers(weights, wires=range(n_qubits), seed=seed)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def simplified_two_design_circuit(inputs, weights, n_qubits, n_layers):
    """
    SimplifiedTwoDesign: Hardware-efficient variational ansatz.
    
    Architecture per layer:
        RY(θ) on each qubit
        CZ entanglement (alternating pairs)
        
    Weight shape: (n_layers, n_qubits)
    More suitable for near-term quantum hardware.
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.SimplifiedTwoDesign(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ============================================================================
# CLASSIFIER CLASS
# ============================================================================

class EntanglerComparisonClassifier:
    """
    Quantum classifier with configurable entangler for comparison experiments.
    """
    
    ENTANGLERS = {
        'basic': {
            'circuit_fn': basic_entangler_circuit,
            'weight_shape_fn': lambda n_layers, n_qubits: (n_layers, n_qubits),
            'description': 'BasicEntanglerLayers (CNOT chain)',
        },
        'strongly': {
            'circuit_fn': strongly_entangling_circuit,
            'weight_shape_fn': lambda n_layers, n_qubits: (n_layers, n_qubits, 3),
            'description': 'StronglyEntanglingLayers (Rot + variable CNOT)',
        },
        'random': {
            'circuit_fn': random_layers_circuit,
            'weight_shape_fn': lambda n_layers, n_qubits: (n_layers, n_qubits),
            'description': 'RandomLayers (random gate placement)',
        },
        'simplified': {
            'circuit_fn': simplified_two_design_circuit,
            'weight_shape_fn': lambda n_layers, n_qubits: (n_layers, n_qubits),
            'description': 'SimplifiedTwoDesign (hardware-efficient)',
        },
    }
    
    def __init__(
        self,
        entangler_type: str = 'basic',
        n_qubits: int = 8,
        n_layers: int = 3,
        n_classes: int = 2,
        lr: float = 0.01,
        steps: int = 100,
        seed: int = 42,
    ):
        if entangler_type not in self.ENTANGLERS:
            raise ValueError(f"Unknown entangler: {entangler_type}. Choose from {list(self.ENTANGLERS.keys())}")
        
        self.entangler_type = entangler_type
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.lr = lr
        self.steps = steps
        self.seed = seed
        
        self.entangler_info = self.ENTANGLERS[entangler_type]
        self.weight_shape = self.entangler_info['weight_shape_fn'](n_layers, n_qubits)
        
        # Initialize device and circuit
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self._qcircuit = self._build_circuit()
        
        # Initialize weights
        np.random.seed(seed)
        self.weights = pnp.array(np.random.randn(*self.weight_shape) * 0.1, requires_grad=True)
        
        # Classical readout layer
        self.weights_classical = pnp.array(
            np.random.randn(n_qubits, n_classes) * 0.1, requires_grad=True
        )
        self.bias_classical = pnp.array(np.zeros(n_classes), requires_grad=True)
        
        # Label encoder
        self.label_encoder = None
        
    def _build_circuit(self):
        """Build the quantum circuit with selected entangler."""
        circuit_fn = self.entangler_info['circuit_fn']
        
        if self.entangler_type == 'random':
            @qml.qnode(self.dev, interface='autograd')
            def qcircuit(inputs, weights):
                return circuit_fn(inputs, weights, self.n_qubits, self.n_layers, self.seed)
        else:
            @qml.qnode(self.dev, interface='autograd')
            def qcircuit(inputs, weights):
                return circuit_fn(inputs, weights, self.n_qubits, self.n_layers)
        
        return qcircuit
    
    def _forward(self, X, weights, weights_classical, bias_classical):
        """Forward pass through quantum circuit + classical readout."""
        # Quantum circuit
        quantum_outputs = []
        for x in X:
            out = self._qcircuit(x, weights)
            quantum_outputs.append(out)
        quantum_outputs = pnp.array(quantum_outputs)
        
        # Classical readout
        logits = pnp.dot(quantum_outputs, weights_classical) + bias_classical
        return logits
    
    def _loss(self, weights, weights_classical, bias_classical, X, y):
        """Cross-entropy loss."""
        logits = self._forward(X, weights, weights_classical, bias_classical)
        # Softmax + cross-entropy
        logits_shifted = logits - pnp.max(logits, axis=1, keepdims=True)
        exp_logits = pnp.exp(logits_shifted)
        probs = exp_logits / pnp.sum(exp_logits, axis=1, keepdims=True)
        probs = pnp.clip(probs, 1e-10, 1 - 1e-10)
        
        # One-hot encode y
        y_onehot = pnp.zeros((len(y), self.n_classes))
        for i, yi in enumerate(y):
            y_onehot[i, int(yi)] = 1
        
        loss = -pnp.mean(pnp.sum(y_onehot * pnp.log(probs), axis=1))
        return loss
    
    def fit(self, X, y, verbose=True):
        """Train the classifier."""
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X = pnp.array(X, requires_grad=False)
        y_encoded = pnp.array(y_encoded, requires_grad=False)
        
        opt = qml.AdamOptimizer(stepsize=self.lr)
        
        for step in range(self.steps):
            (self.weights, self.weights_classical, self.bias_classical), loss = opt.step_and_cost(
                self._loss,
                self.weights,
                self.weights_classical,
                self.bias_classical,
                X,
                y_encoded
            )
            
            if verbose and (step + 1) % 20 == 0:
                log.info(f"  Step {step + 1}/{self.steps}, Loss: {loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        X = pnp.array(X, requires_grad=False)
        logits = self._forward(X, self.weights, self.weights_classical, self.bias_classical)
        
        # Softmax
        logits_shifted = logits - pnp.max(logits, axis=1, keepdims=True)
        exp_logits = pnp.exp(logits_shifted)
        probs = exp_logits / pnp.sum(exp_logits, axis=1, keepdims=True)
        
        preds = pnp.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(preds)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = pnp.array(X, requires_grad=False)
        logits = self._forward(X, self.weights, self.weights_classical, self.bias_classical)
        
        logits_shifted = logits - pnp.max(logits, axis=1, keepdims=True)
        exp_logits = pnp.exp(logits_shifted)
        probs = exp_logits / pnp.sum(exp_logits, axis=1, keepdims=True)
        
        return np.array(probs)
    
    @property
    def n_params(self):
        """Total number of trainable parameters."""
        quantum_params = np.prod(self.weight_shape)
        classical_params = self.n_qubits * self.n_classes + self.n_classes
        return quantum_params + classical_params


# ============================================================================
# DATA LOADING
# ============================================================================

def load_real_data(data_dir: str, datatype: str, n_features: int = 8):
    """Load real data from parquet files."""
    data_path = Path(data_dir) / f"{datatype}.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    log.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Separate features and labels
    if 'class' in df.columns:
        y = df['class'].values
        X = df.drop(columns=['class', 'case_id'], errors='ignore')
    else:
        raise ValueError("No 'class' column found in data")
    
    # Handle NaN
    X = X.fillna(X.median())
    X = X.values
    
    # Standardize and reduce dimensionality
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_features)
    X = pca.fit_transform(X)
    
    log.info(f"Data shape: X={X.shape}, y={y.shape}, classes={np.unique(y)}")
    
    return X, y


def generate_synthetic_data(n_samples: int = 100, n_features: int = 8, n_classes: int = 3, seed: int = 42):
    """Generate synthetic classification data."""
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([f"class_{i}" for i in range(n_classes)], size=n_samples)
    
    # Make data somewhat separable
    for i, cls in enumerate(np.unique(y)):
        mask = y == cls
        X[mask] += i * 0.5
    
    log.info(f"Generated synthetic data: X={X.shape}, y={y.shape}")
    
    return X, y


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    X_train, y_train, X_test, y_test,
    entangler_type: str,
    n_qubits: int,
    n_layers: int,
    n_classes: int,
    lr: float,
    steps: int,
    seed: int,
    verbose: bool = True
):
    """Run a single experiment with given entangler."""
    log.info(f"\n{'='*60}")
    log.info(f"Testing: {EntanglerComparisonClassifier.ENTANGLERS[entangler_type]['description']}")
    log.info(f"{'='*60}")
    
    # Create and train classifier
    clf = EntanglerComparisonClassifier(
        entangler_type=entangler_type,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=n_classes,
        lr=lr,
        steps=steps,
        seed=seed,
    )
    
    log.info(f"Parameters: {clf.n_params} total, weight_shape={clf.weight_shape}")
    
    start_time = time.time()
    clf.fit(X_train, y_train, verbose=verbose)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    results = {
        'entangler': entangler_type,
        'description': EntanglerComparisonClassifier.ENTANGLERS[entangler_type]['description'],
        'accuracy': accuracy,
        'f1_weighted': f1,
        'balanced_accuracy': balanced_acc,
        'train_time_sec': train_time,
        'n_params': clf.n_params,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'steps': steps,
        'seed': seed,
    }
    
    log.info(f"\nResults:")
    log.info(f"  Accuracy: {accuracy:.4f}")
    log.info(f"  F1 (weighted): {f1:.4f}")
    log.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
    log.info(f"  Training Time: {train_time:.2f}s")
    
    return results


def run_comparison(
    X, y,
    entanglers: list = None,
    n_qubits: int = 8,
    n_layers: int = 3,
    lr: float = 0.01,
    steps: int = 100,
    n_trials: int = 1,
    test_size: float = 0.2,
    verbose: bool = True,
):
    """Run comparison across multiple entanglers."""
    if entanglers is None:
        entanglers = ['basic', 'strongly', 'random', 'simplified']
    
    n_classes = len(np.unique(y))
    all_results = []
    
    for trial in range(n_trials):
        log.info(f"\n{'#'*60}")
        log.info(f"TRIAL {trial + 1}/{n_trials}")
        log.info(f"{'#'*60}")
        
        seed = 42 + trial
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        for entangler in entanglers:
            try:
                results = run_single_experiment(
                    X_train, y_train, X_test, y_test,
                    entangler_type=entangler,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    n_classes=n_classes,
                    lr=lr,
                    steps=steps,
                    seed=seed,
                    verbose=verbose,
                )
                results['trial'] = trial + 1
                all_results.append(results)
            except Exception as e:
                log.error(f"Error with {entangler}: {e}")
                continue
    
    return all_results


def summarize_results(results: list):
    """Summarize results across trials."""
    df = pd.DataFrame(results)
    
    # Group by entangler and compute statistics
    summary = df.groupby('entangler').agg({
        'accuracy': ['mean', 'std'],
        'f1_weighted': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'train_time_sec': ['mean', 'std'],
        'n_params': 'first',
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary


def print_summary_table(summary: pd.DataFrame):
    """Print a formatted summary table."""
    print("\n" + "="*100)
    print("ENTANGLER COMPARISON SUMMARY")
    print("="*100)
    
    # Add description
    for _, row in summary.iterrows():
        entangler = row['entangler']
        desc = EntanglerComparisonClassifier.ENTANGLERS[entangler]['description']
        print(f"\n{desc}")
        print(f"  Accuracy:          {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
        print(f"  F1 (weighted):     {row['f1_weighted_mean']:.4f} ± {row['f1_weighted_std']:.4f}")
        print(f"  Balanced Acc:      {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}")
        print(f"  Training Time:     {row['train_time_sec_mean']:.2f}s ± {row['train_time_sec_std']:.2f}s")
        print(f"  Parameters:        {int(row['n_params_first'])}")
    
    # Find best
    best_acc = summary.loc[summary['accuracy_mean'].idxmax()]
    best_f1 = summary.loc[summary['f1_weighted_mean'].idxmax()]
    fastest = summary.loc[summary['train_time_sec_mean'].idxmin()]
    
    print("\n" + "-"*100)
    print("CONCLUSIONS:")
    print(f"  Best Accuracy:  {best_acc['entangler']} ({best_acc['accuracy_mean']:.4f})")
    print(f"  Best F1:        {best_f1['entangler']} ({best_f1['f1_weighted_mean']:.4f})")
    print(f"  Fastest:        {fastest['entangler']} ({fastest['train_time_sec_mean']:.2f}s)")
    
    # Compute speed ratio
    basic_time = summary[summary['entangler'] == 'basic']['train_time_sec_mean'].values[0]
    strongly_time = summary[summary['entangler'] == 'strongly']['train_time_sec_mean'].values[0]
    speed_ratio = strongly_time / basic_time
    
    print(f"\n  StronglyEntangling is {speed_ratio:.1f}x slower than BasicEntangler")
    
    # Accuracy difference
    basic_acc = summary[summary['entangler'] == 'basic']['accuracy_mean'].values[0]
    strongly_acc = summary[summary['entangler'] == 'strongly']['accuracy_mean'].values[0]
    acc_diff = strongly_acc - basic_acc
    
    if acc_diff > 0.02:
        print(f"  StronglyEntangling has {acc_diff*100:.1f}% higher accuracy - WORTH the cost")
    elif acc_diff > 0:
        print(f"  StronglyEntangling has {acc_diff*100:.1f}% higher accuracy - MINIMAL gain")
    else:
        print(f"  StronglyEntangling has {abs(acc_diff)*100:.1f}% LOWER accuracy - NOT worth it")
    
    print("="*100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare quantum entangler architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='final_processed_datasets',
                        help='Directory with parquet data files')
    parser.add_argument('--datatype', type=str, default='GeneExpr',
                        help='Modality to use (GeneExpr, miRNA, etc.)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for quick testing')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of synthetic samples')
    
    # Model arguments
    parser.add_argument('--n_qubits', type=int, default=8,
                        help='Number of qubits')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of variational layers')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--steps', type=int, default=100,
                        help='Training steps')
    
    # Experiment arguments
    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of repeated trials')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set fraction')
    parser.add_argument('--entanglers', nargs='+', 
                        default=['basic', 'strongly', 'random', 'simplified'],
                        help='Entanglers to compare')
    parser.add_argument('--output', type=str, default='examples/entangler_comparison_results.csv',
                        help='Output CSV file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce verbosity')
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.synthetic:
        X, y = generate_synthetic_data(
            n_samples=args.n_samples,
            n_features=args.n_qubits,
            n_classes=3,
        )
    else:
        try:
            X, y = load_real_data(
                data_dir=args.data_dir,
                datatype=args.datatype,
                n_features=args.n_qubits,
            )
        except FileNotFoundError as e:
            log.warning(f"{e}")
            log.info("Falling back to synthetic data...")
            X, y = generate_synthetic_data(
                n_samples=args.n_samples,
                n_features=args.n_qubits,
                n_classes=3,
            )
    
    # Run comparison
    results = run_comparison(
        X, y,
        entanglers=args.entanglers,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        lr=args.lr,
        steps=args.steps,
        n_trials=args.n_trials,
        test_size=args.test_size,
        verbose=not args.quiet,
    )
    
    # Summarize and print
    summary = summarize_results(results)
    print_summary_table(summary)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    log.info(f"\nDetailed results saved to: {output_path}")
    
    summary.to_csv(output_path.with_suffix('.summary.csv'), index=False)
    log.info(f"Summary saved to: {output_path.with_suffix('.summary.csv')}")


if __name__ == '__main__':
    main()
