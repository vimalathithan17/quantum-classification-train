"""Smoke tests for quantum classification training."""
import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all required modules can be imported."""
    try:
        from utils.optim_adam import AdamSerializable
        from utils.io_checkpoint import save_checkpoint, load_checkpoint
        print("✓ All utility imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_optimizer_state_save_load():
    """Test optimizer state save/load functionality."""
    try:
        from utils.optim_adam import AdamSerializable
        from pennylane import numpy as np
        
        # Create optimizer
        opt = AdamSerializable(lr=0.01)
        
        # Create dummy parameters
        param1 = np.array([1.0, 2.0, 3.0], requires_grad=True)
        param2 = np.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        # Define simple cost function
        def cost_fn(p1, p2):
            return np.sum(p1**2) + np.sum(p2**2)
        
        # Take a step
        (new_p1, new_p2), loss = opt.step_and_cost(cost_fn, param1, param2)
        
        # Save state
        state = opt.get_state()
        assert state['t'] == 1, "Step count should be 1"
        assert state['m'] is not None, "Momentum should be initialized"
        assert state['v'] is not None, "Velocity should be initialized"
        
        # Create new optimizer and load state
        opt2 = AdamSerializable(lr=0.01)
        opt2.set_state(state)
        
        assert opt2.t == 1, "Restored step count should match"
        print("✓ Optimizer state save/load test passed")
        return True
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        return False


def test_checkpoint_save_load():
    """Test checkpoint save/load functionality."""
    try:
        from utils.io_checkpoint import (
            save_checkpoint, load_checkpoint, save_best_checkpoint,
            save_periodic_checkpoint, find_latest_checkpoint
        )
        from pennylane import numpy as np
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create test data
            test_data = {
                'step': 10,
                'weights_quantum': np.array([1.0, 2.0, 3.0]),
                'weights_classical': {
                    'W1': np.array([[1.0, 2.0]]),
                    'b1': np.array([0.1])
                },
                'optimizer_state': {'m': [], 'v': [], 't': 10},
                'best_val_metric': 0.85,
                'metric_name': 'weighted_f1'
            }
            
            # Test basic save/load
            checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.joblib')
            save_checkpoint(checkpoint_path, test_data)
            loaded_data = load_checkpoint(checkpoint_path)
            
            assert loaded_data['step'] == test_data['step']
            assert loaded_data['metric_name'] == test_data['metric_name']
            
            # Test best checkpoint
            save_best_checkpoint(temp_dir, test_data)
            best_path = os.path.join(temp_dir, 'best_weights.joblib')
            assert os.path.exists(best_path)
            
            # Test periodic checkpoint
            save_periodic_checkpoint(temp_dir, 20, test_data, keep_last_n=3)
            latest = find_latest_checkpoint(temp_dir)
            assert latest is not None
            
            print("✓ Checkpoint save/load test passed")
            return True
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        return False


def test_minimal_training():
    """Test minimal training loop with checkpointing."""
    try:
        # This test requires pennylane to be installed
        # Skip if not available
        try:
            import pennylane as qml
        except ImportError:
            print("⊘ Minimal training test skipped (pennylane not installed)")
            return True
        
        from utils.optim_adam import AdamSerializable
        from utils.io_checkpoint import save_checkpoint
        from pennylane import numpy as np
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create simple quantum circuit
            dev = qml.device('default.qubit', wires=2)
            
            @qml.qnode(dev, interface='autograd')
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(2))
                qml.BasicEntanglerLayers(weights, wires=range(2))
                return [qml.expval(qml.PauliZ(i)) for i in range(2)]
            
            # Create dummy data
            X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            y = np.array([0, 1, 0])
            y_one_hot = np.eye(2)[y]
            
            # Initialize weights
            weights = np.random.uniform(0, 2*np.pi, (2, 2), requires_grad=True)
            
            # Create optimizer
            opt = AdamSerializable(lr=0.01)
            
            # Train for 2 steps
            for step in range(2):
                def cost(w):
                    preds = np.array([circuit(x, w) for x in X])
                    probs = np.array([np.exp(p) / np.sum(np.exp(p)) for p in preds])
                    return -np.mean(y_one_hot * np.log(probs + 1e-9))
                
                weights, loss = opt.step_and_cost(cost, weights)
            
            # Save checkpoint
            checkpoint_data = {
                'step': 2,
                'weights_quantum': weights,
                'optimizer_state': opt.get_state()
            }
            checkpoint_path = os.path.join(temp_dir, 'training_checkpoint.joblib')
            save_checkpoint(checkpoint_path, checkpoint_data)
            
            assert os.path.exists(checkpoint_path)
            print("✓ Minimal training test passed")
            return True
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ Minimal training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running smoke tests for quantum classification training")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_optimizer_state_save_load,
        test_checkpoint_save_load,
        test_minimal_training
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
