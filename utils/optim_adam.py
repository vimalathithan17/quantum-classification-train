"""
Serializable Adam optimizer wrapper compatible with PennyLane numpy arrays.
Supports step_and_cost(cost_fn, *params) and get_state()/set_state() for persistence.
"""
import pennylane.numpy as np
from typing import Callable, Tuple, Dict, Any


class SerializableAdam:
    """
    Adam optimizer compatible with pennylane.numpy arrays and joblib serialization.
    
    Implements:
    - step_and_cost(cost_fn, *params) -> (updated_params_tuple, loss)
    - get_state() -> dict with m, v, t for serialization
    - set_state(state) -> restores optimizer state
    """
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate (default: 0.01)
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State variables (initialized on first step)
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Time step
        
    def _initialize_state(self, params: Tuple) -> None:
        """Initialize optimizer state based on parameter shapes."""
        self.m = tuple(np.zeros_like(p) for p in params)
        self.v = tuple(np.zeros_like(p) for p in params)
        self.t = 0
        
    def step_and_cost(self, cost_fn: Callable, *params) -> Tuple[Tuple, float]:
        """
        Perform one optimization step and return updated parameters and loss.
        
        Args:
            cost_fn: Cost function that takes *params and returns scalar loss
            *params: Variable number of parameter arrays to optimize
            
        Returns:
            Tuple of (updated_params_tuple, loss_value)
        """
        # Initialize state on first call
        if self.m is None:
            self._initialize_state(params)
            
        # Compute loss and gradients
        loss = cost_fn(*params)
        grads = tuple(np.grad(cost_fn, argnum=i)(*params) for i in range(len(params)))
        
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate
        m_new = tuple(self.beta1 * m_i + (1 - self.beta1) * g_i 
                      for m_i, g_i in zip(self.m, grads))
        
        # Update biased second raw moment estimate
        v_new = tuple(self.beta2 * v_i + (1 - self.beta2) * (g_i ** 2) 
                      for v_i, g_i in zip(self.v, grads))
        
        # Compute bias-corrected first moment estimate
        m_hat = tuple(m_i / (1 - self.beta1 ** self.t) for m_i in m_new)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = tuple(v_i / (1 - self.beta2 ** self.t) for v_i in v_new)
        
        # Update parameters
        params_new = tuple(p - self.learning_rate * m_i / (np.sqrt(v_i) + self.epsilon)
                          for p, m_i, v_i in zip(params, m_hat, v_hat))
        
        # Store state
        self.m = m_new
        self.v = v_new
        
        return params_new, float(loss)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get optimizer state for serialization.
        
        Returns:
            Dictionary containing m, v, t, and hyperparameters
        """
        return {
            'm': self.m,
            'v': self.v,
            't': self.t,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore optimizer state from dictionary.
        
        Args:
            state: Dictionary with optimizer state (from get_state())
        """
        self.m = state.get('m')
        self.v = state.get('v')
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.epsilon = state.get('epsilon', self.epsilon)
