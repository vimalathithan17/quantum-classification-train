"""
Serializable Adam optimizer compatible with PennyLane numpy.
This optimizer can save and restore its state for checkpoint resumption.
"""
from pennylane import numpy as np


class SerializableAdam:
    """
    Adam optimizer with state serialization support.
    
    Compatible with PennyLane's autograd numpy and supports saving/loading
    optimizer state for checkpoint resumption.
    
    Args:
        lr (float): Learning rate
        beta1 (float): Exponential decay rate for first moment estimates (default: 0.9)
        beta2 (float): Exponential decay rate for second moment estimates (default: 0.999)
        eps (float): Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # timestep
        self.m = None  # first moment estimates
        self.v = None  # second moment estimates
    
    def step_and_cost(self, cost_fn, *params):
        """
        Perform one optimization step and return updated parameters and cost.
        
        Args:
            cost_fn: Cost function that takes parameters and returns scalar loss
            *params: Variable number of parameter arrays to optimize
        
        Returns:
            tuple: ((updated_params...), loss_value)
        """
        self.t += 1
        
        # Compute gradient and cost
        grad_and_cost = np.autograd.grad_and_value(cost_fn)
        grads = grad_and_cost(*params)
        
        # grads is a tuple of (gradient_tuple, cost_value)
        if isinstance(grads, tuple) and len(grads) == 2:
            param_grads, cost_value = grads
        else:
            raise ValueError("grad_and_cost should return (gradients, cost)")
        
        # Initialize moments on first step
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        # Ensure param_grads is a list/tuple
        if not isinstance(param_grads, (list, tuple)):
            param_grads = [param_grads]
        
        # Update parameters
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, param_grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            updated_params.append(updated_param)
        
        # Return as tuple if multiple params, single value if one param
        if len(updated_params) == 1:
            return (updated_params[0], cost_value)
        else:
            return (tuple(updated_params), cost_value)
    
    def get_state(self):
        """
        Get the current optimizer state for serialization.
        
        Returns:
            dict: Optimizer state with numpy arrays (not pennylane arrays)
        """
        state = {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,
            'm': [np.array(mi) for mi in self.m] if self.m is not None else None,
            'v': [np.array(vi) for vi in self.v] if self.v is not None else None,
        }
        return state
    
    def set_state(self, state):
        """
        Restore optimizer state from a saved state dict.
        
        Args:
            state (dict): Optimizer state dictionary
        """
        self.lr = state['lr']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
        self.t = state['t']
        
        # Restore moments with requires_grad=False to avoid autograd tracking
        if state['m'] is not None:
            self.m = [np.array(mi, requires_grad=False) for mi in state['m']]
        else:
            self.m = None
        
        if state['v'] is not None:
            self.v = [np.array(vi, requires_grad=False) for vi in state['v']]
        else:
            self.v = None
