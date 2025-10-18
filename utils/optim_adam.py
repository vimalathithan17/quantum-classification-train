"""Serializable Adam optimizer for PennyLane quantum machine learning."""
from pennylane import numpy as np


class AdamSerializable:
    """
    Serializable Adam optimizer compatible with PennyLane autograd.
    
    This optimizer maintains internal state (m, v, t) that can be saved and restored
    for checkpoint/resume functionality.
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Initialize the Adam optimizer.
        
        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            eps: Small constant for numerical stability
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None
        
    def step_and_cost(self, cost_fn, *params):
        """
        Compute cost and update parameters using Adam algorithm.
        
        Args:
            cost_fn: Cost function that takes parameters and returns scalar loss
            *params: Variable number of parameter arrays to optimize
            
        Returns:
            tuple: (updated_params_tuple, loss_value)
        """
        # Compute loss and gradients
        loss = cost_fn(*params)
        grads = np.autograd.grad(cost_fn)(*params)
        
        # Ensure grads is a tuple
        if not isinstance(grads, tuple):
            grads = (grads,)
        
        # Initialize moments on first step
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        # Increment timestep
        self.t += 1
        
        # Update parameters
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            updated_params.append(updated_param)
        
        return tuple(updated_params), loss
    
    def get_state(self):
        """
        Get optimizer state for serialization.
        
        Returns:
            dict: Dictionary containing optimizer state (m, v, t)
        """
        if self.m is None:
            return {'m': None, 'v': None, 't': self.t}
        
        # Convert to plain numpy arrays for joblib serialization
        return {
            'm': [np.asarray(mi) for mi in self.m],
            'v': [np.asarray(vi) for vi in self.v],
            't': self.t
        }
    
    def set_state(self, state_dict):
        """
        Restore optimizer state from dictionary.
        
        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.t = state_dict['t']
        if state_dict['m'] is not None:
            self.m = [np.array(mi, requires_grad=False) for mi in state_dict['m']]
            self.v = [np.array(vi, requires_grad=False) for vi in state_dict['v']]
        else:
            self.m = None
            self.v = None
    
    def set_lr(self, lr):
        """
        Set learning rate.
        
        Args:
            lr: New learning rate
        """
        self.lr = lr
