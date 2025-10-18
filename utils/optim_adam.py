"""
Serializable Adam optimizer wrapper for PennyLane quantum models.

This module provides a custom Adam optimizer that:
- Exposes and persists optimizer state (m, v, t)
- Supports a step_and_cost API similar to PennyLane optimizers
- Can be saved/loaded with joblib for checkpointing
"""
import pennylane.numpy as np


class SerializableAdam:
    """
    A serializable Adam optimizer compatible with PennyLane's autograd.
    
    This optimizer maintains internal state (first and second moments, timestep)
    that can be serialized for checkpointing.
    
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
        
        # Optimizer state
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Timestep
        
    def step_and_cost(self, cost_fn, *params):
        """
        Perform one optimization step and return updated parameters and cost.
        
        Args:
            cost_fn: Cost function to minimize. Should accept *params and return scalar loss
            *params: Variable parameters to optimize (can be multiple arrays)
            
        Returns:
            tuple: (updated_params, loss_value)
                - updated_params: tuple of updated parameter arrays (or single array if one param)
                - loss_value: scalar loss from cost function
        """
        # Increment timestep
        self.t += 1
        
        # Compute gradients using PennyLane's autograd
        grad_fn = np.grad(cost_fn, argnum=list(range(len(params))))
        grads = grad_fn(*params)
        
        # Handle single parameter case
        if len(params) == 1:
            grads = (grads,)
        
        # Update parameters
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Initialize moments if first time seeing this parameter index
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)
            
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
        
        # Compute loss with updated parameters
        loss = cost_fn(*updated_params)
        
        # Return in same format as input (single array or tuple)
        if len(updated_params) == 1:
            return updated_params[0], loss
        else:
            return tuple(updated_params), loss
    
    def get_state(self):
        """
        Get the current optimizer state for serialization.
        
        Returns:
            dict: Dictionary containing m, v, t, and hyperparameters
        """
        return {
            'm': dict(self.m),  # Convert to regular dict for serialization
            'v': dict(self.v),
            't': self.t,
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps
        }
    
    def set_state(self, state_dict):
        """
        Restore optimizer state from a dictionary.
        
        Args:
            state_dict (dict): State dictionary from get_state()
        """
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']
        self.lr = state_dict.get('lr', self.lr)
        self.beta1 = state_dict.get('beta1', self.beta1)
        self.beta2 = state_dict.get('beta2', self.beta2)
        self.eps = state_dict.get('eps', self.eps)
    
    def reset_state(self):
        """Reset optimizer state (useful when starting a new training phase)."""
        self.m = {}
        self.v = {}
        self.t = 0
