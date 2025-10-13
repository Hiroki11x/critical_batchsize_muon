"""
Shampoo optimizer implementation.

Reference: https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo
"""

import torch
from torch.optim.optimizer import Optimizer


def matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """Compute matrix power using SVD decomposition."""
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    s_power = torch.pow(s, power)
    return u @ torch.diag(s_power) @ v


class ShampooOptimizer(Optimizer):
    """
    Shampoo optimizer implementation based on Facebook Research's distributed_shampoo.
    
    Shampoo is a second-order optimization method that uses matrix preconditioning
    to accelerate training by adapting to the geometry of the loss landscape.
    
    Reference: https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        epsilon: float = 1e-8,
        update_freq: int = 1,
        inv_freq: int = 1,
        precond_freq: int = 1,
        start_precond: int = 1000,
        block_size: int = 8192,
        dtype: torch.dtype = torch.float32,
        use_nesterov: bool = False,
        use_bias_correction: bool = True,
        use_decoupled_weight_decay: bool = True,
        graft_type: str = "none",  # "none", "adagrad", "sgd", "rmsprop"
        graft_epsilon: float = 1e-8,
        graft_beta1: float = 0.9,
        graft_beta2: float = 0.999,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not 0 < update_freq:
            raise ValueError(f"Invalid update_freq value: {update_freq}")
        if not 0 < inv_freq:
            raise ValueError(f"Invalid inv_freq value: {inv_freq}")
        if not 0 < precond_freq:
            raise ValueError(f"Invalid precond_freq value: {precond_freq}")
        if not 0 <= start_precond:
            raise ValueError(f"Invalid start_precond value: {start_precond}")
        if not 0 < block_size:
            raise ValueError(f"Invalid block_size value: {block_size}")
        if graft_type not in ["none", "adagrad", "sgd", "rmsprop"]:
            raise ValueError(f"Invalid graft_type: {graft_type}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            inv_freq=inv_freq,
            precond_freq=precond_freq,
            start_precond=start_precond,
            block_size=block_size,
            dtype=dtype,
            use_nesterov=use_nesterov,
            use_bias_correction=use_bias_correction,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            graft_type=graft_type,
            graft_epsilon=graft_epsilon,
            graft_beta1=graft_beta1,
            graft_beta2=graft_beta2,
        )
        super().__init__(params, defaults)
    
    def _init_state_for_param(self, param: torch.Tensor, group: dict) -> None:
        """Initialize state for a parameter."""
        state = self.state[param]
        
        # Initialize step counter
        if 'step' not in state:
            state['step'] = 0
        
        # Initialize momentum buffer
        if group['momentum'] > 0:
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(param)
        
        # Initialize graft optimizer state if needed
        if group['graft_type'] != "none":
            if 'graft_state' not in state:
                state['graft_state'] = {}
                if group['graft_type'] in ["adagrad", "rmsprop"]:
                    state['graft_state']['exp_avg_sq'] = torch.zeros_like(param)
                if group['graft_type'] == "rmsprop":
                    state['graft_state']['exp_avg'] = torch.zeros_like(param)
        
        # Initialize Shampoo preconditioner state for 2D+ tensors
        if param.ndim >= 2:
            if 'shampoo_state' not in state:
                state['shampoo_state'] = {}
                state['shampoo_state']['statistics'] = []
                state['shampoo_state']['preconditioners'] = []
                
                # Initialize statistics and preconditioners for each dimension
                for i in range(param.ndim):
                    dim_size = param.shape[i]
                    if dim_size > 1:  # Only create preconditioners for non-singleton dimensions
                        # Statistics matrix (dim_size x dim_size)
                        stats = torch.zeros(dim_size, dim_size, dtype=group['dtype'], device=param.device)
                        state['shampoo_state']['statistics'].append(stats)
                        
                        # Preconditioner matrix (dim_size x dim_size)
                        precond = torch.eye(dim_size, dtype=group['dtype'], device=param.device)
                        state['shampoo_state']['preconditioners'].append(precond)
                    else:
                        state['shampoo_state']['statistics'].append(None)
                        state['shampoo_state']['preconditioners'].append(None)
    
    def _compute_statistics(self, grad: torch.Tensor, param: torch.Tensor, group: dict) -> None:
        """Compute and update statistics for Shampoo preconditioner."""
        state = self.state[param]
        
        # Check if shampoo_state exists (only for 2D+ parameters)
        if param.ndim < 2 or 'shampoo_state' not in state:
            return
            
        shampoo_state = state['shampoo_state']
        
        # Update statistics for each dimension
        for i in range(param.ndim):
            if shampoo_state['statistics'][i] is not None:
                # Reshape gradient to compute statistics for dimension i
                grad_reshaped = grad.transpose(0, i).contiguous()
                grad_reshaped = grad_reshaped.view(grad_reshaped.shape[0], -1)
                
                # Compute outer product: grad_reshaped @ grad_reshaped.T
                stats_update = grad_reshaped @ grad_reshaped.T
                
                # Update statistics with exponential moving average
                shampoo_state['statistics'][i].mul_(group['momentum']).add_(stats_update, alpha=1 - group['momentum'])
    
    def _update_preconditioners(self, param: torch.Tensor, group: dict) -> None:
        """Update preconditioners using computed statistics."""
        state = self.state[param]
        
        # Check if shampoo_state exists (only for 2D+ parameters)
        if param.ndim < 2 or 'shampoo_state' not in state:
            return
            
        shampoo_state = state['shampoo_state']
        
        for i in range(param.ndim):
            if shampoo_state['statistics'][i] is not None:
                stats = shampoo_state['statistics'][i]
                
                # Add epsilon for numerical stability
                stats_reg = stats + group['epsilon'] * torch.eye(stats.shape[0], device=stats.device, dtype=stats.dtype)
                
                # Compute inverse square root using SVD
                try:
                    # Use matrix_power to compute inverse square root
                    inv_sqrt = matrix_power(stats_reg, -0.5)
                    shampoo_state['preconditioners'][i] = inv_sqrt
                except Exception as e:
                    # Fallback to identity if computation fails
                    print(f"Warning: Failed to compute preconditioner for dimension {i}: {e}")
                    shampoo_state['preconditioners'][i] = torch.eye(stats.shape[0], device=stats.device, dtype=stats.dtype)
    
    def _apply_preconditioner(self, grad: torch.Tensor, param: torch.Tensor, group: dict) -> torch.Tensor:
        """Apply Shampoo preconditioner to gradient."""
        state = self.state[param]
        
        # Check if shampoo_state exists (only for 2D+ parameters)
        if param.ndim < 2 or 'shampoo_state' not in state:
            return grad
            
        shampoo_state = state['shampoo_state']
        
        # Apply preconditioners for each dimension
        preconditioned_grad = grad.clone()
        
        for i in range(param.ndim):
            if shampoo_state['preconditioners'][i] is not None:
                precond = shampoo_state['preconditioners'][i]
                
                # Reshape gradient to apply preconditioner for dimension i
                grad_reshaped = preconditioned_grad.transpose(0, i).contiguous()
                original_shape = grad_reshaped.shape
                grad_reshaped = grad_reshaped.view(grad_reshaped.shape[0], -1)
                
                # Apply preconditioner: precond @ grad_reshaped
                preconditioned_reshaped = precond @ grad_reshaped
                
                # Reshape back
                preconditioned_reshaped = preconditioned_reshaped.view(original_shape)
                preconditioned_grad = preconditioned_reshaped.transpose(0, i).contiguous()
        
        return preconditioned_grad
    
    def _apply_graft(self, grad: torch.Tensor, param: torch.Tensor, group: dict) -> torch.Tensor:
        """Apply graft optimizer to gradient."""
        state = self.state[param]
        
        if group['graft_type'] == "none":
            return grad
        
        # Check if graft_state exists
        if 'graft_state' not in state:
            return grad
            
        graft_state = state['graft_state']
        
        if group['graft_type'] == "adagrad":
            # AdaGrad graft
            graft_state['exp_avg_sq'].add_(grad.square())
            graft_grad = grad / (graft_state['exp_avg_sq'].sqrt() + group['graft_epsilon'])
            
        elif group['graft_type'] == "rmsprop":
            # RMSprop graft
            graft_state['exp_avg_sq'].mul_(group['graft_beta2']).add_(grad.square(), alpha=1 - group['graft_beta2'])
            graft_grad = grad / (graft_state['exp_avg_sq'].sqrt() + group['graft_epsilon'])
            
        elif group['graft_type'] == "sgd":
            # SGD graft (no modification)
            graft_grad = grad
            
        else:
            graft_grad = grad
        
        return graft_grad
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state if needed
                self._init_state_for_param(p, group)
                
                state['step'] += 1
                step = state['step']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    if group['use_decoupled_weight_decay']:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                    else:
                        grad = grad.add(p, alpha=group['weight_decay'])
                
                # Compute statistics for Shampoo
                if step % group['update_freq'] == 0:
                    self._compute_statistics(grad, p, group)
                
                # Update preconditioners
                if (step >= group['start_precond'] and 
                    step % group['precond_freq'] == 0):
                    self._update_preconditioners(p, group)
                
                # Apply Shampoo preconditioner
                if step >= group['start_precond']:
                    preconditioned_grad = self._apply_preconditioner(grad, p, group)
                else:
                    preconditioned_grad = grad
                
                # Apply graft optimizer
                graft_grad = self._apply_graft(grad, p, group)
                
                # Combine Shampoo and graft gradients
                if group['graft_type'] != "none":
                    # Use graft gradient for scaling, Shampoo for direction
                    graft_norm = graft_grad.norm()
                    shampoo_norm = preconditioned_grad.norm()
                    if shampoo_norm > 0:
                        combined_grad = preconditioned_grad * (graft_norm / shampoo_norm)
                    else:
                        combined_grad = graft_grad
                else:
                    combined_grad = preconditioned_grad
                
                # Apply momentum
                if group['momentum'] > 0:
                    momentum_buffer = state['momentum_buffer']
                    if group['use_nesterov']:
                        # Nesterov momentum
                        momentum_buffer.mul_(group['momentum']).add_(combined_grad)
                        update = combined_grad.add(momentum_buffer, alpha=group['momentum'])
                    else:
                        # Standard momentum
                        momentum_buffer.mul_(group['momentum']).add_(combined_grad)
                        update = momentum_buffer
                else:
                    update = combined_grad
                
                # Apply bias correction
                if group['use_bias_correction'] and group['momentum'] > 0:
                    bias_correction = 1 - group['momentum'] ** step
                    update = update / bias_correction
                
                # Update parameter
                p.add_(update, alpha=-group['lr'])
        
        return loss 