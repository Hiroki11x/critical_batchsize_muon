# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Muon optimizer implementation.

Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
"""

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["MuonWithAuxAdam"]


def zeropower_via_newtonschulz5(G, steps: int):
    """Compute matrix power using Newton-Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """Update function for Muon optimizer."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim >= 2:
        original_shape = update.shape
        reshaped_update = update.view(original_shape[0], -1)
        processed_update = zeropower_via_newtonschulz5(
            reshaped_update, steps=ns_steps)
        processed_update *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return processed_update.view(original_shape)
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Update function for Adam optimizer."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)



class MuonWithAuxAdam(Optimizer):
    """
    Single device Muon optimizer with auxiliary Adam optimizer.
    
    This optimizer automatically applies Muon to 2D+ parameters and Adam to 1D parameters
    within a single model part, matching the interface expected by OptimizersContainer.
    """

    def __init__(self, params, muon_lr=0.02, muon_momentum=0.95, muon_wd=0.0, muon_nesterov=True, muon_ns_steps=3,
                 adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.0, 
                 force_adam_only=False, **kwargs):
        
        # Auto-detect if this is a first/last layer based on parameter names and shapes
        param_list = list(params) if hasattr(params, '__iter__') else [params]
        is_embedding_or_output = self._detect_embedding_or_output_layer(param_list)
        
        if is_embedding_or_output:
            force_adam_only = True
            print(f"MUON: Detected embedding/output layer, forcing Adam-only optimization")
        
        # Separate parameters into MUON and Adam groups based on dimensionality
        muon_params = []
        adam_params = []
        
        for p in param_list:
            if p.requires_grad:
                if force_adam_only or p.ndim < 2:
                    adam_params.append(p)
                else:
                    muon_params.append(p)
        
        # Create parameter groups
        param_groups = [
            {
                "params": muon_params, "use_muon": True,
                "lr": muon_lr, "momentum": muon_momentum, "weight_decay": muon_wd,
                "ns_steps": muon_ns_steps, "nesterov": muon_nesterov,
            },
            {
                "params": adam_params, "use_muon": False,
                "lr": adamw_lr, "betas": adamw_betas, "eps": adamw_eps,
                "weight_decay": adamw_wd,
            },
        ]
        
        super().__init__(param_groups, {})

    def _detect_embedding_or_output_layer(self, param_list):
        """Detect if this is an embedding or output layer based on parameter characteristics."""
        # Heuristic: If there's only one large 2D parameter, likely embedding or output
        large_2d_params = [p for p in param_list if p.ndim == 2 and p.numel() > 1000]
        small_params = [p for p in param_list if p.numel() <= 1000]
        
        # Embedding/output layers typically have:
        # 1. One large 2D weight matrix (vocab_size x embed_dim or embed_dim x vocab_size)
        # 2. Few or no small parameters (no bias, layernorm, etc.)
        if len(large_2d_params) == 1 and len(small_params) <= 1:
            return True
        return False

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            params_with_grad = [
                p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue
            if group["use_muon"]:
                for p in params_with_grad:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p.grad)
                    update = muon_update(
                        p.grad, state['momentum_buffer'], beta=group['momentum'],
                        ns_steps=group['ns_steps'], nesterov=group['nesterov'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])
            else:
                for p in params_with_grad:
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'], state['exp_avg'], state['exp_avg_sq'] = 0, torch.zeros_like(
                            p), torch.zeros_like(p)
                    state['step'] += 1
                    update = adam_update(
                        p.grad, state['exp_avg'], state['exp_avg_sq'],
                        state['step'], group['betas'], group['eps'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])