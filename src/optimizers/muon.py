"""
Muon optimizer implementation.

Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
"""

import torch
from torch.optim.optimizer import Optimizer


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


class SingleDeviceMuonWithAuxAdam(Optimizer):
    """
    Single device Muon optimizer with auxiliary Adam optimizer.
    
    This optimizer applies Muon to selected parameters and Adam to the rest.
    """
    
    def __init__(self, param_groups, **kwargs):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
                group.setdefault("ns_steps", 3)
                group.setdefault("nesterov", True)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, {})

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