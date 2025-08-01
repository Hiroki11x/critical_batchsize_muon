"""
Optimizers package containing custom optimizer implementations.
"""

from .muon import SingleDeviceMuonWithAuxAdam
from .shampoo import ShampooOptimizer

__all__ = [
    'SingleDeviceMuonWithAuxAdam',
    'ShampooOptimizer',
] 