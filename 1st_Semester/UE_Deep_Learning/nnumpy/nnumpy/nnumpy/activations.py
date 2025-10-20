"""
Activation functions for neural networks.

This module contains implementations for
some commonly used activation functions.
"""

import numpy as np

from .base import Module

__all__ = ['Identity']


class Identity(Module):
    """ NNumpy implementation of the identity function. """
        
    def compute_outputs(self, s):
        return s, None
    
    def compute_grads(self, grads, cache):
        return grads
