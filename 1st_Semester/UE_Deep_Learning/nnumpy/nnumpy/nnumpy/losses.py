"""
Error functions for training neural networks.

This module contains implementations for some common error functions.
"""

import numpy as np

from .base import Module
from .reductions import get_reduction

__all__ = ["LossFunction"]


class LossFunction(Module):
    """ Base class for NNumpy loss functions. """

    def __init__(self, reduction='mean', target_grads=False):
        """
        Set up the loss function.

        Parameters
        ----------
        reduction : {'none', 'sum', 'mean'}, optional
            Specification of how to reduce the results on the sample dimension.
        target_grads : bool, optional
            Flag to enable gradients w.r.t. to the target values.
        """
        super().__init__()
        self.reduction = reduction
        self.disable_target_grads = not target_grads
        self.reduction = get_reduction(reduction, axis=0)

    def compute_outputs(self, predictions, targets):
        raw_out, cache = self.raw_outputs(predictions, targets)
        out, r_cache = self.reduction.compute_outputs(raw_out)
        return out, (cache, r_cache)

    def compute_grads(self, grads, cache):
        cache, r_cache = cache
        raw_grads = self.reduction.compute_grads(grads, r_cache)
        return self.raw_grads(raw_grads, cache)

    def raw_outputs(self, predictions, targets):
        raise NotImplementedError

    def raw_grads(self, grads, cache):
        raise NotImplementedError
