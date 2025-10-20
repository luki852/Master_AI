import numpy as np


__all__ = ["Optimiser"]


class Optimiser:
    """ Base class for NNumpy optimisers. """

    def __init__(self, parameters, lr: float):
        """
        Create an optimiser instance.

        Parameters
        ----------
        parameters : iterable
            Iterable over the parameters that need to be updated.
        lr : float
            Learning rate or step size for updating the parameters.
        """
        self.parameters = list(parameters)
        if len(self.parameters) == 0:
            raise ValueError("no parameters to optimise")

        self.lr = float(lr)
        if self.lr < 0.:
            raise ValueError("learning rate must be positive")

        self.state = [self.init_state(par) for par in self.parameters]

    def init_state(self, par):
        """
        Create the initial optimiser state for a parameter.

        Parameters
        ----------
        par : Parameter
            The parameter to create the initial state for.

        Returns
        -------
        state : object
            The initial optimiser state for the given parameter.
        """
        return None

    def step(self):
        """
        Update all parameters under control of this optimiser
        by making one step in the update direction
        as computed by this algorithm for each of the parameters.
        """
        new_states = []
        for w, state in zip(self.parameters, self.state):
            delta_w, new_state = self.get_direction(w.grad, state)
            w -= self.lr * delta_w
            del w.grad  # safeguard
            new_states.append(new_state)

        self.state = new_states

    def get_direction(self, grad, state):
        """
        Compute the update direction from gradient and state for single parameter.

        Parameters
        ----------
        grad : ndarray
            Gradient direction.
        state : object or tuple of objects
            State information that is necessary to compute the update direction.

        Returns
        -------
        delta_w : ndarray
            The update direction according to the algorithm.
        new_state: object or tuple of objects
            Updated state information after computing the update direction.
        """
        raise NotImplementedError("method must be implemented in subclass")
