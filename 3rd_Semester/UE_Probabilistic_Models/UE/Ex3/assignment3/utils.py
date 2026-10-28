import numpy as np
from bayesian_network import BayesNet, Variable

def sample_categorical(dist: np.ndarray) -> np.int64:
    """
    Draws a single sample from a categorical distribution.
    :param dist: NumPy array listing the probability of each outcome.
    :returns: Index of the sampled element.
    """

    assert type(dist) == np.ndarray
    assert dist.ndim == 1
    assert np.isclose(dist.sum(), 1)

    return np.random.choice(len(dist), p=dist)


def kld(p, q):
    """
    Computes the Kullback-Leibler divergence between p and q.
    :param p: true distribution
    :param q: estimated distribution
    :return: Kullback-Leibler Divergence between p and q
    """
    return (p * np.log(p / (q + 1e-10))).sum()  # add a small constant for numeric stability


def approx_error(bayes_net, approx_function, exact, query_variable, evidence, sample_counts, n_runs=100, **kwargs):
    """
    Computes the approximation error for a given approximation method.
    """
    mean_errors = []
    
    for num_samples in sample_counts:
        estimates = np.array([approx_function(bayes_net, query_variable, evidence, num_samples, **kwargs) for i in range(n_runs)])
        mean_errors.append(kld(exact, estimates) / len(estimates))
    return mean_errors


def get_default_bayes_net(resampling_distribution=None):
    _A_, _B_, _C_, _D_, _E_ = 0, 1, 2, 3, 4

    A = np.array([0.2, 0.8])
    B_A = np.array([[0.9, 0.2], [0.1, 0.8]])
    C = np.array([0.9, 0.1])
    D_BC = np.array([[[0.1, 0.2], [0.99, 0.8]], [[0.9, 0.8], [0.01, 0.2]]])
    E_C = np.array([[0.7, 0.4], [0.3, 0.6]])

    class BayesNetMock(BayesNet):
        def __iter__(self):
            for i in [2, 4, 0, 1, 3]:
                yield self.nodes[i]  
    
    bn = BayesNetMock(
        (A, [_A_]),
        (B_A, [_B_, _A_]),
        (C, [_C_]),
        (D_BC, [_D_, _B_, _C_]),
        (E_C, [_E_, _C_]),
        resampling_distribution=resampling_distribution
    )

    return bn
