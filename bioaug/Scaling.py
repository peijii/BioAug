import numpy as np


class Scaling(object):
    """Scaling the input time-series data randomly with a given probability.
    
    Args:
        p          (float): probability of applying scaling factor to the input signal.
        sigma      (float, tuple, list): Controls the standard deviation of the scaling factor.
    """
    def __init__(self, p=0.5, sigma=0.1):
        self.p = p
        self.sigma = sigma

    def _select_value(self, param):
        """Helper function to select a value from param if it's a tuple or list."""
        if isinstance(param, (int, float)):
            return param
        elif isinstance(param, tuple) and len(param) == 2:
            return np.random.uniform(param[0], param[1])
        elif isinstance(param, list):
            return np.random.choice(param)
        else:
            raise ValueError("Parameter must be an int, float, tuple of length 2, or list.")

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) < self.p:
            sequence_length, input_dim = signal.shape[0], signal.shape[1]
            sigma = self._select_value(self.sigma)
            self.scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, input_dim))
            self.factor = np.matmul(np.ones((sequence_length, 1)), self.scalingFactor)
            signal_ = np.array(signal).copy()
            signal_ *= self.factor
            return signal_
        return signal