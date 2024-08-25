import numpy as np
import matplotlib.pyplot as plt

class NonlinearDistortion(object):
    """Simulate nonlinear distortion by applying a nonlinear transformation to the input time-series data.

    Args:
        p         (float) : Probability of applying nonlinear distortion to the input signal.
        distortion_degree (float): Degree of nonlinearity applied to the signal (n > 1).
        seed      (int)   : A seed value for the random number generator.
    """
    def __init__(self, p=0.5, distortion_degree=2.0, seed=42):
        self.p = p
        self.distortion_degree = distortion_degree
        self.seed = seed

    def __call__(self, signal):
        """Apply nonlinear distortion to the signal.
        
        Args:
            signal: [sequence_length, input_dim] - The original physiological signal.
        
        Returns:
            Distorted signal with nonlinear transformation.
        """
        if np.random.uniform(0, 1) < self.p:
            np.random.seed(self.seed)
            signal_ = np.array(signal).copy()
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]
            
            # Apply nonlinear transformation to all channels
            for i in range(input_dim):
                signal_[:, i] = np.sign(signal_[:, i]) * np.abs(signal_[:, i]) ** self.distortion_degree
            
            return signal_
        return signal
