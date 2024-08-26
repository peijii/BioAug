import numpy as np
import matplotlib.pyplot as plt


class SignalDrift(object):
    """Simulate signal drift by applying a linear or low-frequency drift to the input time-series data.

    Args:
        p         (float) : Probability of applying signal drift to the input signal.
        drift_rate (float): Rate of the linear drift applied to the signal.
        seed      (int)   : A seed value for the random number generator.
    """
    def __init__(self, p=0.5, drift_rate=0.01, func='linear', seed=42):
        self.p = p
        self.drift_rate = drift_rate
        self.func = func
        self.seed = seed

    def __call__(self, signal):
        """Apply signal drift to the signal.
        
        Args:
            signal: [sequence_length, input_dim] - The original physiological signal.
        
        Returns:
            Modulated signal with signal drift.
        """
        if np.random.uniform(0, 1) < self.p:
            np.random.seed(self.seed)
            signal_ = np.array(signal).copy()
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]
            
            # Generate a drift signal based on the chosen function
            t = np.linspace(0, 1, sequence_length)
            
            if self.func == 'linear':
                drift = self.drift_rate * t
            elif self.func == 'exp':
                drift = self.drift_rate * (np.exp(t) - 1)  # Exponential drift
            else:
                raise ValueError("Function type must be 'linear'")
            
            # Apply the drift to all channels
            for i in range(input_dim):
                signal_[:, i] = signal_[:, i] + drift
            
            return signal_
        return signal
