import numpy as np
import matplotlib.pyplot as plt


class GaussianNoise(object):
    """Jittering the input time-series data randomly with a given probability.

    Args:
        p       (float) : Probability of applying gaussian noise to the input signal.
        SNR     (float) : Signal-to-Noise Ratio, which determines the relative level of noise to be added.
        seed    (int)   : A seed value for the random number generator.
    """
    def __init__(self, p=0.5, SNR=25, seed=0):
        self.p = p
        self.SNR = SNR
        self.seed = seed

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            np.random.seed(self.seed)
            sequence_length = signal_.shape[0]
            input_dim = signal_.shape[1]
            for i in range(input_dim):
                noise = np.random.randn(sequence_length)
                noise = noise - np.mean(noise)
                signal_power = (1 / sequence_length) * np.sum(np.power(signal_[:, i], 2))
                noise_variance = signal_power / np.power(10, (self.SNR / 10))
                noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
                signal_[:, i] = signal_[:, i] + noise
            return signal_
        return signal