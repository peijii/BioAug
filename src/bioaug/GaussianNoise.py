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
        """signal: [length, channel]"""
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            np.random.seed(self.seed)
            length = signal_.shape[0]
            channels = signal_.shape[1]
            for i in range(channels):
                noise = np.random.randn(length)
                noise = noise - np.mean(noise)
                signal_power = (1 / length) * np.sum(np.power(signal_[:, i], 2))
                noise_variance = signal_power / np.power(10, (self.SNR / 10))
                noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
                signal_[:, i] = signal_[:, i] + noise
            return signal_
        return signal


if __name__ == '__main__':
    data = np.random.normal(loc=1, scale=1, size=(500, 6))
    gn = GaussianNoise(p=1.0, SNR=15)
    aug_data = gn(data)

    raw_fig = plt.figure(figsize=(10, 10))
    for plt_index in range(1, 7):
        ax = raw_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), data[:, plt_index-1])

    aug_fig = plt.figure(figsize=(10, 10))
    for plt_index in range(1, 7):
        ax = aug_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), aug_data[:, plt_index-1], color='r')
    plt.show()