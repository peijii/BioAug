import numpy as np
import matplotlib.pyplot as plt


class Scaling(object):
    """Scaling the input time-series data randomly with a given probability.
    
    Args:
        p          (float): probability of applying scaling factor to the input signal.
        sigma      (float): controls the standard deviation of the scaling factor.
    """
    def __init__(self, p=0.5, sigma=0.1):
        self.p = p
        self.sigma = sigma

    def __call__(self, signal):
        """signal: [length, channel]"""
        if np.random.uniform(0, 1) < self.p:
            length = signal.shape[0]
            channels = signal.shape[1]
            self.scalingFactor = np.random.normal(loc=1.0, scale=self.sigma, size=(1, channels))
            self.factor = np.matmul(np.ones((length, 1)), self.scalingFactor)
            signal_ = np.array(signal).copy()
            signal_ *= self.factor
            return signal_
        return signal


if __name__ == '__main__':
    data = np.random.normal(loc=1, scale=1, size=(500, 6))
    gn = Scaling(p=1.0, sigma=0.5)
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

