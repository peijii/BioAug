import numpy as np
import matplotlib.pyplot as plt


class LocalJittering(object):
    """Add localized high-frequency jitter to the input time-series data randomly with a given probability.

    Args:
        p            (float)  : Probability of applying jitter to the input signal.
        alpha        (float)  : Scale factor for the high-frequency noise.
        frequency    (int)    : Frequency of the jitter noise.
        duration     (int)    : Duration (in time steps) of each jitter event.
        num_jitters  (int)    : Number of jitter events to add to the signal.
        seed         (int)    : A seed value for the random number generator.
    """
    def __init__(self, p=0.5, alpha=0.5, frequency=50, duration=20, num_jitters=1, seed=42):
        self.p = p
        self.alpha = alpha
        self.frequency = frequency
        self.duration = duration
        self.num_jitters = num_jitters
        self.seed = seed

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]
           signal = signal + α * high_frequency_noise
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            np.random.seed(self.seed)
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]

            for _ in range(self.num_jitters):
                # Randomly choose a start point for the jitter
                start_point = np.random.randint(0, sequence_length - self.duration)
                end_point = start_point + self.duration
                
                # Generate high-frequency noise for each channel
                for i in range(input_dim):
                    t = np.arange(self.duration)
                    high_freq_noise = np.sin(2 * np.pi * self.frequency * t / self.duration)
                    high_freq_noise += np.random.normal(0, 1, self.duration)  # Add randomness
                    # Normalize noise
                    high_freq_noise = high_freq_noise / np.max(np.abs(high_freq_noise))
                    # Apply the jitter to the selected segment of the signal
                    signal_[start_point:end_point, i] += self.alpha * high_freq_noise
            return signal_
        return signal


if __name__ == '__main__':
    data = np.random.normal(loc=1, scale=1, size=(500, 6))
    gn = LocalJittering(p=1.0, alpha=10, frequency=10, duration=20, num_jitters=1)
    aug_data = gn(data)

    raw_fig = plt.figure(figsize=(5, 5))
    for plt_index in range(1, 7):
        ax = raw_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), data[:, plt_index-1])

    aug_fig = plt.figure(figsize=(5, 5))
    for plt_index in range(1, 7):
        ax = aug_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), aug_data[:, plt_index-1], color='r')
    plt.show()