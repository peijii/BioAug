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

# Example usage
if __name__ == '__main__':
    data = np.random.normal(loc=1, scale=1, size=(500, 6))
    gn = NonlinearDistortion(p=1.0, distortion_degree=2.0)
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
