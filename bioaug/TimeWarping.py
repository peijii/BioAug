import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class TimeWarping(object):
    """Perform the TimeWarping to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of applying TimeWarping to the input signal.
        sigma    (float) : The standard deviation that controls the intensity of the time warping. 
                           A larger sigma value leads to more pronounced temporal distortions, 
                           simulating greater time shifts..
        knot     (int)   : The number of knots in the spline interpolation, 
                           determining the control points for the time warping. 
                           More knots result in more variation in the time distortion..                     
    """

    def __init__(self, p=0.5, sigma=0.1, knot=4, seed=42):
        self.p = p
        self.sigma = sigma
        self.knot = knot
        self.seed = seed

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            sequence_length = signal.shape[0]
            input_dim = signal.shape[1]
            self.init_seed(sequence_length, input_dim, self.seed)
            if len(signal_.shape) == 1:
                signal_ = signal_[:, np.newaxis]

            signal_new = np.zeros((sequence_length, input_dim))
            x_range = np.arange(sequence_length)
            for i in range(input_dim):
                signal_new[:, i] = np.interp(x_range, self.tt_cum[:, 0], signal_[:, i])
            return signal_new
        return signal

    def init_seed(self, sequence_length, input_dim, seed=42):
        self.x = (np.ones((input_dim, 1)) * (np.arange(0, sequence_length, (sequence_length - 1) / (self.knot + 1)))).transpose()
        np.random.seed(seed)
        self.y = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2, input_dim))
        self.x_range = np.arange(sequence_length)
        self.tt = np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range) for i in range(input_dim)]).transpose()
        self.tt_cum = np.cumsum(self.tt, axis=0)
        # set the shape
        self.t_scale = [(sequence_length - 1) / self.tt_cum[-1, i] for i in range(input_dim)]
        for i in range(input_dim):
            self.tt_cum[:, i] = self.tt_cum[:, i] * self.t_scale[i]


if __name__ == '__main__':
    data = np.random.normal(loc=1, scale=1, size=(500, 6))
    fn = TimeWarping(p=1.0, sigma=0.8, knot=4)
    aug_data = fn(data)

    raw_fig = plt.figure(figsize=(5, 5))
    for plt_index in range(1, 7):
        ax = raw_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), data[:, plt_index-1])

    aug_fig = plt.figure(figsize=(5, 5))
    for plt_index in range(1, 7):
        ax = aug_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), aug_data[:, plt_index-1], color='r')
    plt.show()