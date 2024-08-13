import numpy as np
from scipy.interpolate import CubicSpline


class MagnitudeWarping(object):
    """Perform the MagnitudeWarping to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : Probability of applying MagitudeWarping to the input signal.
        sigma    (float) : The standard deviation that controls the scale value.
        knot     (int)   : The number of knots in the spline interpolation, 
                           determining the number of control points for generating the random curves. 
                           More knots result in more variation in the curve.                   
    """

    def __init__(self, p=0.5, sigma=0.1, knot=4, seed=42):
        self.p = p
        self.sigma = sigma
        self.knot = knot
        self.seed = seed
 
    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) < self.p:
            sequence_length, input_dim = signal.shape[0], signal.shape[1]
            self.x = (np.ones((input_dim, 1)) * (np.arange(0, sequence_length, (sequence_length-1)/(self.knot+1)))).transpose()
            np.random.seed(self.seed)
            self.y = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot+2, input_dim))
            self.x_range = np.arange(sequence_length)
            self.randomCurves = np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range) for i in range(input_dim)]).transpose()

            signal_ = np.array(signal).copy()
            return signal_ * self.randomCurves
        return signal