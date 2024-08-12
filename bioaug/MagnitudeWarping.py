import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class MagnitudeWarping(object):
    """Perform the MagnitudeWarping to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the MagitudeWarping. Default value is 0.5.
        sigma    (float) : sd of the scale value.
        knot     (int)   :                      .                     
        wSize    (int)   : Length of the input data.
        channels (int)   : Number of the input channels.
    """

    def __init__(self, sigma=0.1, knot=4, p=0.5, wSize=500, channels=6, seed=SEED):
        self.p = p
        self.x = (np.ones((channels, 1)) * (np.arange(0, wSize, (wSize-1)/(knot+1)))).transpose()
        np.random.seed(seed)
        self.y = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, channels))
        self.x_range = np.arange(wSize)
        if channels == 1:
            self.randomCurves = np.squeeze(np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range) for i in range(channels)]).transpose())
        else:
            self.randomCurves = np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range) for i in range(channels)]).transpose() 

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be performed the MagitudeWarping.

        Returns:
            Signal or Tensor: Randomly MagitudeWarping signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            return signal_ * self.randomCurves
        return signal