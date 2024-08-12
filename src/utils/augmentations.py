# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import CubicSpline
from time import time

SEED = 42



class uLawNormalization(object):
    """
    """
    def __init__(self, p=1.0, u=256):
        self.p = p
        self.u = u

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be performed the RandomSampling.

        Returns:
            Signal or Tensor: Randomly RandomSampling signal.
        """
        if np.random.uniform(0, 1) <= self.p:
            signal_ = np.array(signal).copy()
            signal_ = np.sign(signal_) * np.log(1 + self.u * abs(signal_)) / np.log(1 + self.u)
            return signal_
        return signal