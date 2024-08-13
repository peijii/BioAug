import numpy as np


class uLawNormalization(object):
    """Perform the MagnitudeWarping to the input time-series data randomly with a given probability.
    
    Args:
        p    (float) : Probability of applying uLawNormalization to the input signal.
        u    (float) : The parameter controlling the compression strength. The larger the u value, 
                       the stronger the compression.                  
    """
    def __init__(self, p=1.0, u=8):
        self.p = p
        self.u = u

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) <= self.p:
            signal_ = np.array(signal).copy()
            signal_ = np.sign(signal_) * np.log(1 + self.u * abs(signal_)) / np.log(1+self.u)
            return signal_
        return signal