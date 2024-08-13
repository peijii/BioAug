import numpy as np


class RandomCutout(object):
    """Random cutout selected area of the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the random cutout. Default value is 0.5.
        area     (float) : size of fixed area.
        num      (int)   : number of the area.                     
        default  (float) : replace value of the cutout area 
    """

    def __init__(self, p=0.5, area=25, num=4, default=0.0, seed=42):
        self.p = p
        self.area = area
        self.num = num
        self.default = default
        self.seed = seed

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            sequence_length = signal.shape[0]
            mask = np.ones(sequence_length, np.float32)
            np.random.seed(self.seed)
            
            # Apply the cutout
            for _ in range(self.num):
                x = np.random.randint(sequence_length)
                x1 = np.clip(x - self.area // 2, 0, sequence_length)
                x2 = np.clip(x + self.area // 2, 0, sequence_length)
                mask[x1:x2] = 0
            
            # Apply the mask to all channels
            mask = np.expand_dims(mask, axis=1)  # Ensure mask has the right shape [sequence_length, 1]
            mask_b = 1 - mask  # Inverse mask for the cutout
            
            # Apply the mask to the signal
            signal_ = signal_ * mask + mask_b * self.default
            return signal_
        return signal