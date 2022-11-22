# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import CubicSpline
from time import time

SEED = 42

# 1. "Jittering" can be considered as applying different noise to each sample. 
class Jittering(object):
    """Jittering the input time-series data randomly with a given probability.
    
    Args:
        p        (float): probability of the sensor data being jittered. Default value is 0.5.
        var1     (float): sd of the noise1 (FMG).
        var2     (float): sd of the noise2 (EMG).
        wSize    (int)  : Length of the input data.
        channels (int)  : Number of the input channels.
    """

    def __init__(self, sigma, p=0.5, wSize=500, channels=6) -> None:
        self.p = p
        # self.fmg_noise = np.zeros(shape=(wSize, channels // 2))
        # self.emg_noise = np.zeros(shape=(wSize, channels // 2))
        # for i in range(channels // 2): self.fmg_noise[:, i] = np.random.normal(0.0, sigma, size=(wSize,))
        # for j in range(channels // 2): self.emg_noise[:, j] = np.random.normal(0.0, sigma, size=(wSize,))

        # self.noise = np.concatenate([self.fmg_noise, self.emg_noise], axis=1)
        if channels == 1:
            self.noise = np.squeeze(np.random.normal(loc=0., scale=sigma, size=(wSize, channels)))
        else:
            self.noise = np.random.normal(loc=0., scale=sigma, size=(wSize, channels))

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be jittered.

        Returns:
            Signal or Tensor: Randomly jittered signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            signal_ += self.noise
            return signal_
        return signal

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(p={})'.format(self.p)


# 2. "Scaling" can be considered as "applying constant noise to the entire samples"
class Scaling(object):
    """Scaling the input time-series data randomly with a given probability.
    
    Args:
        p       (float): probability of the sensor data being jittered. Default value is 0.5.
        sigma   (float): sd of the scale value
        wSize    (int)  : Length of the input data.
        channels (int)  : Number of the input channels.
    """

    def __init__(self, sigma=0.1, p=0.5, wSize=500, channels=6):
        self.p = p
        self.scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, channels))
        if channels == 1:
            self.noise = np.squeeze(np.matmul(np.ones((wSize, 1)), self.scalingFactor))
        else:
            self.noise = np.matmul(np.ones((wSize, 1)), self.scalingFactor)

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be scaled.

        Returns:
            Signal or Tensor: Randomly scaled signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            signal_ *= self.noise
            return signal_
        return signal

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(p={})'.format(self.p)


# 3. Permutation
class Permutation(object):
    """Permutate the input time-series data randomly with a given probability.
    
    Args:
        p       (float) : probability of the sensor data being jittered. Default value is 0.5.
        sigma   (float) : sd of the scale value
        wSize    (int)  : Length of the input data.
        channels (int)  : Number of the input channels.
    """

    def __init__(self, nPerm=4, minSegLength=10, p=0.5, wSize=500, channels=6):
        self.p = p
        self.wSize = wSize
        self.nPerm = nPerm
        self.minSegLength = minSegLength
        self.channels = channels

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be Permutated.

        Returns:
            Signal or Tensor: Randomly scaled signal.
        """
        if np.random.uniform(0, 1) < self.p:
            if self.channels == 1:
                signal_ = np.squeeze(np.zeros((self.wSize, self.channels)))
            else:
                signal_ = np.zeros((self.wSize, self.channels))
            idx = np.random.permutation(self.nPerm)
            flag = True
            while flag == True:
                segs = np.zeros(self.nPerm + 1, dtype=int)
                segs[1:-1] = np.sort(
                    np.random.randint(self.minSegLength, self.wSize - self.minSegLength, self.nPerm - 1))
                segs[-1] = self.wSize
                if np.min(segs[1:] - segs[0:-1]) > self.minSegLength:
                    flag = False

            pp = 0
            for i in range(self.nPerm):
                if self.channels == 1:
                    signal_temp = signal[segs[idx[i]]:segs[idx[i] + 1]]
                    signal_[pp:pp + len(signal_temp)] = signal_temp
                else:
                    signal_temp = signal[segs[idx[i]]:segs[idx[i] + 1], :]
                    signal_[pp:pp + len(signal_temp), :] = signal_temp
                pp += len(signal_temp)
            return signal_
        return signal

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(p={})'.format(self.p)


# 4. "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
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


# 5. Time Warping
class TimeWarping(object):
    """Perform the TimeWarping to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the TimeWarping. Default value is 0.5.
        sigma    (float) : sd of the scale value.
        knot     (int)   :                      .                     
        wSize    (int)   : Length of the input data.
        channels (int)   : Number of the input channels.
    """

    def __init__(self, sigma=0.1, knot=4, p=0.5, wSize=20, channels=10, keep=False):
        self.sigma = sigma
        self.knot = knot
        self.p = p
        self.wSize = wSize
        self.channels = channels
        self.keep = keep
        self.init_seed()

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be performed the TimeWarping.

        Returns:
            Signal or Tensor: Randomly TimeWarping signal.
        """
        if np.random.uniform(0, 1) < self.p:
            # print(signal.shape)
            signal_ = np.array(signal).copy()
            self.init_seed()
            if len(signal_.shape) == 1:
                signal_ = signal_[:, np.newaxis]

            # if signal.ndim != 2:
            #     signal_ = np.array(signal[0, :, :]).copy()
            # else:
            #     signal_ = np.array(signal).copy()

            signal_new = np.zeros((self.wSize, self.channels))
            x_range = np.arange(self.wSize)
            for i in range(self.channels):
                signal_new[:, i] = np.interp(x_range, self.tt_cum[:, 0], signal_[:, i])
            signal_new = signal_new[np.newaxis, :, :]
            return np.concatenate((signal_new, np.expand_dims(signal, 0)), 0) if self.keep else np.squeeze(signal_new)
        return signal

    def init_seed(self):
        #seed = int(time()) % 1000
        self.x = (np.ones((self.channels, 1)) * (np.arange(0, self.wSize, (self.wSize - 1) / (self.knot + 1)))).transpose()
        np.random.seed(42)
        self.y = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2, self.channels))
        self.x_range = np.arange(self.wSize)
        self.tt = np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range)
                            for i in range(self.channels)]).transpose()
        self.tt_cum = np.cumsum(self.tt, axis=0)
        # set the shape
        self.t_scale = [(self.wSize - 1) / self.tt_cum[-1, i]
                        for i in range(self.channels)]
        for i in range(self.channels):
            self.tt_cum[:, i] = self.tt_cum[:, i] * self.t_scale[i]


# 6. Random Sampling. (Using TimeWarp is more recommended)
class RandomSampling(object):
    """Perform the RandomSampling to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the TimeWarping. Default value is 0.5.
        sigma    (float) : sd of the scale value.
        knot     (int)   :                      .                     
        wSize    (int)   : Length of the input data.
        channels (int)   : Number of the input channels.
    """

    def __init__(self, p=0.5, nSample=300, wSize=500, channels=6):
        self.p = p
        self.wSize = wSize
        self.channels = channels
        self.tt = np.zeros((nSample, channels), dtype=int)
        for i in range(channels):
            self.tt[1:-1, i] = np.sort(np.random.randint(1, wSize - 1, nSample - 2))
        self.tt[-1, :] = wSize - 1

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be performed the RandomSampling.

        Returns:
            Signal or Tensor: Randomly RandomSampling signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            if self.channels == 1:
                signal_ = signal_[:, np.newaxis]
            new_signal = np.zeros((self.wSize, self.channels))
            for i in range(self.channels):
                new_signal[:, i] = np.interp(np.arange(self.wSize), self.tt[:, i], signal_[self.tt[:, i], i])
            if self.channels == 1:
                return np.squeeze(new_signal)
            return new_signal
        return signal


# 7. Random Cutout.
class RandomCutout(object):
    """Random cutout selected area of the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the random cutout. Default value is 0.5.
        area     (float) : size of fixed area.
        num      (int)   : number of the area.                     
        wSize    (int)   : Length of the input data.
        channels (int)   : Number of the input channels.
        default  (float) : replace value of the cutout area 
    """

    def __init__(self, p=0.5, area=25, num=4, wSize=500, channels=6, default=0.0, seed=SEED):
        self.p = p
        self.area = area
        self.num = num
        self.wSize = wSize
        self.channels = channels
        self.default = default
        self.seed = seed

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be performed the RandomSampling.

        Returns:
            Signal or Tensor: Randomly RandomSampling signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            mask = np.ones(self.wSize, np.float32)
            np.random.seed(self.seed)
            for _ in range(self.num):
                x = np.random.randint(self.wSize)

                x1 = np.clip(x - self.area // 2, 0, self.wSize)
                x2 = np.clip(x + self.area // 2, 0, self.wSize)
                # print(x1, x2)

                mask[x1:x2] = 0

            new_mask = np.zeros((self.wSize, self.channels))
            for i in range(self.channels):
                new_mask[:, i] = mask
            if self.channels == 1:
                new_mask = np.squeeze(new_mask)

            mask_b = 1 - new_mask
            signal_ = signal_ * new_mask + mask_b * self.default
            return signal_
        return signal


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


class GaussianNoise(object):
    """Jittering the input time-series data randomly with a given probability.

    Args:
        p        (float): probability of the sensor data being jittered. Default value is 0.5.
        var1     (float): sd of the noise1 (FMG).
        var2     (float): sd of the noise2 (EMG).
        wSize    (int)  : Length of the input data.
        channels (int)  : Number of the input channels.
    """

    def __init__(self, p=0.5, SNR=25, seed=SEED):
        self.p = p
        self.SNR = SNR
        self.seed = seed

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be jittered.

        Returns:
            Signal or Tensor: Randomly jittered signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            np.random.seed(self.seed)
            if len(signal_.shape) == 1:
                signal_ = signal_[:, np.newaxis]
                signal = signal[:, np.newaxis]
            for i in range(signal_.shape[1]):
                noise = np.random.randn(signal_.shape[0])
                noise = noise - np. mean(noise)
                signal_power = (1 / signal_.shape[0]) * np.sum(np.power(signal_[:, i], 2))
                noise_variance = signal_power / np.power(10, (self.SNR / 10))
                noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
                signal_[:, i] = signal_[:, i] + noise
            new_signal = np.stack((signal, signal_), 0)
            return signal_
        return signal

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(p={})'.format(self.p)
