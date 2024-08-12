import numpy as np
import matplotlib.pyplot as plt


class Permutation(object):
    """Permutate the input time-series data randomly with a given probability.
    
    Args:
        p             (float) : Probability of applying permutation to the input signal.
        nPerm         (int)   : The number of segments into which the signal will be divide. The signal is split into 'nPerm'
                                segments, which are then randomly recorded. 
        minSegLength  (int)   : The minimum length of each segment. 
    """

    def __init__(self, p=0.5, nPerm=4, minSegLength=10):
        self.p = p
        self.nPerm = nPerm
        self.minSegLength = minSegLength

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        sequence_length = signal.shape[0]
        input_dim = signal.shape[1]
        if np.random.uniform(0, 1) < self.p:
            if input_dim == 1:
                signal_ = np.squeeze(np.zeros((sequence_length, input_dim)))
            else:
                signal_ = np.zeros((sequence_length, input_dim))
            idx = np.random.permutation(self.nPerm)
            flag = True
            while flag == True:
                segs = np.zeros(self.nPerm + 1, dtype=int)
                segs[1:-1] = np.sort(
                    np.random.randint(self.minSegLength, sequence_length - self.minSegLength, self.nPerm - 1))
                segs[-1] = sequence_length
                if np.min(segs[1:] - segs[0:-1]) > self.minSegLength:
                    flag = False

            pp = 0
            for i in range(self.nPerm):
                if input_dim == 1:
                    signal_temp = signal[segs[idx[i]]:segs[idx[i] + 1]]
                    signal_[pp:pp + len(signal_temp)] = signal_temp
                else:
                    signal_temp = signal[segs[idx[i]]:segs[idx[i] + 1], :]
                    signal_[pp:pp + len(signal_temp), :] = signal_temp
                pp += len(signal_temp)
            return signal_
        return signal


if __name__ == '__main__':
    data = np.random.normal(loc=1, scale=1, size=(500, 6))
    pe = Permutation(p=1.0, nPerm=2, minSegLength=10)
    aug_data = pe(data)

    raw_fig = plt.figure(figsize=(10, 10))
    for plt_index in range(1, 7):
        ax = raw_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), data[:, plt_index-1])

    aug_fig = plt.figure(figsize=(10, 10))
    for plt_index in range(1, 7):
        ax = aug_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), aug_data[:, plt_index-1], color='r')
    plt.show()