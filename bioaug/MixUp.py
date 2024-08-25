import numpy as np
import matplotlib.pyplot as plt


class MixUp(object):
    """Apply MixUp augmentation to the input data.

    Args:
        alpha (float): The parameter for the Beta distribution from which the mixing factor is sampled.
        p     (float): Probability of applying MixUp to the input batch.
        seed  (int)  : A seed value for the random number generator to ensure reproducibility.
    """
    def __init__(self, p=0.5, alpha=0.2, seed=42):
        self.p = p
        self.alpha = alpha
        self.seed = seed

    def __call__(self, batch):
            """Apply MixUp augmentation to the input batch.
            
            Args:
                batch (tuple): A tuple containing a batch of inputs and labels (inputs, labels).
            
            Returns:
                tuple: Mixed inputs and mixed labels, or the original inputs and labels if MixUp is not applied.
            """
            inputs, labels = batch
            if np.random.rand() < self.p:
                np.random.seed(self.seed)
                lam = np.random.beta(self.alpha, self.alpha)
                
                batch_size = inputs.shape[0]
                indices = np.random.permutation(batch_size)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[indices, :]
                labels_a, labels_b = labels, labels[indices]
                mixed_labels = lam * labels_a + (1 - lam) * labels_b
                
                return mixed_inputs, mixed_labels
            else:
                return inputs, labels


# Example usage
# if __name__ == '__main__':
#     data = np.random.normal(loc=1, scale=1, size=(4, 500, 6))
#     label = np.array([
#         [0, 0, 0, 1],
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0]
#     ])

#     batch = (data, label)
#     gn = MixUp(p=1.0, alpha=0.2)
#     aug_data_batch = gn(batch)

#     raw_fig = plt.figure(figsize=(5, 5))
#     data = data[0]
#     for plt_index in range(1, 7):
#         ax = raw_fig.add_subplot(3, 2, plt_index)
#         ax.plot(list(range(500)), data[:, plt_index-1])

#     aug_fig = plt.figure(figsize=(5, 5))
#     aug_data_1 = aug_data_batch[0][0]

#     for plt_index in range(1, 7):
#         ax = aug_fig.add_subplot(3, 2, plt_index)
#         ax.plot(list(range(500)), aug_data_1[:, plt_index-1], color='r')
#     plt.show()