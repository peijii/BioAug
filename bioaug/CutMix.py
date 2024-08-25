import numpy as np
import matplotlib.pyplot as plt

class CutMix(object):
    """Apply CutMix augmentation to a batch of time-series data.

    Args:
        alpha (float): The parameter for the Beta distribution from which the mixing factor is sampled.
        p     (float): Probability of applying CutMix to the input batch.
        seed  (int)  : A seed value for the random number generator to ensure reproducibility.
    """

    def __init__(self, alpha=1.0, p=0.5, seed=42):
        self.p = p
        self.alpha = alpha
        self.seed = seed

    def __call__(self, batch):
        """Apply CutMix augmentation to the input batch.

        Args:
            batch (tuple): A tuple containing a batch of inputs and labels (inputs, labels).

        Returns:
            tuple: Mixed inputs and mixed labels, or the original inputs and labels if CutMix is not applied.
        """
        x, y = batch
        if np.random.rand() < self.p:
            np.random.seed(self.seed)
            lam = np.random.beta(self.alpha, self.alpha)

            batch_size, seq_len, _ = x.shape
            cut_len = int(seq_len * (1 - lam))

            # Ensure cut_len is not zero to perform a meaningful cut
            if cut_len > 0:
                # Randomly choose the start point for the cut
                cut_start = np.random.randint(0, seq_len - cut_len + 1)

                # Generate a random index for mixing
                indices = np.random.permutation(batch_size)
                x_b = x[indices].copy()
                y_b = y[indices].copy()

                # Debugging: Print out important values
                print(f"cut_len: {cut_len}, cut_start: {cut_start}, lam: {lam}")

                # Replace the region in a copied version of the original sequence with the region from another sequence
                x_mix = x.copy()
                x_mix[:, cut_start:cut_start + cut_len, :] = x_b[:, cut_start:cut_start + cut_len, :]

                # Adjust lambda based on the length of the cut
                lam = 1 - (cut_len / seq_len)
                y_mix = lam * y + (1 - lam) * y_b

                return x_mix, y_mix
        return x, y


# # Example usage
# if __name__ == '__main__':
#     # Example data: batch of 4 sequences, each of length 500 with 6 channels
#     data = np.random.normal(loc=1, scale=1, size=(4, 500, 6))
#     label = np.array([
#         [0, 0, 0, 1],
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0]
#     ])

#     # Initialize CutMix with p=1.0 to always apply CutMix
#     cutmix = CutMix(alpha=0.5, p=1.0)
#     # Simulate a batch of data
#     batch = (data, label)
#     # Apply CutMix augmentation
#     x_mix, y_mix = cutmix(batch)

#     # Plot the original data
#     raw_fig = plt.figure(figsize=(5, 5))
#     original_data = data[1]  # Using the second sequence in the batch
#     for plt_index in range(1, 7):
#         ax = raw_fig.add_subplot(3, 2, plt_index)
#         ax.plot(list(range(500)), original_data[:, plt_index - 1])
#     plt.suptitle("Original Data")

#     # Plot the augmented data
#     aug_fig = plt.figure(figsize=(5, 5))
#     aug_data_1 = x_mix[1]  # Using the second sequence in the batch after augmentation
#     for plt_index in range(1, 7):
#         ax = aug_fig.add_subplot(3, 2, plt_index)
#         ax.plot(list(range(500)), aug_data_1[:, plt_index - 1], color='r')
#     plt.suptitle("Augmented Data (CutMix)")

#     plt.show()