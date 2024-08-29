import torch
import os
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# class OnlineDataset(Dataset):
#
#     def __init__(self, paths=None, transforms=None) -> None:
#         """
#         """
#         super().__init__()
#         self.paths = paths
#         self.transform = transforms
#
#     def __getitem__(self, index):
#         filePath = self.paths[index]
#         saved_data = loadmat(filePath)
#         data = saved_data['data'].transpose(1, 0)
#         label = int(filePath.split(os.sep)[-1].split('_')[-1].split('.')[0])
#
#         if self.transform is not None:
#             data = self.transform(data)
#
#         if data.ndim == 3:
#             data = data.transpose(0, 2, 1)
#             label = np.stack((label, label), 0)
#         else:
#             data = data.transpose(1, 0)
#
#         return data, label
#
#     def __len__(self):
#         return len(self.paths)


class OnlineDataset(Dataset):
    def __init__(self, data_source, transforms=None):
        self.data = []
        self.labels = []
        self.label_mapping = {}
        self.transform = transforms

        for label, windows in data_source.items():
            if label not in self.label_mapping:
                self.label_mapping[label] = len(self.label_mapping)
            for window in windows:
                self.data.append(window)
                self.labels.append(self.label_mapping[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])[:, np.newaxis]
        label = self.labels[idx]

        if self.transform is not None:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32).transpose(1, 0), label


def makedataset(serial, labels, data_length, window_size, step_size):
    # Data collection
    print("Starting data collection...")

    all_data = []
    all_labels = []

    train_data = {}
    test_data = {}

    for _ in range(2):
        for label in labels:
            input(f"Press enter to start collecting data for {label} after ready...")
            serial.flushInput()
            serial.readline()

            data_buffer = []
            while len(data_buffer) < data_length:
                data_point = serial.readline().decode('utf-8').strip()
                data_buffer.append(float(data_point))

            print(f"Data collection for {label} is complete!")

            windows = []
            for start_idx in range(0, len(data_buffer) - window_size + 1, step_size):
                window = data_buffer[start_idx:start_idx + window_size]
                windows.append(window)

            all_data.extend(windows)
            all_labels.extend([label] * len(windows))

    train_data_raw, test_data_raw, train_labels_raw, test_labels_raw = train_test_split(
        all_data, all_labels, test_size=0.8, random_state=42, stratify=all_labels)

    for label in labels:
        train_data[label] = [train_data_raw[i] for i, lbl in enumerate(train_labels_raw) if lbl == label]
        test_data[label] = [test_data_raw[i] for i, lbl in enumerate(test_labels_raw) if lbl == label]

    print("All data collection is complete!")
    return train_data, test_data