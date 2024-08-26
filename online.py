import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def real_time_test(model, num_points=200):
    """
    Real-time testing function. Reads data from the serial port, preprocesses it, and predicts using the model.
    :param model: The trained model
    :param num_points: Number of data points used for prediction, default is 200
    """
    label_mapping = {
        0: "リラックス",
        1: "親指",
        2: "人差し指",
        3: "中指",
        4: "薬指",
        5: "小指",
        6: "g6"
    }
    # Set the model to evaluation mode
    model.eval()

    try:
        while True:
            # Read data
            data_buffer = []
            ser.flushInput()  # Clear serial port buffer
            ser.readline()  # Discard the first data point

            while len(data_buffer) < num_points:
                data_point = ser.readline().decode('utf-8').strip()  # Read a data point
                if data_point:
                    data_buffer.append(float(data_point))  # Add data point to buffer

            # Data preprocessing
            data_array = np.array(data_buffer).reshape(1, -1)  # Convert to numpy array and reshape
            data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device).view(1, 1,
                                                                                        -1)  # Convert to tensor and reshape

            # Make prediction using the model
            with torch.no_grad():
                outputs = model(data_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()

            # Print prediction result
            print(f"Predicted class: {label_mapping[predicted_class]}")

    except KeyboardInterrupt:
        print("Real-time testing stopped.")


class EarlyStopping:
    """Early stopping class for training"""

    def __init__(self, patience=5, min_delta=0, triggered_accuracy=0.9):
        """
        :param patience: Number of consecutive epochs with no improvement after which training will be stopped
        :param min_delta: Minimum change to qualify as an improvement
        :param triggered_accuracy: Required accuracy to trigger early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False
        self.triggered_accuracy = triggered_accuracy

    def step(self, val_loss, val_accuracy):
        if val_accuracy < self.triggered_accuracy:
            return
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# Data loader
class EMGDataset(Dataset):
    def __init__(self, data_source, split='train'):
        assert split in ['train', 'val', 'test'], "Invalid split value. It should be 'train', 'val', or 'test'."

        self.data = []
        self.labels = []
        self.label_mapping = {}

        for label, windows in data_source.items():
            if label not in self.label_mapping:
                self.label_mapping[label] = len(self.label_mapping)

            for window in windows:
                self.data.append(window)
                self.labels.append(self.label_mapping[label])

        print("All labels found:", self.label_mapping.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32).view(1, -1), label  # reshape


# Residual Shrinkage Block
class ResidualShrinkageBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualShrinkageBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.alpha = nn.Parameter(torch.ones(size=(1,), dtype=torch.float32) * 0.01)

        # 1x1 convolution for resizing
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)
        out = out - self.alpha * identity

        out += identity
        out = F.relu(out)

        return out


# Residual Shrinkage Network
class ResidualShrinkageNet(nn.Module):
    def __init__(self, num_classes):
        super(ResidualShrinkageNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.res_block1 = ResidualShrinkageBlock(16, 16)
        self.res_block2 = ResidualShrinkageBlock(16, 32, stride=2)

        self.fc1 = nn.Linear(32 * (200 // 2), 128)  # Assuming input is downsampled by 2 in depth
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Parameter settings
sample_rate = 500
data_length = 4 * sample_rate
window_size = 200
step_size = 1
validation_ratio = 0.2
num_classes = int(input("Enter number of classes: "))
labels = [f"label{i + 1}" for i in range(num_classes)]

ser = serial.Serial('COM3', 115200)

train_data = {}
validation_data = {}
test_data = {}

# Data collection
print("Starting data collection...")
all_data = []
all_labels = []

for _ in range(2):
    for label in labels:
        input(f"Press enter to start collecting data for {label} after ready...")
        ser.flushInput()
        ser.readline()

        data_buffer = []
        while len(data_buffer) < data_length:
            data_point = ser.readline().decode('utf-8').strip()
            data_buffer.append(float(data_point))

        print(f"Data collection for {label} is complete!")

        windows = []
        for start_idx in range(0, len(data_buffer) - window_size + 1, step_size):
            window = data_buffer[start_idx:start_idx + window_size]
            windows.append(window)

        all_data.extend(windows)
        all_labels.extend([label] * len(windows))

print("All data collection is complete!")

train_data_raw, temp_data_raw, train_labels_raw, temp_labels_raw = train_test_split(
    all_data, all_labels, test_size=0.6, random_state=42, stratify=all_labels)

validation_data_raw, test_data_raw, validation_labels_raw, test_labels_raw = train_test_split(
    temp_data_raw, temp_labels_raw, test_size=0.5, random_state=42, stratify=temp_labels_raw)

for label in labels:
    train_data[label] = [train_data_raw[i] for i, lbl in enumerate(train_labels_raw) if lbl == label]
    validation_data[label] = [validation_data_raw[i] for i, lbl in enumerate(validation_labels_raw) if lbl == label]
    test_data[label] = [test_data_raw[i] for i, lbl in enumerate(test_labels_raw) if lbl == label]


def train_model():
    batch_size = 128
    num_epochs = 200
    learning_rate = 0.00003

    train_loader = DataLoader(EMGDataset(data_source=train_data, split='train'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EMGDataset(data_source=validation_data, split='val'), batch_size=batch_size)
    test_loader = DataLoader(EMGDataset(data_source=test_data, split='test'), batch_size=batch_size)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, triggered_accuracy=0.9)

    model = ResidualShrinkageNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy}, F1 Score: {f1}")

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}, F1 Score: {f1}")
        early_stopping.step(val_loss / len(val_loader), accuracy)
        if early_stopping.stop:
            print("Early stopping triggered!")
            break

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Test Loss: {test_loss / len(test_loader)}, Accuracy: {accuracy}, F1 Score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()
    return model


model_trained = train_model()
torch.save(model_trained.state_dict(), 'testmodel.pt')

train_dataset = EMGDataset(data_source=train_data, split='train')
train_loader = DataLoader(train_dataset, batch_size=32)

val_dataset = EMGDataset(data_source=validation_data, split='val')
val_loader = DataLoader(val_dataset, batch_size=32)

test_dataset = EMGDataset(data_source=test_data, split='test')
test_loader = DataLoader(test_dataset, batch_size=32)

torch.save({'data': all_data, 'labels': all_labels}, 'all_data_labels.pt')

loaded_data = torch.load('all_data_labels.pt')
data = loaded_data['data']
labels = loaded_data['labels']

for inputs, _ in val_loader:
    np.save('representative_features.npy', inputs.numpy())
    break

input_sample = torch.randn(1, 1, 200).to(device)
with torch.no_grad():
    torch.onnx.export(model_trained, input_sample, "model_trained.onnx", verbose=False)
real_time_test(model_trained)

