import torch
import torch.nn as nn
import torch.nn.functional as F


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
