# cnn_ben_3d.py
# PyTorch reimplementation of the 3D CNN architecture from the TensorFlow script.
# Copied unchanged from 3D_CNN/3D_supervised_HC_vs_TLE/models/cnn_ben_3d.py
# Input shape: (N, 1, D, H, W)
# Conv3D -> BN -> (optional MaxPool3D) -> ReLU x 6 -> GAP -> Dropout -> FC(128) -> FC(num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MRIClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.2, filters: list = None):
        """
        PyTorch version of the 3D CNN:

        FILTERS = [64, 128, 256, 256, 256, 256] (default)
        KERNELS = [3,   3,   3,   3,   3,   3]  (all 3s)
        STRIDES = [2,   1,   1,   1,   1,   1]  (first is 2, rest are 1)

        MaxPool3D is applied after every block except the last one.
        """
        super().__init__()

        if filters is None:
            filters = [64, 128, 256, 256, 256, 256]

        # Generate kernels and strides based on filters length
        kernels = [3] * len(filters)
        strides = [2] + [1] * (len(filters) - 1)

        in_channels = 1
        layers = []

        for i, (out_channels, k, s) in enumerate(zip(filters, kernels, strides)):
            # Conv3D with 'same' padding in TF -> padding = k // 2 here
            layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm3d(out_channels))

            # MaxPooling3D after every block except the last
            if i < len(filters) - 1:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # GlobalAveragePooling3D -> AdaptiveAvgPool3d(1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(filters[-1], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1, D, H, W)
        returns logits of shape (N, num_classes)
        """
        x = self.features(x)                  # (N, C, D, H, W)
        x = self.global_pool(x)               # (N, C, 1, 1, 1)
        x = torch.flatten(x, 1)               # (N, C)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                       # logits (no softmax)
        return x


def build_model(num_classes: int = 3, dropout_rate: float = 0.2) -> MRIClassifier:
    return MRIClassifier(num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == "__main__":
    model = build_model(num_classes=1)
    dummy = torch.randn(2, 1, 115, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape)  # should be (2, 1)
