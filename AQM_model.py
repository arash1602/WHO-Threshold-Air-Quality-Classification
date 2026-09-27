import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    """
    Removes the extra right-side padding so the convolution is causal.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, channels, kernel_size=3, dropout=0.2):
        """
        num_inputs: number of input channels
        channels: list like [32, 32, 64] for stacked TCN blocks
        """
        super().__init__()

        layers = []
        for i, out_channels in enumerate(channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        # Each TemporalBlock has 2 dilated convolutions at dilation 2^i.
        # RF = 1 + sum_i [ 2 * (kernel_size - 1) * 2^i ]
        self._rf = 1 + 2 * (kernel_size - 1) * ((2 ** len(channels)) - 1)

    @property
    def receptive_field(self):
        return self._rf

    def forward(self, x):
        return self.network(x)


class AQM_TCN_Model(nn.Module):
    def __init__(self, input_channels, tcn_channels=(32, 32, 32), kernel_size=3,
                 dropout=0.2, output_channels=2):
        super().__init__()

        if isinstance(tcn_channels, int):
            tcn_channels = (tcn_channels,)

        self.tcn = TemporalConvNet(
            num_inputs=input_channels,
            channels=list(tcn_channels),
            kernel_size=kernel_size,
            dropout=dropout,
        )

        print(
            f"AQM_TCN_Model | receptive field: {self.tcn.receptive_field} steps "
            f"| output_channels={output_channels}"
        )

        # output_channels=2 for regression (PM10, PM2.5); 1 for a single
        # classification logit.
        self.head = nn.Linear(tcn_channels[-1], output_channels)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)  # -> [batch, channels, seq_len]
        x = self.tcn(x)
        x = x[:, :, -1]         # last time step -> [batch, tcn_channels[-1]]
        return self.head(x)