import torch
import torch.nn as nn


class DownScaleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class UpscaleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding=0,
            bias=True,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Unet1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        start_filters,
        steps,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()

        self.down_blocks = nn.ModuleList()
        for i in range(steps):
            in_c = in_channels if i == 0 else start_filters * 2 ** (i - 1)
            out_c = start_filters * 2**i
            print(in_c, out_c)
            self.down_blocks.append(
                DownScaleBlock1D(in_c, out_c, kernel_size, stride, padding)
            )
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(steps)):
            in_c = start_filters * 2**i
            out_c = out_channels if i == 0 else start_filters * 2 ** (i - 1)
            print(in_c, out_c)
            self.up_blocks.append(UpscaleBlock1D(in_c, out_c, kernel_size, 2, padding))

    def forward(self, x):
        down_outputs = []
        print()
        for block in self.down_blocks:
            x = block(x)
            print("down", x.shape)
            print()
            down_outputs.append(x)

        for i, block in enumerate(self.up_blocks):
            print(x.shape)
            x = block(x)
            print("up", x.shape)
            print(down_outputs[-i - 2].shape)
            x = torch.cat([x, down_outputs[-i - 2]], dim=1)

        return x
