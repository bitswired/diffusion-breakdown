import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DownScaleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class UpscaleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock1D(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
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
            self.down_blocks.append(
                DownScaleBlock1D(in_c, out_c, kernel_size, stride, padding)
            )
        # Middle block
        self.middle_block = ConvBlock1D(
            start_filters * 2 ** (steps - 1),
            start_filters * 2 ** (steps - 1),
            kernel_size,
            stride,
            padding,
        )
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(steps)):
            in_c = start_filters * 2**(i + 1)
            out_c = out_channels if i == 0 else start_filters * 2 ** (i - 1)
            print(in_c, out_c)
            self.up_blocks.append(UpscaleBlock1D(in_c, out_c, kernel_size, stride, padding))

        # Final block
        self.final_block = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        
        print(self.down_blocks)
        print(self.middle_block)
        print(self.up_blocks)

    def forward(self, x):
        down_outputs = []
        print()
        for i, block in enumerate(self.down_blocks):
            if i == 0:
                x = block(x)
                down_outputs.append(x)
            else:
                x = block(x[1])
                down_outputs.append(x)
            
            

        x = self.middle_block(x[1])

        for i, block in enumerate(self.up_blocks):
            x = block(x, down_outputs[-i-1][0])

        x = self.final_block(x)

        return x
