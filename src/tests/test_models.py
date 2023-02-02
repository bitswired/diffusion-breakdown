import torch
from diffusion_breakdown.models import Unet1D


class TestUnet1D:
    def test_forward(self):
        unet = Unet1D(
            in_channels=1,
            out_channels=1,
            start_filters=16,
            steps=3,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        x = torch.rand(10, 1, 128)
        y = unet(x)
        print(y.shape)
        assert y.shape == (10, 1, 100)
