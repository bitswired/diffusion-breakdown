import torch
from diffusion_breakdown.noise import Noise, LinearVarianceSchedule
import matplotlib.pyplot as plt


class TestNoise:
    def test_forward_diffusion(self):
        x = torch.stack([torch.linspace(-1, 1, 200)] * 2)
        timesteps = 100
        variance_schedule = LinearVarianceSchedule(1e-4, 2e-2, timesteps)
        noise = Noise(timesteps, variance_schedule)
        y = noise.forward_diffusion(x, torch.arange(timesteps).long()).numpy()

        assert y.shape == (timesteps, 2, 200)
