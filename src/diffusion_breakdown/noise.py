from dataclasses import dataclass

import torch


@dataclass
class VarianceSchedule:
    """Abstract class for variance schedule."""

    def __call__(self):
        raise NotImplementedError()


class LinearVarianceSchedule(VarianceSchedule):
    """Linearly interpolate between start_variance and end_variance over timesteps."""

    def __init__(self, start_variance: float, end_variance: float, timesteps: int):
        self.start_variance = start_variance
        self.end_variance = end_variance
        self.timesteps = timesteps

    def __call__(self, t: torch.Tensor):
        return (
            self.start_variance
            + (self.end_variance - self.start_variance) * t / self.timesteps
        )


@dataclass
class Noise:
    """Class to handle noise operations for the diffusion process."""

    timesteps: int
    variance_schedule: VarianceSchedule

    def __post_init__(self):
        t = torch.arange(self.timesteps)
        self.betas = self.variance_schedule(t)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        x = x.view(-1)
        # Compute mean and variance
        mean = torch.sqrt(self.alphas_bar[t]).unsqueeze(-1) * x
        var = 1 - self.alphas_bar[t]
        # Sample noise, scale to variance and add to mean (sampling re-paramerization trick)
        res = mean + var.unsqueeze(-1) * torch.randn_like(mean)
        # Reshape to original shape with time dimension and return
        return res.view(*t.shape, *sh)
