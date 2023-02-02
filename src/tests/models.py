import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from diffusion_breakdown.models import Unet1D


def get_sine_wave_dataset(samples, dim, freq_range, n_freqs, n_freqs_sum):
    x = np.linspace(-np.pi, np.pi, dim)
    freq = np.random.uniform(*freq_range, (samples, n_freqs))
    cos = np.cos(freq.reshape(-1, 1) * x)
    sin = np.sin(freq.reshape(-1, 1) * x)

    waves = np.vstack([cos, sin])

    Y = np.random.randint(0, len(waves), (samples, n_freqs_sum))
    Y = np.expand_dims(waves[Y].sum(axis=1) / n_freqs_sum, 1)

    X = Y + np.random.normal(0, 0.1, Y.shape)

    print()
    print(X.shape)
    

    # fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    # for i in range(9):
    #     row = i // 3
    #     col = i % 3
    #     axs[row][col].plot(Y[i][0,:], Y[i][1,:])
    #     axs[row][col].scatter(X[i][0,:], X[i][1,:], c='r', s=1, alpha=0.5)
    # plt.show()

    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    return dataset


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Unet1D(
            in_channels=1,
            out_channels=1,
            start_filters=64,
            steps=3,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Unet1D(
            in_channels=1,
            out_channels=1,
            start_filters=16,
            steps=3,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return opt
        