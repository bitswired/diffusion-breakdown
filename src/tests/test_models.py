import numpy as np
import pytest
import torch
from diffusion_breakdown.models import Unet1D
from tests.models import Model, get_sine_wave_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor



class TestUnet1D:
    def test_forward(self):
        unet = Unet1D(
            in_channels=1,
            out_channels=1,
            start_filters=16,
            steps=3,
            kernel_size=1,
            stride=1,
            padding='same',
        )
        x = torch.rand(10, 1, 128)
        y = unet(x)
        assert y.shape == (10, 1, 128)

    @pytest.mark.slow
    def test_2(self):
        model = Model()
        x = model(torch.ones(10, 1, 128))

        train_dataset = get_sine_wave_dataset(1000, 128, (0, 4), 10,  5)
        val_dataset = get_sine_wave_dataset(128, 128, (0, 4), 10,  5)
        dl_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
        dl_val = DataLoader(val_dataset, batch_size=128, shuffle=True)

        

        trainer = pl.Trainer(callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
            LearningRateMonitor(logging_interval='step')
            ], gradient_clip_val=0.5)
        # trainer = pl.Trainer(overfit_batches=1)

        trainer.fit(model, dl_train, dl_val)

        with torch.no_grad():
            X, Y = val_dataset[:]
            y = Y.numpy()
            x = X.numpy()
            xx = np.arange(0, Y.shape[-1])
            print(X.shape)
            res = model(X).numpy()
            fig, axs = plt.subplots(3, 3, figsize=(10, 10))
            for i in range(9):
                row = i // 3
                col = i % 3
                axs[row][col].plot(xx, y[i][0])
                axs[row][col].plot(xx, res[i][0], c='r', alpha=0.5)
                axs[row][col].scatter(xx, x[i][0], c='g', s=1, alpha=0.5)
            plt.show()

        




        


        

        
