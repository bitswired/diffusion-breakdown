from functools import partial

import numpy as np
import torchdata.datapipes as dp
from sklearn.datasets import make_moons


def get_rotation_matrix(deg: float):
    """Returns a 2D rotation matrix for a given angle in degrees."""

    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


def rotate(x: np.ndarray, deg: float):
    """Rotates a 2D array by a given angle in degrees."""

    return x @ get_rotation_matrix(deg)


def get_moons_data_pipe(size: int):
    """Returns a data pipe of size `size` with 2D moon data."""

    data, _ = make_moons(size, noise=0)
    data_pipe = dp.iter.Cycler([data]).map(
        partial(rotate, deg=np.random.randint(360)),
    )
    return data_pipe
