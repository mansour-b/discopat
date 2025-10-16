import numpy as np


def to_int(array: np.array) -> np.array:
    vmin = array.min()
    vmax = array.max()
    return ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
