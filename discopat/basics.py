import numpy as np


def to_int(array: np.ndarray) -> np.ndarray:
    vmin = array.min()
    vmax = array.max()
    return ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def to_01(array: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vmin = array.min()
    vmax = array.max()
    return (array - vmin) / max((vmax - vmin), eps)


def gs_to_rgb(
    array: np.ndarray, channel_mode: str = "channels_last"
) -> np.ndarray:
    if len(array.shape) != 2:
        msg = f"Grayscale image should be of size (H, W). Actual shape: {array.shape}"
        raise ValueError(msg)
    channel_axis = {"channels_first": 0, "channels_last": -1}[channel_mode]
    return np.repeat(
        np.expand_dims(array, axis=channel_axis), repeats=3, axis=channel_axis
    )
