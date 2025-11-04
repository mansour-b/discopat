import numpy as np
import pytest

from discopat.display import get_center_path, to_int


@pytest.mark.parametrize(
    ("image_array", "expected"),
    [
        pytest.param([0, 0, 1, 1], [0, 0, 255, 255]),
        pytest.param([0.0, 0.0, 1.0, 1.0], [0, 0, 255, 255]),
        pytest.param([-1, 1, -1, 1], [0, 255, 0, 255]),
    ],
)
def test_to_int(image_array, expected):
    assert np.allclose(to_int(np.array(image_array)), expected)


def test_get_center_path():
    track_ids = np.zeros((10, 1))
    xs = np.arange(10).reshape(-1, 1)
    ys = np.arange(0, 20, 2).reshape(-1, 1)

    track = np.hstack([track_ids, xs, ys, xs + 2, ys + 3])
    expected = np.hstack([track_ids, xs + 1, ys + 1.5])

    assert np.allclose(get_center_path(track), expected)
