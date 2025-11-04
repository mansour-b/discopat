import numpy as np
import pytest

from discopat.display import to_int


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
