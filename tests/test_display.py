import numpy as np
import pytest

from discopat.core import Box, Frame
from discopat.display import frame_to_pil, get_center_path, to_int


@pytest.mark.parametrize(
    ("image_array", "expected"),
    [
        pytest.param([0, 0, 1, 1], [0, 0, 255, 255]),
        pytest.param([0, 0, 0, 0], [0, 0, 0, 0]),
        pytest.param([0.0, 0.0, 1.0, 1.0], [0, 0, 255, 255]),
        pytest.param([-1, 1, -1, 1], [0, 255, 0, 255]),
    ],
)
def test_to_int(image_array, expected):
    assert np.allclose(to_int(np.array(image_array)), expected)


def test_get_center_path():
    track_ids = np.arange(10).reshape(-1, 1)
    xs = np.arange(10).reshape(-1, 1)
    ys = np.arange(0, 20, 2).reshape(-1, 1)

    track = np.hstack([track_ids, xs, ys, xs + 2, ys + 3])
    expected = np.hstack([track_ids, xs + 1, ys + 1.5])

    assert np.allclose(get_center_path(track), expected)


def test_frame_to_pil():
    frame = Frame(
        name="100",
        width=10,
        height=10,
        annotations=[],
        image_array=np.zeros((10, 10)),
    )
    tracks = {
        0: np.hstack([np.arange(10).reshape(-1, 1), np.zeros((10, 4))]),
        1: np.hstack(
            [
                np.arange(200).reshape(-1, 1),
                np.ones((200, 1)),
                np.ones((200, 1)),
                np.ones((200, 1)),
                np.ones((200, 1)),
            ]
        ),
    }

    pil_image = frame_to_pil(frame, tracks)

    assert np.allclose(np.array(pil_image), np.ones((10, 10, 3)))
