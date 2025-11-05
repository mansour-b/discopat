import hashlib

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
    frame_id = 100
    frame = Frame(
        name=str(frame_id),
        width=100,
        height=100,
        annotations=[],
        image_array=np.zeros((100, 100)),
    )
    tracks = {
        0: np.hstack([np.arange(10).reshape(-1, 1), np.zeros((10, 4))]),
        1: np.hstack(
            [
                np.arange(200).reshape(-1, 1),
                np.linspace(10, 80, 200).reshape(-1, 1),
                np.linspace(5, 50, 200).reshape(-1, 1),
                np.linspace(20, 85, 200).reshape(-1, 1),
                np.linspace(20, 70, 200).reshape(-1, 1),
            ]
        ),
    }
    x = tracks[1][frame_id, 1]
    y = tracks[1][frame_id, 2]
    width = tracks[1][frame_id, 3] - x
    height = tracks[1][frame_id, 4] - y
    frame.annotations = [
        Box(label="dummy", x=x, y=y, width=width, height=height, score=1.0)
    ]

    pil_image = frame_to_pil(frame, tracks)
    expected_hash = (
        "35e1e20a41d69c1f5724d5ef1c1e8339236c31d30818042d87330a2aefc39613"
    )
    actual_hash = hashlib.sha256(np.array(pil_image)).hexdigest()
    assert actual_hash == expected_hash
