from unittest.mock import patch

import pytest

from discopat.utils import get_device


@pytest.mark.parametrize(
    ("cuda", "mps", "allow_mps", "expected"),
    [
        (True, False, True, "cuda"),
        (False, True, False, "cpu"),
        (False, True, True, "mps"),
        (False, False, True, "cpu"),
    ],
)
def test_get_device(cuda, mps, allow_mps, expected):
    import torch

    with (
        patch.object(torch.cuda, "is_available", return_value=cuda),
        patch.object(torch.mps, "is_available", return_value=mps),
    ):
        assert get_device(allow_mps=allow_mps) == expected
