import numpy as np
import pytest

from discopat.metrics import compute_ap_at_threshold


class TestMetrics:
    @pytest.mark.parametrize(
        ("groundtruths", "predictions", "expected"),
        [
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1, 0.9]],
                1.0,
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[0, 0, 1, 1, 0.9], [2, 2, 3, 3, 0.8]],
                1.0,
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[0, 0, 1, 1, 0.9], [2, 2, 3, 3, 0.8], [4, 4, 5, 5, 0.7]],
                1.0,
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[0, 0, 1, 1, 0.9], [2, 2, 3, 3, 0.6], [4, 4, 5, 5, 0.7]],
                1.0,
            ),
        ],
    )
    def test_compute_ap_at_threshold(self, groundtruths, predictions, expected):
        ap50 = compute_ap_at_threshold(groundtruths, predictions, threshold=0.5)
        assert np.isclose(ap50, expected)
