import numpy as np
import pytest

from discopat.metrics import compute_ap, compute_iomean, compute_iou


class TestMetrics:
    @pytest.mark.parametrize(
        ("box1", "box2", "expected"),
        [
            pytest.param([0, 0, 1, 1], [0, 0, 1, 1], 1.0),
            pytest.param([0, 0, 1, 1], [0.5, 0, 1.5, 1], 0.5),
            pytest.param([0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5], 0.25),
            pytest.param([0, 0, 1, 1], [0, 0.5, 1, 1.5], 0.5),
            pytest.param([0, 0, 1, 1], [-0.5, 0.5, 0.5, 1.5], 0.25),
            pytest.param([0, 0, 1, 1], [-0.5, 0, 0.5, 1], 0.5),
            pytest.param([0, 0, 1, 1], [-0.5, -0.5, 0.5, 0.5], 0.25),
            pytest.param([0, 0, 1, 1], [0, -0.5, 1, 0.5], 0.5),
            pytest.param([0, 0, 1, 1], [0.5, -0.5, 1.5, 0.5], 0.25),
        ],
    )
    def test_compute_iomean(self, box1, box2, expected):
        assert np.isclose(compute_iomean(box1, box2), expected)

    @pytest.mark.parametrize(
        ("box1", "box2", "expected"),
        [
            pytest.param([0, 0, 1, 1], [0, 0, 1, 1], 1.0),
            pytest.param([0, 0, 1, 1], [0.5, 0, 1.5, 1], 1 / 3),
            pytest.param([0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5], 1 / 7),
            pytest.param([0, 0, 1, 1], [0, 0.5, 1, 1.5], 1 / 3),
            pytest.param([0, 0, 1, 1], [-0.5, 0.5, 0.5, 1.5], 1 / 7),
            pytest.param([0, 0, 1, 1], [-0.5, 0, 0.5, 1], 1 / 3),
            pytest.param([0, 0, 1, 1], [-0.5, -0.5, 0.5, 0.5], 1 / 7),
            pytest.param([0, 0, 1, 1], [0, -0.5, 1, 0.5], 1 / 3),
            pytest.param([0, 0, 1, 1], [0.5, -0.5, 1.5, 0.5], 1 / 7),
        ],
    )
    def test_compute_iou(self, box1, box2, expected):
        assert np.isclose(compute_iou(box1, box2), expected)

    @pytest.mark.parametrize(
        ("groundtruths", "predictions", "expected"),
        [
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1, 0.9]],
                1.0,
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 0.5, 1, 0.9]],
                1.0,
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 0.4, 1, 0.9]],
                0.0,
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
                5 / 6,
            ),
        ],
    )
    def test_compute_ap50(self, groundtruths, predictions, expected):
        ap50 = compute_ap(
            groundtruths,
            predictions,
            threshold=0.5,
            localization_criterion="iou",
        )
        assert np.isclose(ap50, expected)

    @pytest.mark.parametrize(
        ("threshold", "expected"),
        [
            pytest.param(0.5, 1.0),
            pytest.param(0.6, 5 / 6),
            pytest.param(0.7, 2 / 3),
            pytest.param(0.8, 1 / 2),
            pytest.param(0.9, 1 / 3),
            pytest.param(0.95, 1 / 6),
        ],
    )
    def test_compute_ap_multiple_iou_thresholds(self, threshold, expected):
        groundtruths = [[i, i, i + 1, i + 1] for i in range(0, 12, 2)]
        predictions = [
            [i, i, i + 1 - 0.049 * i, i + 1, 0.9] for i in range(0, 12, 2)
        ]
        ap = compute_ap(
            groundtruths,
            predictions,
            threshold=threshold,
            localization_criterion="iou",
        )
        assert np.isclose(ap, expected)
