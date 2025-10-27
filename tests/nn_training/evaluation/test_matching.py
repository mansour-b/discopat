import numpy as np
import pytest

from discopat.nn_training.evaluation.matching import (
    match_groundtruths_and_predictions,
)


class TestMatching:
    @pytest.mark.parametrize(
        (
            "groundtruths",
            "predictions",
            "threshold",
            "localization_criterion",
            "expected",
        ),
        [
            pytest.param(
                [[0, 0, 1, 1]], [[0, 0, 1, 1, 0.9]], 0.5, "iou", (1, (0.9, 1))
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0, 1.5, 1, 0.9]],
                0.5,
                "iou",
                (1, (0.9, 0)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0.5, 1.5, 1.5, 0.9]],
                0.5,
                "iou",
                (1, (0.9, 0)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1, 0.9]],
                0.5,
                "iomean",
                (1, (0.9, 1)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0, 1.5, 1, 0.9]],
                0.5,
                "iomean",
                (1, (0.9, 1)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0.5, 1.5, 1.5, 0.9]],
                0.5,
                "iomean",
                (1, (0.9, 0)),
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[0, 0, 1, 1, 0.5], [2, 2, 3, 3, 0.9]],
                0.5,
                "iou",
                (2, [(0.9, 1), (0.5, 1)]),
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[3, 3, 4, 4, 0.9], [2, 2, 3, 3, 0.5]],
                0.5,
                "iou",
                (2, [(0.9, 0), (0.5, 1)]),
            ),
        ],
    )
    def test_match_groundtruths_and_predictions(
        self,
        groundtruths,
        predictions,
        threshold,
        localization_criterion,
        expected,
    ):
        boxes = [p[:4] for p in predictions]
        scores = [p[-1] for p in predictions]
        num_groundtruths, tp_vector = match_groundtruths_and_predictions(
            groundtruths,
            boxes,
            scores,
            threshold,
            localization_criterion,
        )
        assert num_groundtruths == expected[0]
        assert np.allclose(tp_vector, expected[1])
