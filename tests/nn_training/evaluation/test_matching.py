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
            "localization_criterion",
            "expected",
        ),
        [
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1, 0.9]],
                "iou",
                {"matching_matrix": np.array([[1]]), "scores": np.array([0.9])},
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0, 1.5, 1, 0.9]],
                "iou",
                {
                    "matching_matrix": np.array([[1 / 3]]),
                    "scores": np.array([0.9]),
                },
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0.5, 1.5, 1.5, 0.9]],
                "iou",
                {
                    "matching_matrix": np.array([[1 / 7]]),
                    "scores": np.array([0.9]),
                },
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1, 0.9]],
                "iomean",
                {
                    "matching_matrix": np.array([[1]]),
                    "scores": np.array([0.9]),
                },
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0, 1.5, 1, 0.9]],
                "iomean",
                {
                    "matching_matrix": np.array([[0.5]]),
                    "scores": np.array([0.9]),
                },
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0.5, 1.5, 1.5, 0.9]],
                "iomean",
                {
                    "matching_matrix": np.array([[1 / 4]]),
                    "scores": np.array([0.9]),
                },
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[0, 0, 1, 1, 0.5], [2, 2, 3, 3, 0.9]],
                "iou",
                {
                    "matching_matrix": np.array(
                        [
                            [0, 1],
                            [1, 0],
                        ]
                    ),
                    "scores": np.array([0.9, 0.5]),
                },
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[3, 3, 4, 4, 0.9], [2, 2, 3, 3, 0.5]],
                "iou",
                {
                    "matching_matrix": np.array(
                        [
                            [0, 0],
                            [0, 1],
                        ]
                    ),
                    "scores": np.array([0.9, 0.5]),
                },
            ),
        ],
    )
    def test_match_groundtruths_and_predictions(
        self,
        groundtruths,
        predictions,
        localization_criterion,
        expected,
    ):
        boxes = [p[:4] for p in predictions]
        scores = [p[-1] for p in predictions]
        res = match_groundtruths_and_predictions(
            groundtruths,
            boxes,
            scores,
            localization_criterion,
        )
        assert np.allclose(res["matching_matrix"], expected["matching_matrix"])
        assert np.allclose(res["scores"], expected["scores"])
