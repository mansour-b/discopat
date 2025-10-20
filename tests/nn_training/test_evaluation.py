from typing import Self

import numpy as np
import pytest

from discopat.core import ComputingDevice, Frame, Model
from discopat.nn_training.evaluation import evaluate, match_gts_and_preds


class TestEval:
    @staticmethod
    def make_model():
        class DumbModel(Model):
            def pre_process(self):
                pass

            def post_process(self):
                pass

            def predict(self, frame: Frame) -> Frame:
                return frame

            @classmethod
            def from_dict(cls, model_as_dict: dict) -> Self:
                raise NotImplementedError

            def to_dict(self) -> dict:
                raise NotImplementedError

            def set_device(self, device: ComputingDevice) -> None:
                raise NotImplementedError

        return DumbModel()

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
        ],
    )
    def test_match_gts_and_preds(
        self,
        groundtruths,
        predictions,
        threshold,
        localization_criterion,
        expected,
    ):
        num_groundtruths, tp_vector = match_gts_and_preds(
            groundtruths,
            predictions,
            threshold,
            localization_criterion,
        )
        assert num_groundtruths == expected[0]
        assert np.allclose(tp_vector, expected[1])

    def test_evaluate(self):
        pass
