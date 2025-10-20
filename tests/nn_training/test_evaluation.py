from typing import Self

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
        ("groundtruths", "predictions", "expected"),
        [
            pytest.param([[0, 0, 1, 1]], [[0, 0, 1, 1]], 1.0),
            pytest.param([[0, 0, 1, 1]], [[0.5, 0, 1.5, 1]], 0.5),
            pytest.param([[0, 0, 1, 1]], [[0.5, 0.5, 1.5, 1.5]], 0.25),
        ],
    )
    def test_match_gts_and_preds(self, groundtruths, predictions, expected):
        for threshold in [0.5, 0.75, 0.95]:
            for localization_criterion in ["iou", "iomean"]:
                assert (
                    match_gts_and_preds(
                        groundtruths,
                        predictions,
                        threshold,
                        localization_criterion,
                    )
                    == expected
                )

    def test_evaluate(self):
        pass
