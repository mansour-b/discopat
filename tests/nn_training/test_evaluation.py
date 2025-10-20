from typing import Self

from dsicopat.nn_training.evaluation import evaluate

from discopat.core import ComputingDevice, Frame, Model


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

    def test_evaluate(self):
        pass
