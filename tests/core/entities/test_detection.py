from __future__ import annotations

import numpy as np
from typing_extensions import Self

from patrick.core import (
    Annotation,
    Array,
    Box,
    ComputingDevice,
    Frame,
    Model,
    NeuralNet,
    NNModel,
)


class TestModel:

    @staticmethod
    def make_model():
        class DumbModel(Model):
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

    def test_init(self):
        _ = self.make_model()

    def test_predict(self):
        frame = Frame(name="frame_0", width=128, height=128, annotations=[])
        model = self.make_model()
        assert model.predict(frame) == frame


class TestNeuralNet:

    @staticmethod
    def make_net():
        class DumbNN(NeuralNet):
            def __call__(self, input_array: Array) -> Array:
                return input_array

        return DumbNN()

    def test_init(self):
        _ = self.make_net()

    def test_call(self):
        net = self.make_net()
        assert np.equal(net(np.zeros(3)), np.zeros(3)).max()


class TestNNModel:
    @staticmethod
    def make_model():
        class DumbNN(NeuralNet):
            def __call__(self, input_array: Array) -> Array:
                return input_array

        class DumbNNModel(NNModel):

            def pre_process(self, frame: Frame) -> Array:
                return frame.image_array

            def post_process(self, net_predictions: Array) -> list[Annotation]:
                return [
                    Box(label="blob", x=1, y=1, width=1, height=1, score=1.0)
                ]

            @classmethod
            def from_dict(cls, model_as_dict: dict) -> Self:
                raise NotImplementedError

            def to_dict(self) -> dict:
                raise NotImplementedError

            def set_device(self, device: ComputingDevice) -> None:
                raise NotImplementedError

        return DumbNNModel(
            net=DumbNN(),
            label_map={"blob": 1},
            model_parameters={},
        )

    def test_init(self):
        model = self.make_model()
        assert model.label_map == {"blob": 1}
        assert model.model_parameters == {}

    def test_pre_process(self):
        model = self.make_model()
        frame = Frame(
            name="frame_0",
            width=16,
            height=16,
            annotations=[],
            image_array=np.ones((16, 16)),
        )
        assert np.equal(model.pre_process(frame), np.ones((16, 16))).max()

    def test_post_process(self):
        model = self.make_model()
        assert model.post_process(np.zeros(3)) == [
            Box(label="blob", x=1, y=1, width=1, height=1, score=1.0)
        ]

    def test_predict(self):
        model = self.make_model()
        frame = Frame(
            name="frame_0",
            width=16,
            height=16,
            annotations=[],
            image_array=np.ones((16, 16)),
        )
        expected_result = Frame(
            name="frame_0",
            width=16,
            height=16,
            annotations=[
                Box(label="blob", x=1, y=1, width=1, height=1, score=1.0)
            ],
            image_array=np.ones((16, 16)),
        )
        assert model.predict(frame) == expected_result
