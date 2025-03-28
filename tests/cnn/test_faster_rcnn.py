from __future__ import annotations

import numpy as np
import torch

from patrick.cnn.faster_rcnn import FasterRCNNModel
from patrick.core import Box, Frame


class MockNet:
    def __call__(self, frame: Frame) -> list[dict[torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[1, 2, 3, 4]], dtype=float),
                "labels": torch.tensor([1], dtype=int),
                "scores": torch.tensor([1.0], dtype=float),
            }
        ]


class TestFasterRCNN:
    @staticmethod
    def get_model():
        model = FasterRCNNModel(
            net=MockNet(),
            label_map={"blob": 1},
            model_parameters={
                "pre_processing": {},
                "net": {},
                "post_processing": {
                    "nms_iou_threshold": 0.2,
                    "score_threshold": 0.7,
                },
            },
        )
        return model

    def test_init(self):
        model = self.get_model()
        assert model.label_map == {"blob": 1}
        assert model.post_proc_params["nms_iou_threshold"] == 0.2
        assert model.post_proc_params["score_threshold"] == 0.7

    def test_reversed_label_map(self):
        model = self.get_model()
        assert model.reversed_label_map == {1: "blob"}

    def test_xyxy_to_xywh(self):
        model = self.get_model()
        assert model.xyxy_to_xywh(1, 2, 3, 4) == (1, 2, 2, 2)

    def test_make_box_from_tensors(self):
        model = self.get_model()
        box_xyxy = torch.tensor([1, 2, 3, 4])
        label = torch.tensor(1)
        score = torch.tensor(1.0)
        assert model.make_box_from_tensors(box_xyxy, label, score) == Box(
            label="blob",
            x=1,
            y=2,
            width=2,
            height=2,
            score=1,
        )

    def test_pre_process(self):
        model = self.get_model()
        frame = Frame(
            name="frame_0",
            width=32,
            height=32,
            annotations=[],
            image_array=np.zeros((32, 32)),
        )
        input_array = model.pre_process(frame)
        assert torch.equal(input_array, torch.zeros(1, 3, 32, 32))

    def test_post_process(self):
        model = self.get_model()
        predictions = [
            {
                "boxes": torch.tensor([[1, 2, 3, 4]], dtype=float),
                "labels": torch.tensor([1], dtype=int),
                "scores": torch.tensor([1.0], dtype=float),
            }
        ]
        assert model.post_process(predictions) == [
            Box(label="blob", x=1, y=2, width=2, height=2, score=1)
        ]

    def test_predict(self):
        model = self.get_model()
        model.net = MockNet()
        frame = Frame(
            name="frame_0",
            width=32,
            height=32,
            annotations=[],
            image_array=np.zeros((32, 32)),
        )
        output_frame = model.predict(frame)
        assert output_frame == Frame(
            name="frame_0",
            width=32,
            height=32,
            annotations=[
                Box(label="blob", x=1, y=2, width=2, height=2, score=1)
            ],
        )
