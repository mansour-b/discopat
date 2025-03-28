from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from typing_extensions import Self

from patrick.core import Box, ComputingDevice, Frame, NeuralNet, NNModel


class FasterRCNNModel(NNModel):
    _device: ComputingDevice

    def __init__(
        self,
        net: NeuralNet,
        label_map: dict[str, int],
        model_parameters: dict,
    ):
        self.net = net
        self.label_map = label_map
        self.pre_proc_params = model_parameters["pre_processing"]
        self.post_proc_params = model_parameters["post_processing"]
        self._device = "cpu"

    def pre_process(self, frame: Frame) -> torch.Tensor:
        input_array = np.expand_dims(frame.image_array, axis=0)
        input_array = np.repeat(input_array, repeats=3, axis=0)

        input_array = np.expand_dims(input_array, axis=0)

        input_array = torch.as_tensor(input_array)
        return input_array.to(torch.float32).to(self._concrete_device)

    def post_process(self, predictions: list[dict[torch.Tensor]]) -> list[Box]:
        predictions = predictions[0]

        kept_indices = nms(
            boxes=predictions["boxes"],
            scores=predictions["scores"],
            iou_threshold=self.post_proc_params["nms_iou_threshold"],
        )
        for k in predictions:
            predictions[k] = predictions[k][kept_indices]

        box_list = []
        for box_xyxy, label, score in zip(
            predictions["boxes"],
            predictions["labels"],
            predictions["scores"],
        ):
            if score < self.post_proc_params["score_threshold"]:
                continue
            box = self.make_box_from_tensors(box_xyxy, label, score)
            box_list.append(box)
        return box_list

    @property
    def reversed_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def make_box_from_tensors(
        self,
        box_xyxy: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
    ) -> Box:
        x, y, width, height = self.xyxy_to_xywh(*box_xyxy)
        str_label = self.reversed_label_map[int(label)]
        return Box(
            label=str_label, x=x, y=y, width=width, height=height, score=score
        )

    @staticmethod
    def xyxy_to_xywh(
        xmin: float, ymin: float, xmax: float, ymax: float
    ) -> tuple[float, float, float, float]:
        x = xmin
        y = ymin
        width = xmax - xmin
        height = ymax - ymin
        return x, y, width, height

    @classmethod
    def from_dict(cls, model_as_dict: dict) -> Self:
        net_builder = TorchNetBuilder()

        return cls(
            net=net_builder.build(model_as_dict),
            label_map=model_as_dict["label_map"],
            model_parameters=model_as_dict["model_parameters"],
        )

    def to_dict(self) -> dict:
        pass

    @property
    def _concrete_device(self) -> torch.device:
        return {"cpu": torch.device("cpu"), "gpu": torch.device("cuda")}[
            self._device
        ]

    def set_device(self, device: ComputingDevice) -> None:
        self._device = device
        self.net.to(self._concrete_device)


class TorchNetBuilder:

    def build(self, model_as_dict: dict[str, str or BytesIO]) -> NeuralNet:
        weights = torch.load(
            model_as_dict["raw_net"], weights_only=True, map_location="cpu"
        )
        net_parameters = model_as_dict["model_parameters"]["net"]

        net = self._define_architecture(net_parameters)
        self._load_weights(net, weights)
        net.eval()

        return net

    def _define_architecture(self, net_parameters: dict[str, Any]) -> NeuralNet:
        net = fasterrcnn_resnet50_fpn()
        in_features = net.roi_heads.box_predictor.cls_score.in_features

        num_classes = net_parameters["num_classes"]
        num_classes_including_background_class = num_classes + 1
        net.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=num_classes_including_background_class
        )
        return net

    def _load_weights(self, net: NeuralNet, weights: dict) -> None:
        net.load_state_dict(weights)
