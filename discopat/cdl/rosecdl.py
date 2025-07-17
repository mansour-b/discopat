import numpy as np
import torch
from rosecdl.csc import ConvolutionalSparseCoder
from skimage.feature import peak_local_max
from torchvision.ops import nms

from discopat.core import Box, CDModel, Frame


class RoseCDLModel(CDModel):
    def __init__(
        self,
        csc: ConvolutionalSparseCoder,
        label_map: dict[str, int],
        iou_threshold: float,
        score_threshold: float,
    ):
        self.csc = csc
        self.label_map = label_map

        self.box_width = self.csc.kernel_size[1]
        self.box_height = self.csc.kernel_size[0]

        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    @classmethod
    def from_dict(cls, model_as_dict: dict):
        pass

    def to_dict(self) -> dict:
        return {}

    def predict(self, frame: Frame) -> Frame:
        image_tensor = (
            torch.tensor(np.tile(frame.image_array, (1, 1, 1, 1)))
            .to(torch.float32)
            .to(device="mps")
        )

        _, activation_vector = self.csc(image_tensor)
        activation_vector = activation_vector.detach().cpu().numpy().squeeze()
        max_activation = activation_vector.max()

        coords = np.concat(
            [peak_local_max(activation) for activation in activation_vector]
        )  # loop over atoms

        scores = []
        for activation in activation_vector:  # loop over atoms
            detections = peak_local_max(activation)
            if len(detections) == 0:
                continue

            rows, cols = zip(*detections)
            rows = np.array(rows)
            cols = np.array(cols)
            scores = np.concat([scores, activation[rows, cols]])

        box_tensor = torch.zeros(len(coords), 4)
        box_tensor[:, 0] = torch.tensor(coords[:, 1])
        box_tensor[:, 1] = torch.tensor(coords[:, 0])
        box_tensor[:, 2] = torch.tensor(coords[:, 1]) + self.box_width
        box_tensor[:, 3] = torch.tensor(coords[:, 0]) + self.box_height

        kept_indices = nms(
            boxes=box_tensor,
            scores=torch.tensor(scores).to(torch.float32),
            iou_threshold=self.iou_threshold,
        ).numpy()

        if len(scores) == 0:
            return Frame(
                name=frame.name,
                width=frame.width,
                height=frame.height,
                annotations=[],
                image_array=frame.image_array,
            )

        box_list = [
            Box(
                label="blob_front",
                x=x,
                y=y,
                width=self.box_width,
                height=self.box_height,
                score=score,
            )
            for (y, x), score in zip(
                coords[kept_indices], np.array(scores)[kept_indices]
            )
            if score >= self.score_threshold * max_activation
        ]
        return Frame(
            name=frame.name,
            width=frame.width,
            height=frame.height,
            annotations=box_list,
            image_array=frame.image_array,
        )
