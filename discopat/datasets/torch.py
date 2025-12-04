from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T  # noqa: N812

from discopat.core import Box, Frame, Keypoint
from discopat.manual_annotations.operations import keypoint_to_box


class TorchDataset(Dataset):
    def __init__(
        self,
        frame_list: list[Frame],
        label_map: dict[str, int],
        transforms: T.Transform or None = None,
        channel_mode: str = "channels_first",
    ):
        self._frame_list = frame_list
        self._label_map = label_map
        self._transforms = (
            transforms if transforms is not None else T.Identity()
        )
        self._channel_mode = channel_mode

    def __getitem__(self, index: int):
        frame = self._frame_list[index]
        image = self.prepare_image_tensor(frame)
        target = self.make_target(frame)
        return self._transforms(image, target)

    def __len__(self: int):
        return len(self._frame_list)

    def prepare_image_tensor(self, frame: Frame) -> torch.Tensor:
        image_tensor = torch.from_numpy(self._preprocess_image_array(frame))
        return self._rescale_histogram(image_tensor).to(torch.float32)

    def _preprocess_image_array(self, frame: Frame) -> np.array:
        channel_axis_dict = {"channels_first": 0, "channels_last": -1}
        channel_axis = channel_axis_dict[self._channel_mode]
        image_array = np.expand_dims(frame.image_array, axis=channel_axis)
        return np.repeat(image_array, repeats=3, axis=channel_axis)

    @staticmethod
    def _rescale_histogram(image_tensor: torch.Tensor) -> torch.Tensor:
        min_value = image_tensor.min()
        max_value = image_tensor.max()
        max_range = max_value - min_value
        return (image_tensor - min_value) / max_range

    @abstractmethod
    def make_target(self, frame: Frame) -> dict[str, torch.Tensor]:
        pass

    @staticmethod
    def _box_to_xyxy_format(box: Box) -> list[float]:
        return [box.xmin, box.ymin, box.xmax, box.ymax]

    def _make_xyxy_box_array(self, box_list: list[Box]) -> np.array:
        return np.array([self._box_to_xyxy_format(box) for box in box_list])

    def _make_label_array(self, box_list: list[Box]) -> np.array:
        label_map = self._label_map
        return np.array([label_map[box.label] for box in box_list])

    @staticmethod
    def _make_area_array(box_list: list[Box]) -> np.array:
        return np.array([box.width * box.height for box in box_list])


class TorchBoxDataset(TorchDataset):
    def make_target(self, frame: Frame) -> dict[str, int or torch.Tensor]:
        box_list = [
            annotation
            for annotation in frame.annotations
            if annotation.type == "box"
        ]
        area_array = self._make_area_array(box_list)
        box_array = self._make_xyxy_box_array(box_list)
        label_array = self._make_label_array(box_list)

        box_tensor = torch.as_tensor(box_array)
        box_tensor[..., 0::2] = box_tensor[..., 0::2].clamp(0, frame.width - 1)
        box_tensor[..., 1::2] = box_tensor[..., 1::2].clamp(0, frame.height - 1)

        box_tensor = tv_tensors.BoundingBoxes(
            box_tensor,
            format="XYXY",
            canvas_size=(frame.height, frame.width),
            dtype=torch.float32,
        )

        return {
            "area": torch.as_tensor(area_array, dtype=torch.float32),
            "boxes": box_tensor,
            "image_id": torch.as_tensor(int(frame.name)),
            "iscrowd": torch.zeros((len(box_list),), dtype=torch.int64),
            "labels": torch.as_tensor(label_array),
        }


class TorchKeypointDataset(TorchDataset):
    def __init__(
        self,
        frame_list: list[Frame],
        label_map: dict[str, int],
        box_w_padding: float = 0.5,
        box_h_padding: float = 0.5,
    ):
        super().__init__(self, frame_list, label_map)
        self.box_w_padding = box_w_padding
        self.box_h_padding = box_h_padding

    def make_target(self, frame: Frame) -> dict[str, int or torch.Tensor]:
        keypoint_list = [
            annotation
            for annotation in frame.annotations
            if annotation.type == "keypoint"
        ]
        box_list = [
            keypoint_to_box(
                keypoint,
                w_padding=self.box_w_padding,
                h_padding=self.box_h_padding,
            )
            for keypoint in keypoint_list
        ]
        area_array = self._make_area_array(box_list)
        box_array = self._make_xyxy_box_array(box_list)
        keypoint_array = self._make_keypoint_array(frame.annotations)
        label_array = self._make_label_array(box_list)

        return {
            "area": torch.as_tensor(area_array),
            "boxes": torch.as_tensor(box_array, dtype=torch.float32),
            "image_id": int(frame.name),
            "iscrowd": torch.zeros((len(box_list),), dtype=torch.int64),
            "keypoints": torch.as_tensor(keypoint_array, dtype=torch.float32),
            "labels": torch.as_tensor(label_array),
        }

    @staticmethod
    def _make_keypoint_array(keypoint_list: list[Keypoint]) -> np.array:
        return np.array(
            [
                [(x, y, 1.0) for x, y in keypoint.point_list]
                for keypoint in keypoint_list
            ]
        )
