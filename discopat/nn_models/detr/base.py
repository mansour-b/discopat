import torch

from discopat.core import Box, ComputingDevice, Frame, NNModel


class DETRModel(NNModel):
    _device: ComputingDevice

    def pre_process(self, frame: Frame) -> torch.Tensor:
        pass

    def post_process(
        self, raw_predictions: list[dict[torch.Tensor]]
    ) -> list[Box]:
        pass
