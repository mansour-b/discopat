from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from patrick.core import ComputingDevice, Dataset, NeuralNet


class NNTrainer(ABC):
    def __init__(
        self,
        net: NeuralNet,
        dataset: Dataset,
        val_dataset: Dataset,
        parameters: dict[str, Any],
        device: ComputingDevice,
    ):
        self.net = net
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.device = device

        (
            self.optimiser_params,
            self.lr_scheduler_params,
            self.training_loop_params,
        ) = self.parse_parameters(parameters)

        self.optimiser = self.set_default_optimiser()
        self.lr_scheduler = self.set_default_lr_scheduler()

    @abstractmethod
    def train(self) -> None:
        pass

    @staticmethod
    def parse_parameters(parameters: dict[str, Any]) -> tuple(dict, dict, dict):
        optimiser_params = parameters["optimiser"]
        lr_scheduler_params = parameters["lr_scheduler"]
        training_loop_params = parameters["training_loop"]
        return optimiser_params, lr_scheduler_params, training_loop_params

    @abstractmethod
    def set_default_optimiser(self) -> Any:
        pass

    @abstractmethod
    def set_default_lr_scheduler(self) -> Any:
        pass
