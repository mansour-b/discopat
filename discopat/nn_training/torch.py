from typing import Any

import torch

from discopat.core import ComputingDevice, Dataset, NeuralNet
from discopat.nn_models.detr.engine import evaluate as evaluate_detr
from discopat.nn_models.detr.engine import train_one_epoch as train_step_detr
from discopat.nn_models.detr.models.detr import PostProcess, SetCriterion
from discopat.nn_models.detr.models.matcher import HungarianMatcher
from discopat.nn_training.evaluation import evaluate
from discopat.nn_training.nn_trainer import NNTrainer
from discopat.nn_training.torch_detection_utils.coco_utils import (
    get_coco_api_from_dataset,
)
from discopat.nn_training.torch_detection_utils.engine import train_one_epoch
from discopat.repositories.local import DISCOPATH


class TorchNNTrainer(NNTrainer):
    def train(self, num_epochs: int):
        print_frequency = self.training_loop_params["print_frequency"]

        for epoch in range(num_epochs):
            train_one_epoch(
                self.net,
                self.optimiser,
                self.dataset,
                self.device,
                epoch,
                print_freq=print_frequency,
            )
            self.lr_scheduler.step()
            evaluation_dict = evaluate(
                self.net,
                self.val_dataset,
                localization_criterion="iomean",
                device=self.device,
            )
            print()
            print("===")
            print(f"Evaluation after epoch {epoch}:")
            print()
            for k, v in evaluation_dict.items():
                print(f"{k:<10}: {v:.3f}")
            print("===")
            print()
            for callback in self.callbacks:
                callback(self.net, self.device)

    def set_default_optimiser(self) -> torch.optim.Optimizer:
        net_params = [p for p in self.net.parameters() if p.requires_grad]
        return torch.optim.SGD(
            net_params,
            lr=self.optimiser_params["learning_rate"],
            momentum=self.optimiser_params["momentum"],
            weight_decay=self.optimiser_params["weight_decay"],
        )

    def set_default_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=self.lr_scheduler_params["step_size"],
            gamma=self.lr_scheduler_params["gamma"],
        )

    @property
    def _concrete_device(self) -> torch.device:
        return {
            "cpu": torch.device("cpu"),
            "cuda": torch.device("cuda"),
            "cuda:3": torch.device("cuda:3"),
            "gpu": torch.device("cuda"),
            "mps": torch.device("mps"),
        }[self.device]


class DetrTrainer(NNTrainer):
    def __init__(
        self,
        net: NeuralNet,
        dataset: Dataset,
        val_dataset: Dataset,
        parameters: dict[str, Any],
        device: ComputingDevice,
        callbacks: list or None = None,
        num_classes=0,
    ):
        super().__init__(
            net, dataset, val_dataset, parameters, device, callbacks
        )
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

        weight_dict = {"loss_ce": 1, "loss_bbox": 5}
        weight_dict["loss_giou"] = 2

        dec_layers = 6
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "cardinality"]
        self.criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=losses,
        )
        self.criterion.to(device)
        self.postprocessors = {"bbox": PostProcess()}

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            train_step_detr(
                self.net,
                self.criterion,
                self.dataset,
                self.optimiser,
                self.device,
                epoch,
                max_norm=0,
            )
            self.lr_scheduler.step()
            evaluate_detr(
                self.net,
                self.criterion,
                self.postprocessors,
                self.val_dataset,
                base_ds=get_coco_api_from_dataset(self.val_dataset.dataset),
                device=self.device,
                output_dir=DISCOPATH / "detr_outputs",
            )
            for callback in self.callbacks:
                callback(self.net, self.device)

    def set_default_optimiser(self) -> torch.optim.Optimizer:
        net_params = [p for p in self.net.parameters() if p.requires_grad]
        return torch.optim.SGD(
            net_params,
            lr=self.optimiser_params["learning_rate"],
            momentum=self.optimiser_params["momentum"],
            weight_decay=self.optimiser_params["weight_decay"],
        )

    def set_default_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=self.lr_scheduler_params["step_size"],
            gamma=self.lr_scheduler_params["gamma"],
        )

    @property
    def _concrete_device(self) -> torch.device:
        return {
            "cpu": torch.device("cpu"),
            "cuda": torch.device("cuda"),
            "cuda:3": torch.device("cuda:3"),
            "gpu": torch.device("cuda"),
            "mps": torch.device("mps"),
        }[self.device]
