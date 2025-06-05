import torch

from patrick.nn_training.nn_trainer import NNTrainer
from patrick.nn_training.torch_detection_utils.engine import (
    evaluate,
    train_one_epoch,
)


class TorchNNTrainer(NNTrainer):

    def train(self, num_epochs: int):

        print_frequency = self.training_loop_params["print_frequency"]

        for epoch in range(num_epochs):
            train_one_epoch(
                self.net,
                self.optimiser,
                self.dataset,
                self._concrete_device,
                epoch,
                print_freq=print_frequency,
            )
            self.lr_scheduler.step()
            evaluate(self.net, self.val_dataset, device=self._concrete_device)

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
        return {"cpu": torch.device("cpu"), "gpu": torch.device("cuda")}[
            self.device
        ]
