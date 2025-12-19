from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device,
    criterion: torch.nn.Module | None = None,
) -> dict:
    """Train the model over one epoch."""
    model.train()
    if criterion is not None:
        criterion.train()

    for imgs, tgts in dataloader:
        images = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        if criterion is None:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        else:
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict
                if k in weight_dict
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_dict
