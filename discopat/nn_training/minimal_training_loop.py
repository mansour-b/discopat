from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device,
) -> dict:
    """Train the model over one epoch."""
    model.train()

    for imgs, tgts in dataloader:
        images = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_dict
