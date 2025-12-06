import torch

from discopat.nn_training.detr.criterion import SetCriterion, accuracy


def make_fake_outputs():
    batch_size = 4
    num_queries = 100
    num_classes = 2
    return {
        "pred_logits": torch.zeros(batch_size, num_queries, num_classes),
        "pred_boxes": torch.zeros(batch_size, num_queries, 4),
    }


def test_accuracy():
    output = torch.tensor(
        [
            [-0.3, 0.7],
            [-0.3, 0.7],
            [-0.3, 0.7],
            [-0.3, 0.7],
            [-0.3, 0.7],
            [-0.3, 0.7],
            [-0.3, 0.7],
            [-0.3, 0.7],
        ]
    )
    target = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
    assert accuracy(output, target)[0] == 100
