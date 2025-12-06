import torch

from discopat.nn_training.detr.criterion import SetCriterion, accuracy
from discopat.nn_training.detr.matcher import HungarianMatcher


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


class TestSetCriterion:
    def test_init(self):
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

        criterion = SetCriterion(
            num_classes=2,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=losses,
        )

        assert criterion.num_classes == 2
        assert criterion.eos_coef == 0.1
        assert criterion.losses == ["labels", "boxes", "cardinality"]
