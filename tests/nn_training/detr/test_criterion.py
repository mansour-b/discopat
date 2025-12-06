import torch

from discopat.nn_training.detr.criterion import SetCriterion, accuracy
from discopat.nn_training.detr.matcher import HungarianMatcher


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
    @staticmethod
    def init_criterion(num_classes=2):
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

        return SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=losses,
        )

    def test_init(self):
        criterion = self.init_criterion()

        assert criterion.num_classes == 2
        assert criterion.eos_coef == 0.1
        assert criterion.losses == ["labels", "boxes", "cardinality"]

    def test_loss_labels(self):
        outputs = {
            "pred_logits": torch.tensor(
                [
                    [
                        [0.2730, 0.0024],
                        [0.4295, 0.2533],
                        [0.0157, -0.2179],
                        [0.7240, 0.1416],
                        [0.1600, 0.6418],
                        [-0.0357, 0.2144],
                        [0.6913, -0.3039],
                        [0.0020, -0.0107],
                        [0.3174, 0.2027],
                        [0.2150, 0.2840],
                    ],
                ],
            ),
        }
        targets = [{"labels": torch.tensor([1, 1, 1, 1])}]
        indices = [
            (
                torch.tensor([4, 5, 1, 3]),
                torch.tensor([0, 3, 1, 2]),
            )
        ]

        num_classes = (
            outputs["pred_logits"].shape[-1] - 1
        )  # Remove background class
        criterion = self.init_criterion(num_classes)

        loss = criterion.loss_labels(
            outputs, targets, indices=indices, num_boxes=-1
        )
        expected = {
            "class_error": torch.tensor(50.0),
            "loss_ce": torch.tensor(0.7943),
        }
        assert loss.keys() == expected.keys()
        assert loss["class_error"] == expected["class_error"]
        assert torch.isclose(loss["loss_ce"], expected["loss_ce"], rtol=1e-3)
