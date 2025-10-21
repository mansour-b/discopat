import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from discopat.nn_training.evaluation import evaluate, match_gts_and_preds


class TestEval:
    @staticmethod
    def make_model(predictions):
        class DummyModel:
            def __call__(self, images):  # noqa: ARG002
                return [
                    {
                        "boxes": p[:, :4],
                        "scores": p[:, -1],
                    }
                    for p in predictions
                ]

        return DummyModel()

    @staticmethod
    def make_data_loader(
        groundtruths, batch_size=2, shuffle=False, num_workers=0
    ):
        class DummyDetectionDataset(Dataset):
            """
            A simple dataset that returns an image tensor and its corresponding list of ground-truth boxes.

            Args:
                groundtruths (list): list of targets, each target is a list of bounding boxes.
                    Each bounding box can be a tensor or list of [x_min, y_min, x_max, y_max].
                image_size (tuple): (C, H, W) size for the dummy image tensor.
            """

            def __init__(self, groundtruths, image_size=(3, 224, 224)):
                self.groundtruths = groundtruths
                self.image_size = image_size

            def __len__(self):
                return len(self.groundtruths)

            def __getitem__(self, idx):
                # Dummy image (for testing)
                image = torch.zeros(self.image_size, dtype=torch.float32)

                # Convert list of boxes to tensor
                boxes = torch.tensor(
                    self.groundtruths[idx], dtype=torch.float32
                )

                # Create a dummy target dict (like torchvision detection datasets)
                target = {
                    "boxes": boxes,
                    "labels": torch.ones(
                        (len(boxes),), dtype=torch.int64
                    ),  # optional dummy labels
                }

                return image, target

        dataset = DummyDetectionDataset(groundtruths)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    @pytest.mark.parametrize(
        (
            "groundtruths",
            "predictions",
            "threshold",
            "localization_criterion",
            "expected",
        ),
        [
            pytest.param(
                [[0, 0, 1, 1]], [[0, 0, 1, 1, 0.9]], 0.5, "iou", (1, (0.9, 1))
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0, 1.5, 1, 0.9]],
                0.5,
                "iou",
                (1, (0.9, 0)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0.5, 1.5, 1.5, 0.9]],
                0.5,
                "iou",
                (1, (0.9, 0)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0, 0, 1, 1, 0.9]],
                0.5,
                "iomean",
                (1, (0.9, 1)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0, 1.5, 1, 0.9]],
                0.5,
                "iomean",
                (1, (0.9, 1)),
            ),
            pytest.param(
                [[0, 0, 1, 1]],
                [[0.5, 0.5, 1.5, 1.5, 0.9]],
                0.5,
                "iomean",
                (1, (0.9, 0)),
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[0, 0, 1, 1, 0.5], [2, 2, 3, 3, 0.9]],
                0.5,
                "iou",
                (2, [(0.9, 1), (0.5, 1)]),
            ),
            pytest.param(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                [[3, 3, 4, 4, 0.9], [2, 2, 3, 3, 0.5]],
                0.5,
                "iou",
                (2, [(0.9, 0), (0.5, 1)]),
            ),
        ],
    )
    def test_match_gts_and_preds(
        self,
        groundtruths,
        predictions,
        threshold,
        localization_criterion,
        expected,
    ):
        num_groundtruths, tp_vector = match_gts_and_preds(
            groundtruths,
            predictions,
            threshold,
            localization_criterion,
        )
        assert num_groundtruths == expected[0]
        assert np.allclose(tp_vector, expected[1])

    @pytest.mark.parametrize(
        ("groundtruths", "predictions", "localization_criterion", "expected"),
        [
            # pytest.param([[]], [np.empty((0, 5))], "iou", {"AP50": 0, "AP": 0}),
            pytest.param(
                [[0, 0, 1, 1]],
                [np.array([[0, 0, 1, 1, 0.9]])],
                "iou",
                {"AP50": 0, "AP": 0},
            ),
        ],
    )
    def test_evaluate(
        self, predictions, groundtruths, localization_criterion, expected
    ):
        model = self.make_model(predictions)
        data_loader = self.make_data_loader(groundtruths)
        res = evaluate(model, data_loader, localization_criterion)
        assert evaluate(model, data_loader, localization_criterion) == expected
