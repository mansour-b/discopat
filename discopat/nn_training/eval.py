import numpy as np
from torch.utils.data import DataLoader

from discopat.core import ComputingDevice, NeuralNet
from discopat.metrics import compute_iomean, compute_iou


def match_gts_and_preds(
    groundtruths: list,
    predictions: list,
    threshold: float,
    localization_criterion: str,
) -> tuple[int, list[tuple(float, float)]]:
    """Match GTs and preds on an image in the dataset.

    Args:
        groundtruths: list of groundtruths, boxes [x1, y1, x2, y2],
        predictions: list of predictions, boxes [x1, y1, x2, y2, score],
        threshold: threshold for the localization metric,
        localization_criterion: metric used to assess the fit between GTs and preds.

    Returns:
        A report for the considered image, containing:
            - The total number of groundtruths,
            - For each pred, a tuple (score, is_tp).

    """
    localizing_function = {"iou": compute_iou, "iomean": compute_iomean}[
        localization_criterion
    ]

    # Sort predictions by score descending
    predictions = sorted(predictions, key=lambda x: x[-1], reverse=True)
    pred_boxes = [p[:4] for p in predictions]
    scores = [p[-1] for p in predictions]

    # Track matches
    gt_matched = np.zeros(len(groundtruths), dtype=bool)
    tps = np.zeros(len(predictions))

    for i, pred in enumerate(pred_boxes):
        # Find best matching GT
        loc_scores = [localizing_function(pred, gt) for gt in groundtruths]
        best_gt = int(np.argmax(loc_scores))
        best_loc = loc_scores[best_gt]
        if best_loc >= threshold and not gt_matched[best_gt]:
            tps[i] = 1
            gt_matched[best_gt] = True

    return len(groundtruths), zip(scores, tps)


def evaluate(
    model: NeuralNet, data_loader: DataLoader, device: ComputingDevice
):
    pass
