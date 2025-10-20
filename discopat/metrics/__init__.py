import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(box1: list, box2: list, eps: float = 1e-10) -> float:
    """Compute IoU between two boxes.

    Args:
        box1: [x1, y1, x2, y2, (score)],
        box2: [x1, y1, x2, y2, (score)],
        eps: Safety term for the denominator.

    Returns:
        The IoU (float).

    """
    xmin_max = max(box1[0], box2[0])
    xmax_min = min(box1[2], box2[2])
    ymin_max = max(box1[1], box2[1])
    ymax_min = min(box1[3], box2[3])

    intersection_area = max(xmax_min - xmin_max, 0) * max(
        ymax_min - ymin_max, 0
    )
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / max(union_area, eps)


def compute_iomean(box1: list, box2: list, eps: float = 1e-10) -> float:
    """Compute IoMean between two boxes.

    Args:
        box1: [x1, y1, x2, y2, (score)],
        box2: [x1, y1, x2, y2, (score)],
        eps: Safety term for the denominator.

    Returns:
        The IoMean (float).

    """
    xmin_max = max(box1[0], box2[0])
    xmax_min = min(box1[2], box2[2])
    ymin_max = max(box1[1], box2[1])
    ymax_min = min(box1[3], box2[3])

    intersection_area = max(xmax_min - xmin_max, 0) * max(
        ymax_min - ymin_max, 0
    )
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    mean_area = (box1_area + box2_area) / 2
    return intersection_area / max(mean_area, eps)


def compute_iou_matrix(groundtruths: list, predictions: list) -> list:
    """Compute IoU matrix between groundtruths and predictions.

    Args:
        groundtruths: list of boxes in the format [x1, y1, x2, y2],
        predictions: list of boxes in the format [x1, y1, x2, y2, score].

    Returns:
        A matrix where every columns corresponds to a groundtruth, every row to a prediction, and the coefficient at (i, j) is the iou between predictions[i] and groundtruth[j].

    """
    return [[compute_iou(g, p) for g in groundtruths] for p in predictions]


def compute_ap(
    groundtruths: list,
    predictions: list,
    threshold: float,
    localization_criterion: str,
) -> float:
    """Average Precision (AP) at a given IoU/IoMean/whatever threshold."""
    if len(groundtruths) == 0:
        logger.warning("No groundtruth boxes.")
        return 0.0

    localizing_function = {"iou": compute_iou, "iomean": compute_iomean}[
        localization_criterion
    ]

    # Sort predictions by score descending
    predictions = sorted(predictions, key=lambda x: x[-1], reverse=True)
    pred_boxes = [p[:4] for p in predictions]

    # Track matches
    gt_matched = np.zeros(len(groundtruths), dtype=bool)
    tps = np.zeros(len(predictions))
    fps = np.zeros(len(predictions))

    for i, pred in enumerate(pred_boxes):
        # Find best matching GT
        loc_scores = [localizing_function(pred, gt) for gt in groundtruths]
        best_gt = int(np.argmax(loc_scores))
        best_loc = loc_scores[best_gt]
        if best_loc >= threshold and not gt_matched[best_gt]:
            tps[i] = 1
            gt_matched[best_gt] = True
        else:
            fps[i] = 1

    # Cumulative sums
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)

    # Prepend zeros for the case score_threshold=1
    tp_cum = np.concatenate([[0], tp_cum])
    fp_cum = np.concatenate([[0], fp_cum])

    recall = tp_cum / len(groundtruths)
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)

    # Ensure precision is non-increasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Compute area under curve (AP)
    return np.trapezoid(precision, recall)
