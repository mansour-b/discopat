import numpy as np


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
    xmax_min = min(box1[1], box2[1])
    ymin_max = max(box1[2], box2[2])
    ymax_min = min(box1[3], box2[3])

    intersection_area = max(xmax_min - xmin_max, 0) * max(
        ymax_min - ymin_max, 0
    )
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / max(union_area, eps)


def compute_iou_matrix(groundtruths: list, predictions: list) -> list:
    """Compute IoU matrix between groundtruths and predictions.

    Args:
        groundtruths: list of boxes in the format [x1, y1, x2, y2],
        predictions: list of boxes in the format [x1, y1, x2, y2, score].

    Returns:
        A matrix where every columns corresponds to a groundtruth, every row to a prediction, and the coefficient at (i, j) is the iou between predictions[i] and groundtruth[j].

    """
    return [[compute_iou(g, p) for g in groundtruths] for p in predictions]


def compute_ap_at_threshold(
    groundtruths: list, predictions: list, threshold: float
) -> float:
    predictions.sort(key=lambda x: x[-1], reverse=True)
    iou_matrix = compute_iou_matrix(groundtruths, predictions)
    iou_matrix = np.sort(iou_matrix, axis=-1, reverse=True)

    candidate_matrix = iou_matrix[iou_matrix >= threshold]

    unmatched_preds = []
    matches = {}
    for pred in candidate_matrix:
        matching_gts = np.argwhere(
            np.delete(candidate_matrix[pred], matches.values())
        )
        if len(matching_gts) == 0:
            unmatched_preds.append(pred)
        matches[pred] = matching_gts[0]
    unmatched_gts = np.delete(np.arange(len(groundtruths)), matches.values())

    precision = []
    recall = []
    for i in matches:
        tp = i + 1  # Matched preds wth score >= current prediction's
        fp = len(
            [
                pred for pred in unmatched_gts if pred[-1] >= predictions[i][-1]
            ]  # Unmatched preds with score >= current prediction's
        )
        fn = (
            len(unmatched_gts)  # Unmatched gts
            + (
                len(matches) - i - 1
            )  # Matched preds with score < current prediction's
        )
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    pass
