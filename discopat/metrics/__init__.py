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


def compute_ap(groundtruths: list, predictions: list) -> float:
    iou_matrix = compute_iou_matrix(groundtruths, predictions)
