import numpy as np

from discopat.core import Array


def to_np_array(list_or_tensor: Array) -> np.array:
    """Cast to numpy array."""
    if type(list_or_tensor) is list:
        return np.array(list_or_tensor)
    return list_or_tensor.detach().numpy()


def match_groundtruths_and_predictions(
    groundtruths: list,
    predictions: list,
    scores: list,
    threshold: float,
    localization_criterion: str,
) -> tuple[int, list[tuple[float, float]]]:
    """Match GTs and predictions on an image in the dataset.

    Args:
        groundtruths: list of groundtruths, boxes [x1, y1, x2, y2],
        predictions: list of predictions, boxes [x1, y1, x2, y2],
        scores: list of confidence score, same length as predictions
        threshold: threshold for the localization metric,
        localization_criterion: metric used to assess the fit between GTs and predictions.

    Returns:
        A report for the considered image, containing:
            - The total number of groundtruths,
            - For each pred, a tuple (score, is_tp).

    """
    make_matching_matrix = {
        "iou": compute_iou_matrix,
        "iomean": compute_iomean_matrix,
        "center_distance": compute_center_distance_matrix,
    }[localization_criterion]

    # Sort predictions by score descending
    predictions = to_np_array(predictions)
    scores = to_np_array(scores)
    order = np.argsort(-scores)
    predictions = predictions[order]
    scores = scores[order]

    # Track matches
    gt_matched = np.zeros(len(groundtruths), dtype=bool)
    tps = np.zeros(len(predictions))

    for i, pred in enumerate(predictions):
        # Find best matching GT
        loc_scores = [make_matching_matrix(pred, gt) for gt in groundtruths]
        best_gt = int(np.argmax(loc_scores))
        best_loc = loc_scores[best_gt]
        if best_loc >= threshold and not gt_matched[best_gt]:
            tps[i] = 1
            gt_matched[best_gt] = True

    return len(groundtruths), np.array(list(zip(scores, tps))).reshape(-1, 2)


def compute_iou_matrix(
    groundtruths: np.array, predictions: np.array
) -> np.array:
    """Compute IoU matrix between predicted and GT boxes (both [N, 4] arrays in xyxy format).

    Args:
        groundtruths: (N_gt, 4)
        predictions: (N_pred, 4)

    Returns:
        (N_pred, N_gt) matrix of pairwise IoUs

    """
    if len(predictions) == 0 or len(groundtruths) == 0:
        return np.zeros((len(predictions), len(groundtruths)), dtype=np.float32)

    # Pred boxes
    px1, py1, px2, py2 = (
        predictions[:, 0][:, None],
        predictions[:, 1][:, None],
        predictions[:, 2][:, None],
        predictions[:, 3][:, None],
    )
    # GT boxes
    gx1, gy1, gx2, gy2 = (
        groundtruths[:, 0][None, :],
        groundtruths[:, 1][None, :],
        groundtruths[:, 2][None, :],
        groundtruths[:, 3][None, :],
    )

    # Areas
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)

    # Intersection boxes
    inter_x1 = np.maximum(px1, gx1)
    inter_y1 = np.maximum(py1, gy1)
    inter_x2 = np.minimum(px2, gx2)
    inter_y2 = np.minimum(py2, gy2)

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    # Union
    union = area_p + area_g - inter_area
    return inter_area / np.clip(union, 1e-7, None)


def compute_iomean_matrix(
    groundtruths: np.array, predictions: np.array
) -> np.array:
    """Compute IoMean matrix between predicted and GT boxes (both [N, 4] arrays in xyxy format).

    Args:
        groundtruths: (N_gt, 4)
        predictions: (N_pred, 4)

    Returns:
        (N_pred, N_gt) matrix of pairwise IoUs

    """
    if len(predictions) == 0 or len(groundtruths) == 0:
        return np.zeros((len(predictions), len(groundtruths)), dtype=np.float32)

    px1, py1, px2, py2 = (
        predictions[:, 0][:, None],
        predictions[:, 1][:, None],
        predictions[:, 2][:, None],
        predictions[:, 3][:, None],
    )
    gx1, gy1, gx2, gy2 = (
        groundtruths[:, 0][None, :],
        groundtruths[:, 1][None, :],
        groundtruths[:, 2][None, :],
        groundtruths[:, 3][None, :],
    )

    # Areas
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)

    # Intersection boxes
    inter_x1 = np.maximum(px1, gx1)
    inter_y1 = np.maximum(py1, gy1)
    inter_x2 = np.minimum(px2, gx2)
    inter_y2 = np.minimum(py2, gy2)

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    # Mean
    mean_area = (area_p + area_g) / 2
    return inter_area / np.clip(mean_area, 1e-7, None)


def compute_center_distance_matrix(
    groundtruths: np.array, predictions: np.array
) -> np.array:
    """Compute pairwise Euclidean distance between box centers.

    Args:
        groundtruths: (N_gt, 4)
        predictions: (N_pred, 4)

    Returns:
        (N_pred, N_gt) matrix of pairwise center distances.

    """
    if len(predictions) == 0 or len(groundtruths) == 0:
        return np.zeros((len(predictions), len(groundtruths)), dtype=np.float32)

    # Centers
    pcx = ((predictions[:, 0] + predictions[:, 2]) / 2)[:, None]
    pcy = ((predictions[:, 1] + predictions[:, 3]) / 2)[:, None]
    gcx = ((groundtruths[:, 0] + groundtruths[:, 2]) / 2)[None, :]
    gcy = ((groundtruths[:, 1] + groundtruths[:, 3]) / 2)[None, :]

    return np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
