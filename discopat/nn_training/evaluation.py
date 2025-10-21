import numpy as np
from torch.utils.data import DataLoader

from discopat.core import NeuralNet
from discopat.metrics import compute_iomean, compute_iou


def match_gts_and_preds(
    groundtruths: list,
    predictions: list,
    threshold: float,
    localization_criterion: str,
) -> tuple[int, list[tuple[float, float]]]:
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

    return len(groundtruths), np.array(list(zip(scores, tps))).reshape(-1, 2)


def compute_ap(
    model: NeuralNet,
    data_loader: DataLoader,
    threshold: float,
    localization_criterion: str,
) -> float:
    """Compute the Average Precision (AP) for a given localization threshold.

    Args:
        model: the neural network to be evaluated,
        data_loader: the evaluation dataloader,
        threshold: localization threshold,
        localization_criterion: metric used to match groundtruths and preds.

    Returns:
        The AP.

    """
    num_groundtruths = 0
    big_tp_vector = np.empty((0, 2))
    for images, targets in data_loader:
        outputs = model(images)
        for target, output in zip(targets, outputs):
            num_gts, tp_vector = match_gts_and_preds(
                target["boxes"],
                output["boxes"],
                threshold=threshold,
                localization_criterion=localization_criterion,
            )
            num_groundtruths += num_gts
            big_tp_vector = np.concat((big_tp_vector, tp_vector))

    if num_groundtruths == 0:
        return 0

    # Sort the TP vector by decreasing prediction score over the whole dataset
    big_tp_vector = np.array(
        big_tp_vector, dtype=[("score", float), ("is_tp", float)]
    )
    big_tp_vector = np.sort(big_tp_vector, order="score")

    # Cumulative sums
    tp_cumulative = np.cumsum(tp_vector[:, 1])
    fp_cumulative = np.cumsum(1 - tp_vector[:, 1])

    # Prepend zeros for the case score_threshold=1
    tp_cum = np.concatenate([[0], tp_cumulative])
    fp_cum = np.concatenate([[0], fp_cumulative])

    recall = tp_cum / num_groundtruths
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)

    # Ensure precision is non-increasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Compute area under curve (AP)
    return np.trapezoid(precision, recall)


def evaluate(
    model: NeuralNet, data_loader: DataLoader, localization_criterion: str
) -> dict[str, float]:
    """Evaluate a model on a data loader.

    Args:
        model: the neural network to be evaluated,
        data_loader: the evaluation dataloader,
        localization_criterion: metric used for GT-pred matching.

    Returns:
        A dict containing the name and values of the following metrics:
        AP50, AP[50:95:05].

    """
    ap_dict = {
        f"AP{int(100 * threshold)}": compute_ap(
            model,
            data_loader,
            threshold=threshold,
            localization_criterion=localization_criterion,
        )
        for threshold in [0.5, 1.0, 0.05]
    }
    return {"AP50": ap_dict["AP50"], "AP": np.mean(list(ap_dict.values()))}
