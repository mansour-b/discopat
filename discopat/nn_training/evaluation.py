import time

import numpy as np

from discopat.core import Array, ComputingDevice, DataLoader, NeuralNet
from discopat.metrics import compute_iomean, compute_iou


def to_np_array(list_or_tensor) -> np.array:
    """Cast to numpy array."""
    if type(list_or_tensor) is list:
        return np.array(list_or_tensor)
    return list_or_tensor.detach().numpy()


def match_gts_and_preds(
    groundtruths: list,
    predictions: list,
    scores: list,
    threshold: float,
    localization_criterion: str,
) -> tuple[int, list[tuple[float, float]]]:
    """Match GTs and preds on an image in the dataset.

    Args:
        groundtruths: list of groundtruths, boxes [x1, y1, x2, y2],
        predictions: list of predictions, boxes [x1, y1, x2, y2],
        scores: list of confidence score, same length as predictions
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
        loc_scores = [localizing_function(pred, gt) for gt in groundtruths]
        best_gt = int(np.argmax(loc_scores))
        best_loc = loc_scores[best_gt]
        if best_loc >= threshold and not gt_matched[best_gt]:
            tps[i] = 1
            gt_matched[best_gt] = True

    return len(groundtruths), np.array(list(zip(scores, tps))).reshape(-1, 2)


def compute_ap(
    prediction_dict: dict[str, Array],
    data_loader: DataLoader,
    threshold: float,
    localization_criterion: str,
    device: ComputingDevice,
) -> float:
    """Compute the Average Precision (AP) for a given localization threshold.

    Args:
        model: the neural network to be evaluated,
        data_loader: the evaluation dataloader,
        threshold: localization threshold,
        localization_criterion: metric used to match groundtruths and preds,
        device: computing device on which the model is stored.

    Returns:
        The AP.

    """
    print("===")
    print(f"Compute AP@{threshold:.2f}")
    print()
    num_groundtruths = 0
    big_tp_vector = np.empty((0, 2))
    for _, targets in data_loader:
        print("Start computing outputs...")
        start = time.process_time()
        outputs = [prediction_dict[t["image_id"]] for t in targets]
        end = time.process_time()
        print(f"Done after {end - start:.2f} seconds.")
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        for target, output in zip(targets, outputs):
            num_gts, tp_vector = match_gts_and_preds(
                groundtruths=target["boxes"],
                predictions=output["boxes"],
                scores=output["scores"],
                threshold=threshold,
                localization_criterion=localization_criterion,
            )
            num_groundtruths += num_gts
            big_tp_vector = np.concat((big_tp_vector, tp_vector))
    if num_groundtruths == 0:
        return 0

    # Sort the TP vector by decreasing prediction score over the whole dataset
    big_tp_vector = big_tp_vector[np.argsort(-big_tp_vector[:, 0])]

    # Cumulative sums
    tp_cumulative = np.cumsum(big_tp_vector[:, 1])
    fp_cumulative = np.cumsum(1 - big_tp_vector[:, 1])

    # Prepend zeros for the case score_threshold=1
    tp_cum = np.concatenate([[0], tp_cumulative])
    fp_cum = np.concatenate([[0], fp_cumulative])

    recall = tp_cum / num_groundtruths
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)

    # Ensure precision is non-increasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Compute area under curve (AP)
    print()
    return np.trapezoid(precision, recall)


def evaluate(
    model: NeuralNet,
    data_loader: DataLoader,
    localization_criterion: str,
    device: ComputingDevice,
) -> dict[str, float]:
    """Evaluate a model on a data loader.

    Args:
        model: the neural network to be evaluated,
        data_loader: the evaluation dataloader,
        localization_criterion: metric used for GT-pred matching,
        device: computing device on which the model is stored.

    Returns:
        A dict containing the name and values of the following metrics:
        AP50, AP[50:95:05].

    """
    model.eval()
    prediction_dict = {
        t["image_id"]: pred
        for images, targets in data_loader
        for pred, t in zip(
            model([img.to(device).float() for img in images]), targets
        )
    }
    ap_dict = {
        f"AP{int(100 * threshold)}": compute_ap(
            prediction_dict,
            data_loader,
            threshold=threshold,
            localization_criterion=localization_criterion,
            device=device,
        )
        for threshold in np.arange(0.5, 1.0, 0.05)
    }
    return {"AP50": ap_dict["AP50"], "AP": np.mean(list(ap_dict.values()))}
