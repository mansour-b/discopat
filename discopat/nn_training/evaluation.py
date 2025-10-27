import time

import numpy as np

from discopat.core import Array, ComputingDevice, DataLoader, NeuralNet
from discopat.metrics import compute_iomean, compute_iou


def to_np_array(list_or_tensor: Array) -> np.array:
    """Cast to numpy array."""
    if type(list_or_tensor) is list:
        return np.array(list_or_tensor)
    return list_or_tensor.detach().numpy()


def box_iou_matrix(preds, gts):
    """Compute IoU matrix between predicted and GT boxes (both [N, 4] arrays in xyxy format).

    Args:
        preds: (N_pred, 4)
        gts:   (N_gt, 4)

    Returns:
        ious: (N_pred, N_gt) matrix of pairwise IoUs

    """
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)

    # Pred boxes
    px1, py1, px2, py2 = (
        preds[:, 0][:, None],
        preds[:, 1][:, None],
        preds[:, 2][:, None],
        preds[:, 3][:, None],
    )
    # GT boxes
    gx1, gy1, gx2, gy2 = (
        gts[:, 0][None, :],
        gts[:, 1][None, :],
        gts[:, 2][None, :],
        gts[:, 3][None, :],
    )

    # Intersection box
    inter_x1 = np.maximum(px1, gx1)
    inter_y1 = np.maximum(py1, gy1)
    inter_x2 = np.minimum(px2, gx2)
    inter_y2 = np.minimum(py2, gy2)

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    # Areas
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)

    # Union
    union = area_p + area_g - inter_area
    return inter_area / np.clip(union, 1e-7, None)


def box_iomean_matrix(preds, gts):
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)

    # Reuse intersection logic from IoU
    iou = box_iou_matrix(preds, gts)

    # To get IoMean, recompute areas
    px1, py1, px2, py2 = (
        preds[:, 0][:, None],
        preds[:, 1][:, None],
        preds[:, 2][:, None],
        preds[:, 3][:, None],
    )
    gx1, gy1, gx2, gy2 = (
        gts[:, 0][None, :],
        gts[:, 1][None, :],
        gts[:, 2][None, :],
        gts[:, 3][None, :],
    )

    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    inter_area = iou * (
        area_p + area_g - iou * (area_p + area_g - (area_p + area_g))
    )  # or recompute directly

    mean_area = (area_p + area_g) / 2
    return inter_area / np.clip(mean_area, 1e-7, None)


def box_center_distance_matrix(preds, gts):
    """Compute pairwise Euclidean distance between box centers."""
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)

    # Centers
    pcx = ((preds[:, 0] + preds[:, 2]) / 2)[:, None]
    pcy = ((preds[:, 1] + preds[:, 3]) / 2)[:, None]
    gcx = ((gts[:, 0] + gts[:, 2]) / 2)[None, :]
    gcy = ((gts[:, 1] + gts[:, 3]) / 2)[None, :]

    return np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)


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


def make_localization_matrix(
    groundtruths: list,
    predictions: list,
    scores: list,
    threshold: float,
    localization_criterion: str,
) -> Array:
    """Compute IoU/IoMean/whatever matrix between GTs and preds on one sample.

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
    num_groundtruths = 0
    big_tp_vector = np.empty((0, 2))
    for _, targets in data_loader:
        outputs = [prediction_dict[t["image_id"]] for t in targets]
        print("    Transfering outputs to CPU...")
        start = time.process_time()
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        end = time.process_time()
        print(f"    Done in {end - start:.2f} seconds.")
        print()
        for target, output in zip(targets, outputs):
            print("        Matching...")
            start = time.process_time()
            num_gts, tp_vector = match_gts_and_preds(
                groundtruths=target["boxes"],
                predictions=output["boxes"],
                scores=output["scores"],
                threshold=threshold,
                localization_criterion=localization_criterion,
            )
            end = time.process_time()
            print(f"        Done in {end - start:.2f} seconds.")
            num_groundtruths += num_gts
            big_tp_vector = np.concat((big_tp_vector, tp_vector))
        print()
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
    print()
    print("===")
    print("Compute predictions...")
    start = time.process_time()
    prediction_dict = {
        t["image_id"]: pred
        for images, targets in data_loader
        for pred, t in zip(
            model([img.to(device).float() for img in images]), targets
        )
    }
    end = time.process_time()
    print(f"Done in {end - start:.2f} seconds.")
    print()
    print("Build AP dict...")
    start = time.process_time()
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
    end = time.process_time()
    print(f"Done in {end - start:.2f} seconds.")
    print("===")
    print()
    return {"AP50": ap_dict["AP50"], "AP": np.mean(list(ap_dict.values()))}
